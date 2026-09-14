# -*- coding: utf-8 -*-
"""
公共流程编排：缓存资格 -> 按需采集 -> 校验 -> 发布 -> 返回数据及使用说明。

功能或目的：
    实现 Issue #5 §3 的唯一公共入口。三个互不等价的概念在这里被分别判定：
        - 刷新间隔（何时再查来源）
        - 陈旧上限（刷新失败后旧数据还能不能用）
        - 范围 / 身份 / 质量（分别独立验证，任何一项不满足都不能被“文件刚创建”放行）

    关键不变量：
        - `last_verified_at` 只在“真实来源结果成功解析并按合同核验”后推进；
          本地读取、报告生成、失败重试都不推进。
        - 缓存复评不会把旧检查范围当作新请求已通过：只有当上次检查范围覆盖本次请求范围时，
          上次的 `scan_complete` 才可继承，否则本次请求范围内存在“从未被扫描过”的部分。
        - 任何提交前的失败都不改变旧正式状态；发布是唯一提交点。

调用关系：
    - `acquire_dataset()` 为单数据集入口，被 CLI、trading shim 与测试直接调用。
    - `run_task()` 为多数据集任务入口：可选项失败可继续，所有必需项满足后才算成功。
    - 依赖 `rbt_fin.storage`（锁与提交）、`rbt_fin.validation`（合同）、`Dataset.acquire`（来源）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import __about__
from .contracts import (
    COVERAGE_COMPLETE,
    MODE_AUTO,
    MODE_FORCE,
    MODE_OFFLINE,
    REVISION_ACCEPT_LATEST,
    REVISION_REJECT,
    AcquireContext,
    AcquireResult,
    CoverageReport,
    DataRequest,
    DataResult,
    Dataset,
)
from .errors import (
    ConfigError,
    CoverageError,
    OfflineUnavailableError,
    RevisionConflictError,
    StaleCacheError,
)
from .storage import DatasetStore, Manifest
from .timeutils import require_utc


@dataclass(frozen=True)
class _CacheState:
    """
    一次缓存复评的中间结果（只在 `acquire_dataset` 内部使用）。

    Attributes:
        manifest: 当前正式 manifest；None 表示尚未初始化。
        frame: 当前正式数据；None 表示尚未初始化或身份不匹配。
        coverage: 针对**本次请求范围**重新计算的覆盖报告；None 表示无可复评数据。
        identity_match: 身份与 schema/transform 版本是否与当前数据集定义一致。
        satisfies_request: 覆盖是否满足本次请求的严格程度。
        age: 距离上次成功核对来源的时长；None 表示无 manifest。
    """

    manifest: Optional[Manifest]
    frame: Optional[pd.DataFrame]
    coverage: Optional[CoverageReport]
    identity_match: bool
    satisfies_request: bool
    age: Optional[dt.timedelta]


def _identity_matches(*, manifest: Manifest, dataset: Dataset) -> bool:
    """
    判断缓存的身份与格式版本是否仍与当前数据集定义一致。

    为什么身份不一致就不能复用：
        §4.1 明确“不能仅靠文件名防止混写”。交易所/市场/交易对/价格类型/频率/单位
        任一项变化，旧内容就不再是本数据集的数据；schema/转换版本变化同理。
    """
    return (
        dict(manifest.identity) == dict(dataset.identity)
        and manifest.schema_version == dataset.schema_version
        and manifest.transform_version == dataset.transform_version
        and list(manifest.columns) == list(dataset.columns)
        and manifest.time_column == dataset.time_column
    )


def _previous_scan_covers(*, manifest: Manifest, request: DataRequest) -> bool:
    """
    判断上一次覆盖检查的范围是否完整包住本次请求范围。

    为什么必须判断：
        若本次请求比上次检查范围更宽，超出的部分**从未被来源扫描过**，
        此时继承上次的 `scan_complete=True` 就等于“把旧检查范围当新请求已通过”。
    """
    coverage = manifest.coverage
    checked_start = pd.Timestamp(coverage["checked_start"])
    checked_end = pd.Timestamp(coverage["checked_end_exclusive"])
    return checked_start <= request.start and checked_end >= request.end_exclusive


def _evaluate_cache(
    *, store: DatasetStore, dataset: Dataset, request: DataRequest, now_utc: pd.Timestamp,
    verify_content: bool,
) -> _CacheState:
    """
    读取并复评当前缓存：范围、身份、质量三项分别判定。

    Args:
        store: 数据集缓存目录。
        dataset: 数据集定义。
        request: 本次请求。
        now_utc: 本次任务固定的当前时刻。
        verify_content: 读取内容文件时是否重算哈希。

    Returns:
        `_CacheState`。
    """
    manifest = store.read_manifest()
    if manifest is None:
        return _CacheState(
            manifest=None, frame=None, coverage=None,
            identity_match=False, satisfies_request=False, age=None,
        )

    age = now_utc - require_utc(pd.Timestamp(manifest.last_verified_at), arg_name="last_verified_at")
    if not _identity_matches(manifest=manifest, dataset=dataset):
        # 身份/版本不匹配：不读内容，也不复用；由刷新分支重新采集并覆盖。
        return _CacheState(
            manifest=manifest, frame=None, coverage=None,
            identity_match=False, satisfies_request=False, age=age,
        )

    frame = store.read_frame(manifest, verify_content=verify_content)
    # 复评时的 scan_complete 只能来自“上一次真的扫过本次请求范围”的事实。
    inherited_scan_complete = bool(manifest.coverage.get("scan_complete", False)) and _previous_scan_covers(
        manifest=manifest, request=request
    )
    scan_note = "缓存复评（零网络请求）"
    if not inherited_scan_complete:
        scan_note += "；上次扫描未覆盖本次请求范围或本身未完成，不继承完整性证据"
    coverage = dataset.validate(
        frame,
        dataset=dataset,
        request=request,
        scan_complete=inherited_scan_complete,
        scan_note=scan_note,
    )
    satisfies = coverage.status == COVERAGE_COMPLETE if request.require_full_coverage else True
    return _CacheState(
        manifest=manifest, frame=frame, coverage=coverage,
        identity_match=True, satisfies_request=satisfies, age=age,
    )


def _check_revision(
    *, dataset: Dataset, request: DataRequest, cached: Optional[pd.DataFrame], acquired: pd.DataFrame
) -> Tuple[int, str]:
    """
    在发布前比较新旧内容，执行显式的修订/冲突策略。

    三种情形：
        - 同主键相同内容：无事发生（去重即可）。
        - 同主键不同内容：`reject` 策略直接失败；`accept_latest` 记录变更数量与范围。
        - 历史缩短：请求范围内原有的主键在新数据中消失 -> 保留旧正式数据并报错。

    Args:
        dataset: 数据集定义（提供 revision_policy）。
        request: 本次请求，限定比较窗口。
        cached: 旧正式数据；None 表示首次初始化。
        acquired: 本次采集得到的完整新内容。

    Returns:
        `(changed_rows, note)`。

    Raises:
        RevisionConflictError: 未经解释的内容冲突或历史缩短。
    """
    if cached is None or len(cached) == 0:
        return 0, ""
    window = (cached.index >= request.start) & (cached.index < request.end_exclusive)
    old_window = cached[window]
    if len(old_window) == 0:
        return 0, ""

    disappeared = old_window.index.difference(acquired.index)
    if len(disappeared) > 0:
        raise RevisionConflictError(
            f"{dataset.dataset_id}: 请求范围内有 {len(disappeared)} 条既有记录在新结果中消失"
            f"（示例: {[str(t) for t in disappeared[:5]]}）。"
            "旧正式数据保持不变；仅按显式窗口请求或来源证据才允许缩短历史。"
        )

    common = old_window.index.intersection(acquired.index)
    if len(common) == 0:
        return 0, ""
    columns = list(dataset.value_columns)
    old_values = old_window.loc[common, columns]
    new_values = acquired.loc[common, columns]
    differs = ~(old_values.eq(new_values).all(axis=1))
    changed = int(differs.sum())
    if changed == 0:
        return 0, ""
    changed_index = common[differs.to_numpy()]
    if dataset.revision_policy == REVISION_REJECT:
        raise RevisionConflictError(
            f"{dataset.dataset_id}: 同主键出现 {changed} 条内容不同的记录"
            f"（范围 {changed_index.min()} .. {changed_index.max()}，示例: "
            f"{[str(t) for t in changed_index[:5]]}）。"
            "该数据集声明为不可修订，拒绝让未经解释的冲突覆盖正式数据。"
        )
    return changed, (
        f"修订：{changed} 条记录内容变化，范围 {changed_index.min()} .. {changed_index.max()}"
        f"（policy={REVISION_ACCEPT_LATEST}，已保留上一份内容文件）。"
    )


def _build_result(
    *,
    dataset: Dataset,
    request: DataRequest,
    manifest: Manifest,
    frame: pd.DataFrame,
    store: DatasetStore,
    coverage: CoverageReport,
    used_cache: bool,
    refresh_failed: bool,
    degradations: Sequence[str],
    errors: Sequence[str],
) -> DataResult:
    """把 manifest + 数据 + 本次覆盖复评组装成公共返回结构。"""
    return DataResult(
        dataset_id=dataset.dataset_id,
        frame=frame,
        content_path=str(store.content_path(manifest.content_file)),
        content_sha256=manifest.content_sha256,
        identity=dict(manifest.identity),
        schema_version=manifest.schema_version,
        transform_version=manifest.transform_version,
        requested_start=request.start,
        requested_end_exclusive=request.end_exclusive,
        coverage=coverage,
        fetched_at=pd.Timestamp(manifest.fetched_at),
        last_verified_at=pd.Timestamp(manifest.last_verified_at),
        used_cache=used_cache,
        refresh_failed=refresh_failed,
        point_in_time_verified=bool(manifest.point_in_time_verified),
        degradations=tuple(degradations),
        errors=tuple(errors),
    )


def acquire_dataset(
    *,
    dataset: Dataset,
    request: DataRequest,
    store_root: Path,
    now_utc: pd.Timestamp,
    http: Optional[Any] = None,
    credentials: Optional[Mapping[str, str]] = None,
    local_inputs: Optional[Mapping[str, Any]] = None,
    code_version: Optional[str] = None,
    phase_hook: Optional[Callable[[str], None]] = None,
    verify_content: bool = True,
) -> DataResult:
    """
    公共单数据集入口：取得、校验、缓存、保存、重新读取并返回使用说明。

    Args:
        dataset: 数据集定义。
        request: 本次请求（含固定的 `[start, end_exclusive)` 与严格程度）。
        store_root: 缓存根目录；显式传入，不扫描 cwd。
        now_utc: 本次任务固定的唯一截止时刻（tz-aware UTC）。
        http: HTTP 客户端；`offline` 模式或本地数据集可为 None。
        credentials: 显式凭据；其值不会进入 params/manifest/日志。
        local_inputs: 本地导入所需输入（例如 csv_path）。
        code_version: 代码版本标识；默认取包版本。
        phase_hook: 发布流程命名同步点回调（受控中断测试用）。
        verify_content: 读取缓存内容文件时是否重算哈希（默认 True，缓存损坏必须拒绝）。

    Returns:
        DataResult。

    Raises:
        ConfigError: 数据集/请求组合非法（例如 offline 却要求联网数据集刷新）。
        CoverageError: 严格请求所需覆盖未达成，或要求历史当时可得但数据集未证明。
        OfflineUnavailableError: offline 模式下缓存不满足请求。
        StaleCacheError: 刷新失败且旧缓存已超过陈旧上限。
    """
    now = require_utc(now_utc, arg_name="now_utc")
    resolved_code_version = code_version if code_version is not None else __about__.__version__
    credentials = dict(credentials or {})
    local_inputs = dict(local_inputs or {})

    # 历史当时可得性是可执行边界：未证明就不能被当作 point-in-time 数据使用。
    if request.require_point_in_time and not dataset.point_in_time_verified:
        raise CoverageError(
            f"{dataset.dataset_id}: 请求要求历史当时可得（point-in-time），"
            "但该数据集未证明此性质；本地采集时间不证明此前历史可得性。"
        )

    store = DatasetStore(root=Path(store_root), dataset_id=dataset.dataset_id)
    with store.lock():
        # 取锁后必须重新读取当前状态：第二写者不能基于取锁前的旧快照更新。
        state = _evaluate_cache(
            store=store, dataset=dataset, request=request, now_utc=now,
            verify_content=verify_content,
        )

        if request.mode == MODE_OFFLINE:
            if state.frame is None or state.coverage is None or not state.satisfies_request:
                raise OfflineUnavailableError(
                    f"{dataset.dataset_id}: offline 模式下缓存不满足请求 "
                    f"[{request.start}, {request.end_exclusive})；"
                    f"identity_match={state.identity_match}, "
                    f"coverage={None if state.coverage is None else state.coverage.status}。"
                )
            return _build_result(
                dataset=dataset, request=request, manifest=state.manifest, frame=state.frame,
                store=store, coverage=state.coverage, used_cache=True, refresh_failed=False,
                degradations=("offline 模式：本次未发起任何网络请求。",), errors=(),
            )

        if request.mode == MODE_AUTO and state.satisfies_request and state.coverage is not None:
            # 只有“本次请求范围已完整且该来源不可修订”才允许彻底跳过时钟：
            # 固定历史区间可离线使用，不要求追到现在（§3.2）。
            closed_history = (
                state.coverage.status == COVERAGE_COMPLETE
                and dataset.revision_policy == REVISION_REJECT
            )
            within_refresh_window = (
                state.age is not None and state.age < dataset.default_refresh_interval
            )
            if closed_history or within_refresh_window:
                reason = "请求范围已完整且来源不可修订" if closed_history else "未到重新检查来源的时机"
                return _build_result(
                    dataset=dataset, request=request, manifest=state.manifest, frame=state.frame,
                    store=store, coverage=state.coverage, used_cache=True, refresh_failed=False,
                    degradations=(f"缓存命中（{reason}）：本次未发起任何网络请求。",), errors=(),
                )

        # ---- 刷新分支 ----
        try:
            result = _refresh_and_publish(
                dataset=dataset, request=request, store=store, state=state, now=now,
                http=http, credentials=credentials, local_inputs=local_inputs,
                code_version=resolved_code_version, phase_hook=phase_hook,
            )
        except Exception as exc:  # 刷新失败：是否允许降级由请求显式声明，且必须如实报告。
            return _handle_refresh_failure(
                dataset=dataset, request=request, store=store, state=state, error=exc
            )
        return result


def _refresh_and_publish(
    *,
    dataset: Dataset,
    request: DataRequest,
    store: DatasetStore,
    state: _CacheState,
    now: pd.Timestamp,
    http: Optional[Any],
    credentials: Mapping[str, str],
    local_inputs: Mapping[str, Any],
    code_version: str,
    phase_hook: Optional[Callable[[str], None]],
) -> DataResult:
    """
    执行一次真实刷新并发布。

    只有走完“采集 -> 校验通过 -> 发布成功”才推进 `last_verified_at`。

    Raises:
        CoverageError: 严格请求的覆盖未达成（发布不会发生，旧正式状态不变）。
    """
    context = AcquireContext(
        dataset=dataset,
        request=request,
        now_utc=now,
        http=http,
        cached_frame=state.frame,
        credentials=credentials,
        local_inputs=local_inputs,
        force_refresh=request.mode == MODE_FORCE,
    )
    acquired: AcquireResult = dataset.acquire(context)
    if not isinstance(acquired, AcquireResult):
        raise ConfigError(
            f"{dataset.dataset_id}: 采集函数必须返回 AcquireResult，实际为 {type(acquired)!r}"
        )

    revision_changed, revision_note = _check_revision(
        dataset=dataset, request=request, cached=state.frame, acquired=acquired.frame
    )
    coverage = dataset.validate(
        acquired.frame,
        dataset=dataset,
        request=request,
        scan_complete=acquired.scan_complete,
        scan_note=" | ".join(note for note in (acquired.scan_note, revision_note) if note),
    )
    if request.require_full_coverage and coverage.status != COVERAGE_COMPLETE:
        raise CoverageError(
            f"{dataset.dataset_id}: 严格请求要求完整覆盖，实际 status={coverage.status}；"
            f"缺失 head={coverage.missing_head} interior={coverage.missing_interior} "
            f"tail={coverage.missing_tail}，示例 {list(coverage.missing_examples)}；"
            f"说明: {coverage.note}"
        )

    frame = acquired.frame.sort_index()
    actual_start = None if len(frame) == 0 else frame.index.min().isoformat()
    actual_end = None if len(frame) == 0 else frame.index.max().isoformat()
    manifest = store.publish(
        frame=frame,
        columns=dataset.columns,
        time_column=dataset.time_column,
        previous=state.manifest,
        phase_hook=phase_hook,
        manifest_fields={
            "identity": dict(dataset.identity),
            "params": dict(dataset.params),
            "time_meaning": dataset.time_meaning,
            "schema_version": dataset.schema_version,
            "transform_version": dataset.transform_version,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "coverage": coverage.to_manifest_dict(),
            "fetched_at": now.isoformat(),
            # 只有真实来源结果成功解析并按合同核验后才推进核对时间。
            "last_verified_at": now.isoformat(),
            "code_version": code_version,
            "point_in_time_verified": bool(dataset.point_in_time_verified),
            "note": dataset.remark,
            "extra": {
                "requests_used": int(acquired.requests_used),
                "revision_changed_rows": int(revision_changed),
                "scan_note": acquired.scan_note,
            },
        },
    )
    # 重新按 manifest 读回：证明“写下去的东西能被读回来”，而不是只信任内存对象。
    published = store.read_frame(manifest, verify_content=True)
    return _build_result(
        dataset=dataset, request=request, manifest=manifest, frame=published, store=store,
        coverage=coverage, used_cache=False, refresh_failed=False,
        degradations=(revision_note,) if revision_note else (), errors=(),
    )


def _handle_refresh_failure(
    *,
    dataset: Dataset,
    request: DataRequest,
    store: DatasetStore,
    state: _CacheState,
    error: Exception,
) -> DataResult:
    """
    刷新失败后的显式处置：按配置返回失败，或使用仍合格的旧数据并如实标注降级。

    Raises:
        原始异常: 未允许降级，或无可用缓存。
        StaleCacheError: 允许降级但旧缓存已超过陈旧上限。
    """
    message = f"{type(error).__name__}: {error}"
    if not request.accept_stale_on_refresh_failure:
        raise error
    if state.frame is None or state.coverage is None or not state.satisfies_request:
        # 严格请求所需范围/质量本就不满足：不能因为“缓存文件存在”就放行。
        raise error
    assert request.max_staleness is not None  # DataRequest 已在构造时保证二者同时提供
    if state.age is None or state.age > request.max_staleness:
        raise StaleCacheError(
            f"{dataset.dataset_id}: 刷新失败且缓存已陈旧 {state.age}，"
            f"超过允许上限 {request.max_staleness}；原始失败: {message}"
        ) from error
    return _build_result(
        dataset=dataset, request=request, manifest=state.manifest, frame=state.frame, store=store,
        coverage=state.coverage, used_cache=True, refresh_failed=True,
        degradations=(
            f"刷新失败，使用仍合格的旧缓存（已陈旧 {state.age}，上限 {request.max_staleness}）；"
            "核对时间未推进。",
        ),
        errors=(message,),
    )


@dataclass(frozen=True)
class TaskOutcome:
    """
    多数据集任务的结果。

    Attributes:
        results: 成功项的 `dataset_id -> DataResult`。
        failures: 失败项的 `dataset_id -> 错误描述`。
        required_failures: 其中属于必需项的失败 id。
        succeeded: 所有必需项是否都已满足请求。
    """

    results: Mapping[str, DataResult]
    failures: Mapping[str, str]
    required_failures: Sequence[str]
    succeeded: bool


def run_task(
    *,
    items: Sequence[Tuple[Dataset, DataRequest]],
    store_root: Path,
    now_utc: pd.Timestamp,
    http: Optional[Any] = None,
    credentials: Optional[Mapping[str, str]] = None,
    local_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    code_version: Optional[str] = None,
) -> TaskOutcome:
    """
    一次多数据集任务：可选项失败可继续，所有必需项满足后才算成功。

    为什么固定一次 `now_utc`：
        §3.1 要求“一次任务只确定一次截止时刻”。若每个数据集各自取当前时间，
        同一批次内不同数据集会落在不同截止点上，批次输入就不再可复现。

    Args:
        items: `(dataset, request)` 序列。
        store_root: 缓存根目录。
        now_utc: 本次任务固定的截止时刻。
        http: HTTP 客户端。
        credentials: 显式凭据。
        local_inputs: `dataset_id -> 本地输入映射`。
        code_version: 代码版本标识。

    Returns:
        TaskOutcome；`succeeded=False` 时调用方不得让下游（例如训练）继续。
    """
    local_inputs = dict(local_inputs or {})
    results: Dict[str, DataResult] = {}
    failures: Dict[str, str] = {}
    required_failures: List[str] = []
    for dataset, request in items:
        try:
            results[dataset.dataset_id] = acquire_dataset(
                dataset=dataset,
                request=request,
                store_root=store_root,
                now_utc=now_utc,
                http=http,
                credentials=credentials,
                local_inputs=local_inputs.get(dataset.dataset_id, {}),
                code_version=code_version,
            )
        except Exception as exc:
            failures[dataset.dataset_id] = f"{type(exc).__name__}: {exc}"
            if request.required:
                required_failures.append(dataset.dataset_id)
    return TaskOutcome(
        results=results,
        failures=failures,
        required_failures=tuple(required_failures),
        succeeded=not required_failures,
    )
