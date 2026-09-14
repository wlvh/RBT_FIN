# -*- coding: utf-8 -*-
"""
数据集定义、请求与结果的公共合同。

功能或目的：
    Issue #5 §3.1 / §3.2 要求把“数据集稳定规则”“本次请求参数”“公共结果”分成三个
    小型结构，并明确区分三类互不等价的时间/资格概念：
        - 刷新间隔（refresh_interval）：何时再去查来源。
        - 陈旧上限（max_staleness）：刷新失败后旧数据还能不能用。
        - 范围/身份/质量：分别独立验证，任何一项不满足都不能被“文件刚创建”放行。

调用关系：
    - `rbt_fin.registry` 构造 `Dataset`。
    - 调用方构造 `DataRequest`，交给 `rbt_fin.pipeline.acquire_dataset`。
    - `rbt_fin.pipeline` 返回 `DataResult`；`rbt_fin.storage.Manifest` 记录持久化部分。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .errors import ConfigError
from .timeutils import parse_interval, require_utc

# 覆盖状态三态：
#   complete   —— 期望范围内每个记录点都已取得并通过校验。
#   partial    —— 已验证存在缺口（头部/内部/尾部），缺口位置可报告。
#   unverified —— 无法证明完整性（来源日历未验证 / 扫描被预算或空页提前终止）。
COVERAGE_COMPLETE = "complete"
COVERAGE_PARTIAL = "partial"
COVERAGE_UNVERIFIED = "unverified"
COVERAGE_STATUSES = (COVERAGE_COMPLETE, COVERAGE_PARTIAL, COVERAGE_UNVERIFIED)

MODE_AUTO = "auto"
MODE_FORCE = "force"
MODE_OFFLINE = "offline"
REQUEST_MODES = (MODE_AUTO, MODE_FORCE, MODE_OFFLINE)

# 同主键不同内容时的处理策略（Issue #5 §4.3）。
REVISION_REJECT = "reject"          # 默认：拒绝，不让未经解释的冲突覆盖正式数据。
REVISION_ACCEPT_LATEST = "accept_latest"  # 显式声明“该来源允许修订”的数据集才可用。
REVISION_POLICIES = (REVISION_REJECT, REVISION_ACCEPT_LATEST)


@dataclass(frozen=True)
class GridSpec:
    """
    严格时间网格声明。

    Attributes:
        interval: 固定长度周期字符串（'1m' 等）；周/月周期必须单独实现，不在此表达。
        origin_note: 该网格的业务含义说明（例如 'bar 开盘时间'）。
    """

    interval: str
    origin_note: str

    def __post_init__(self) -> None:
        # 为什么在构造时解析：网格非法必须在注册表构建阶段暴露，而不是等到跑批时。
        parse_interval(self.interval)


@dataclass(frozen=True)
class Dataset:
    """
    数据集的稳定规则（与“本次请求”严格分开）。

    Attributes:
        dataset_id: 稳定标识；同时是缓存目录名。
        identity: 数据身份，至少区分交易所/市场/交易对/价格类型/频率/单位及原始或转换后含义。
                  仅靠文件名不能防止混写（Issue #5 §4.1），因此身份要进 manifest 并在读取时核对。
        params: 采集所需的有效参数（脱敏后）；参数记录与真实请求从同一份配置生成。
        acquire: 采集函数，签名 `(context: AcquireContext) -> pd.DataFrame`。
        validate: 校验函数，签名 `(frame, dataset, request) -> CoverageReport`。
        time_column: 主键时间列名（落盘后的列名；索引统一还原为该列）。
        time_meaning: 该时间的业务含义，例如 'bar_open_time' / 'funding_settlement_time'。
        value_columns: 数据列名（有序）。
        grid: 严格网格；`None` 表示非规则序列（低频/事件流），覆盖判定走非网格分支。
        default_refresh_interval: 默认重新检查来源的间隔。
        schema_version: 落盘 schema 版本；变更后旧缓存不得直接复用。
        transform_version: 转换口径版本；修正转换行为时必须递增（Issue #5 §4.4）。
        revision_policy: 同主键不同内容时的处理策略。
        requires_network: 是否需要联网；本地导入类数据集为 False。
        point_in_time_verified: 是否已证明“历史当时可得”；v1 默认 False 且是可执行边界。
        calendar_verified: 是否已验证来源发布日历；False 时覆盖状态最高只能是 unverified。
        optional_dependency: 可选依赖名；缺失时该数据集不可用但不影响核心行情能力。
        remark: 人类可读说明，进入概览输出。
    """

    dataset_id: str
    identity: Mapping[str, Any]
    params: Mapping[str, Any]
    acquire: Callable[..., pd.DataFrame]
    validate: Callable[..., "CoverageReport"]
    time_column: str
    time_meaning: str
    value_columns: Sequence[str]
    default_refresh_interval: dt.timedelta
    schema_version: str
    transform_version: str
    grid: Optional[GridSpec] = None
    revision_policy: str = REVISION_REJECT
    requires_network: bool = True
    point_in_time_verified: bool = False
    calendar_verified: bool = False
    optional_dependency: Optional[str] = None
    remark: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ConfigError("dataset_id 必须是非空字符串。")
        if self.revision_policy not in REVISION_POLICIES:
            raise ConfigError(
                f"{self.dataset_id}: revision_policy={self.revision_policy!r} 非法，"
                f"允许值: {REVISION_POLICIES}"
            )
        if not self.value_columns:
            raise ConfigError(f"{self.dataset_id}: value_columns 不能为空。")
        if self.time_column in self.value_columns:
            raise ConfigError(f"{self.dataset_id}: time_column 不能同时出现在 value_columns 中。")
        if not isinstance(self.default_refresh_interval, dt.timedelta):
            raise ConfigError(f"{self.dataset_id}: default_refresh_interval 必须是 timedelta。")
        if self.default_refresh_interval < dt.timedelta(0):
            raise ConfigError(f"{self.dataset_id}: default_refresh_interval 不能为负。")

    @property
    def columns(self) -> Tuple[str, ...]:
        """落盘 CSV 的列序：时间列在前，其后是声明顺序的数据列。"""
        return (self.time_column, *self.value_columns)


@dataclass(frozen=True)
class DataRequest:
    """
    一次具体请求（与数据集稳定规则分开表达）。

    Attributes:
        start: 所需范围左闭端点（tz-aware UTC）。
        end_exclusive: 所需范围右开端点（tz-aware UTC）。公共范围统一 `[start, end_exclusive)`。
        mode: 'auto' 按刷新间隔决定是否联网；'force' 强制刷新；'offline' 禁止任何网络请求。
        accept_stale_on_refresh_failure: 刷新失败后是否允许使用仍合格的旧数据。
        max_staleness: 允许多久未成功核对来源；`None` 表示不限制（仅在非严格研究场景使用）。
        require_full_coverage: 是否要求覆盖状态为 complete；严格任务必须为 True。
        require_point_in_time: 是否要求“历史当时可得”；数据集未证明时必须拒绝。
        required: 在一次多数据集任务中该项是否必需。同一数据集可在不同任务里必需或可选。
    """

    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    mode: str = MODE_AUTO
    accept_stale_on_refresh_failure: bool = False
    max_staleness: Optional[dt.timedelta] = None
    require_full_coverage: bool = True
    require_point_in_time: bool = False
    required: bool = True

    def __post_init__(self) -> None:
        # frozen dataclass：用 object.__setattr__ 写回规范化后的 UTC 时间。
        object.__setattr__(self, "start", require_utc(self.start, arg_name="request.start"))
        object.__setattr__(
            self, "end_exclusive", require_utc(self.end_exclusive, arg_name="request.end_exclusive")
        )
        if self.start >= self.end_exclusive:
            raise ConfigError(
                f"请求区间非法：start={self.start} 必须早于 end_exclusive={self.end_exclusive}"
                "（公共范围语义为 [start, end_exclusive)）。"
            )
        if self.mode not in REQUEST_MODES:
            raise ConfigError(f"mode={self.mode!r} 非法，允许值: {REQUEST_MODES}")
        if self.max_staleness is not None:
            if not isinstance(self.max_staleness, dt.timedelta):
                raise ConfigError("max_staleness 必须是 timedelta 或 None。")
            if self.max_staleness < dt.timedelta(0):
                raise ConfigError("max_staleness 不能为负。")
        if self.accept_stale_on_refresh_failure and self.max_staleness is None:
            # 为什么强制：允许降级却不声明陈旧上限，等于“永远可用旧数据”，
            # 会让 §3.2“严格任务必须拒绝不合格缓存”失效。
            raise ConfigError(
                "accept_stale_on_refresh_failure=True 时必须显式提供 max_staleness，"
                "否则无法区分‘可接受的降级’与‘无限期使用旧数据’。"
            )


@dataclass(frozen=True)
class CoverageReport:
    """
    一次覆盖检查的结果。

    Attributes:
        status: COVERAGE_COMPLETE / COVERAGE_PARTIAL / COVERAGE_UNVERIFIED。
        checked_start: 本次实际检查的范围左闭端点。
        checked_end_exclusive: 本次实际检查的范围右开端点。
        actual_start: 数据中最早主键时间；空数据为 None。
        actual_end: 数据中最晚主键时间；空数据为 None。
        rows: 行数。
        missing_head / missing_interior / missing_tail: 网格数据的缺失计数（分头/内/尾）。
        missing_examples: 最多若干个缺失时间戳文本，便于定位。
        extra_count: 落在网格外或范围外的额外记录数。
        scan_complete: 产生这份数据的那次采集是否完成了对 `checked_*` 范围的完整扫描。
                       它与 `status` 是两件事：扫描未完成时即使当前看不到缺口，也不能
                       把“没发现缺口”当成完整性证据（Issue #5 §4.1 提前空页条款）。
        note: 覆盖判定说明（例如“来源日历未验证”）。
    """

    status: str
    checked_start: pd.Timestamp
    checked_end_exclusive: pd.Timestamp
    actual_start: Optional[pd.Timestamp]
    actual_end: Optional[pd.Timestamp]
    rows: int
    missing_head: int = 0
    missing_interior: int = 0
    missing_tail: int = 0
    missing_examples: Tuple[str, ...] = field(default_factory=tuple)
    extra_count: int = 0
    scan_complete: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in COVERAGE_STATUSES:
            raise ConfigError(f"coverage status={self.status!r} 非法，允许值: {COVERAGE_STATUSES}")

    @property
    def missing_total(self) -> int:
        """缺失记录总数（仅对严格网格数据有意义）。"""
        return self.missing_head + self.missing_interior + self.missing_tail

    def to_manifest_dict(self) -> dict:
        """序列化为 manifest 中的覆盖检查片段（时间统一 ISO-8601 UTC 文本）。"""
        return {
            "status": self.status,
            "checked_start": self.checked_start.isoformat(),
            "checked_end_exclusive": self.checked_end_exclusive.isoformat(),
            "actual_start": None if self.actual_start is None else self.actual_start.isoformat(),
            "actual_end": None if self.actual_end is None else self.actual_end.isoformat(),
            "rows": int(self.rows),
            "missing_head": int(self.missing_head),
            "missing_interior": int(self.missing_interior),
            "missing_tail": int(self.missing_tail),
            "missing_examples": list(self.missing_examples),
            "extra_count": int(self.extra_count),
            "scan_complete": bool(self.scan_complete),
            "note": self.note,
        }


@dataclass(frozen=True)
class DataResult:
    """
    公共返回结构（Issue #5 §3.2 要求的最小字段集）。

    Attributes:
        dataset_id: 数据集标识。
        frame: 数据本体（时间列为 tz-aware UTC 索引）。
        content_path: 正式内容文件路径。
        content_sha256: 内容文件字节哈希。
        identity: 数据身份。
        schema_version / transform_version: 数据格式与转换口径版本。
        requested_start / requested_end_exclusive: 本次请求范围。
        coverage: 覆盖检查结果。
        fetched_at: 本次内容对应的最近一次成功获取时间。
        last_verified_at: 最近一次“真实来源结果成功解析并按合同核验”的时间。
                          本地读取、报告生成、失败重试都不会推进它（Issue #5 §3.2）。
        used_cache: 本次是否直接使用了缓存（未联网）。
        refresh_failed: 本次是否尝试刷新但失败。
        point_in_time_verified: 是否可用于“历史当时可得”类研究。
        degradations: 本次降级说明（例如“刷新失败后使用合格旧缓存”）。
        errors: 本次错误说明（刷新失败原因等）。
    """

    dataset_id: str
    frame: pd.DataFrame
    content_path: str
    content_sha256: str
    identity: Mapping[str, Any]
    schema_version: str
    transform_version: str
    requested_start: pd.Timestamp
    requested_end_exclusive: pd.Timestamp
    coverage: CoverageReport
    fetched_at: pd.Timestamp
    last_verified_at: pd.Timestamp
    used_cache: bool
    refresh_failed: bool
    point_in_time_verified: bool
    degradations: Tuple[str, ...] = field(default_factory=tuple)
    errors: Tuple[str, ...] = field(default_factory=tuple)

    def usage_notes(self) -> dict:
        """
        返回“使用说明”：调用方据此判断能否用于严格任务，而不是只看有没有数据。

        Returns:
            dict：不含数据本体的可序列化摘要。
        """
        return {
            "dataset_id": self.dataset_id,
            "content_path": self.content_path,
            "content_sha256": self.content_sha256,
            "identity": dict(self.identity),
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
            "requested_start": self.requested_start.isoformat(),
            "requested_end_exclusive": self.requested_end_exclusive.isoformat(),
            "coverage": self.coverage.to_manifest_dict(),
            "fetched_at": self.fetched_at.isoformat(),
            "last_verified_at": self.last_verified_at.isoformat(),
            "used_cache": bool(self.used_cache),
            "refresh_failed": bool(self.refresh_failed),
            "point_in_time_verified": bool(self.point_in_time_verified),
            "degradations": list(self.degradations),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class AcquireContext:
    """
    传给采集函数的全部外部依赖（数据进数据出，不依赖隐式全局状态）。

    Attributes:
        dataset: 数据集定义。
        request: 本次请求（含固定的 `[start, end_exclusive)`）。
        now_utc: 本次任务固定的一个截止时刻；采集函数不得自行调用 `datetime.now()`。
        http: HTTP 客户端；`offline` 模式或本地导入数据集为 None，采集函数必须显式处理。
        cached_frame: 已有正式数据（可能覆盖比本次请求更早的历史）；用于增量采集与修订比对。
        credentials: 显式传入的凭据映射；其值永不进入 params/manifest/日志/异常信息。
        local_inputs: 本地导入所需的路径等输入；本地采集不联网、不修改用户原文件。
        force_refresh: 是否为 `force` 模式。为 True 时采集函数必须真的重新向来源取数，
                       不允许因为“缓存已覆盖请求范围”而跳过网络请求 —— 否则 `force`
                       就退化成了普通缓存命中，无法用于核对来源是否发生修订。
    """

    dataset: "Dataset"
    request: "DataRequest"
    now_utc: pd.Timestamp
    http: Optional[Any] = None
    cached_frame: Optional[pd.DataFrame] = None
    force_refresh: bool = False
    credentials: Mapping[str, str] = field(default_factory=dict)
    local_inputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquireResult:
    """
    采集函数的返回值。

    Attributes:
        frame: 本次采集得到的数据（索引为 tz-aware UTC 时间，名为 dataset.time_column）。
        scan_complete: 是否完成了对请求范围的完整扫描。
                       提前空页、预算耗尽等情况必须为 False —— 它只证明“本次响应为空”，
                       不证明区间已被扫完（Issue #5 §4.1）。
        scan_note: 扫描说明，会写入覆盖报告。
        requests_used: 本次采集消耗的 HTTP 请求数，用于证据归档。
    """

    frame: pd.DataFrame
    scan_complete: bool = True
    scan_note: str = ""
    requests_used: int = 0
