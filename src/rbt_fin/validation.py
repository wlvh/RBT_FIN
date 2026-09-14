# -*- coding: utf-8 -*-
"""
时间轴、数值与覆盖校验。

功能或目的：
    实现 Issue #5 §4.2 / §4.3 的数据质量合同：

    严格网格（分钟行情）：
        - 时间戳精确落在网格上（秒、微秒、纳秒都检查）；严格升序、主键唯一。
        - OHLCV 必需、数值合法有限，价格与高低区间关系正确，成交量满足定义。
        - `expected_grid - actual_timestamps` 分头部 / 内部 / 尾部报告，并检查额外/越界记录。
        - 绝不使用 `coerce + dropna` 静默删除异常行。

    非规则序列（低频 / 事件流）：
        - 不强加分钟新鲜度或“每天一条”；区分非发布日、记录缺失与字段缺失。
        - 来源日历未验证时覆盖状态最高只能是 `unverified`，要求严格证明完整的调用必须拒绝。

调用关系：
    - `rbt_fin.registry` 把这里的校验函数绑定到具体 `Dataset`。
    - `rbt_fin.pipeline` 调用 `Dataset.validate` 得到 `CoverageReport` 并据此决定是否放行。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import (
    COVERAGE_COMPLETE,
    COVERAGE_PARTIAL,
    COVERAGE_UNVERIFIED,
    CoverageReport,
    DataRequest,
    Dataset,
)
from .errors import ContractError, TimeContractError, ValueContractError
from .timeutils import (
    assert_on_grid,
    assert_strictly_increasing_unique,
    expected_grid,
)

# 缺失样例最多展示的条数；只为定位，不作为完整缺口清单。
MISSING_EXAMPLE_LIMIT = 5

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def parse_strict_float(values: Sequence[object], *, column: str, context: str) -> np.ndarray:
    """
    严格解析为 float64；任何无法解析或非有限的值都直接失败。

    为什么不用 `pd.to_numeric(errors='coerce')`：
        coerce 会把坏值变成 NaN，随后的 dropna 就静默删掉了整行 K 线，让缺口以
        “数据本来就没有”的形式混进正式数据（Issue #5 §4.2 明令禁止）。

    Args:
        values: 原始值序列（通常是交易所返回的字符串）。
        column: 列名，用于报错。
        context: 上下文（数据集/端点）。

    Returns:
        float64 数组。

    Raises:
        ValueContractError: 存在无法解析或非有限（NaN/inf）的值。
    """
    out = np.empty(len(values), dtype="float64")
    for position, raw in enumerate(values):
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueContractError(
                f"{context}: 列 {column!r} 第 {position} 行的值 {raw!r} 无法解析为数值。"
            ) from exc
        if not np.isfinite(parsed):
            raise ValueContractError(
                f"{context}: 列 {column!r} 第 {position} 行的值 {raw!r} 不是有限数值。"
            )
        out[position] = parsed
    return out


def dedupe_by_time_primary_key(
    frame: pd.DataFrame, *, value_columns: Sequence[str], context: str
) -> pd.DataFrame:
    """
    按时间主键去重，并把“同时间不同内容”作为显式冲突暴露出来。

    三条规则（Issue #5 §4.2 最后一条）：
        - 不同时间、相同数值：保留。真实 K 线完全可能连续两分钟数值一致，
          旧实现按 OHLCV 去重会误删真实 bar，直接污染回测。
        - 同时间、相同内容：去重（分页重叠的正常结果）。
        - 同时间、不同内容：默认不接受。让未经解释的冲突静默覆盖，正是本库要消除的行为。

    Args:
        frame: 索引为时间主键的数据帧（可未排序）。
        value_columns: 参与内容比较的列。
        context: 报错上下文。

    Returns:
        按时间升序、主键唯一的数据帧。

    Raises:
        ContractError: 同一时间主键出现内容不同的记录。
    """
    ordered = frame.sort_index(kind="stable")
    duplicated_mask = ordered.index.duplicated(keep=False)
    if not bool(duplicated_mask.any()):
        return ordered

    # 只对重复主键做内容比较，避免对全表做昂贵的分组运算。
    duplicated_part = ordered[duplicated_mask]
    grouped = duplicated_part.groupby(level=0)[list(value_columns)]
    inconsistent = grouped.nunique().gt(1).any(axis=1)
    conflicting = inconsistent[inconsistent].index
    if len(conflicting) > 0:
        raise ContractError(
            f"{context}: 同一时间主键出现内容不同的记录 {len(conflicting)} 处，"
            f"示例: {[str(t) for t in conflicting[:5]]}；"
            "默认不允许未经解释的冲突覆盖，必须由显式修订策略处理。"
        )
    return ordered[~ordered.index.duplicated(keep="first")]


def _classify_missing(
    *, missing: pd.DatetimeIndex, actual: pd.DatetimeIndex
) -> Tuple[int, int, int, str]:
    """
    把缺失时间戳分类为头部 / 内部 / 尾部。

    为什么要分三类：
        尾部缺失通常意味着“本次扫描未追到截止时刻”（例如提前空页），内部缺失意味着
        来源确实有洞，头部缺失通常是历史起点问题。三者的处置完全不同，合并成一个
        “缺 N 条”会让调用方无法判断该补哪一段。

    Args:
        missing: 缺失时间戳（升序）。
        actual: 实际取得的时间戳（升序，可能为空）。

    Returns:
        (head, interior, tail, note)。`actual` 为空时全部记入 head 并在 note 中说明。
    """
    if len(missing) == 0:
        return 0, 0, 0, ""
    if len(actual) == 0:
        return len(missing), 0, 0, "请求范围内没有任何记录，全部缺失记入头部。"
    first, last = actual[0], actual[-1]
    head = int((missing < first).sum())
    tail = int((missing > last).sum())
    interior = len(missing) - head - tail
    return head, interior, tail, ""


def validate_strict_grid_ohlcv(
    frame: pd.DataFrame,
    *,
    dataset: Dataset,
    request: DataRequest,
    scan_complete: bool = True,
    scan_note: str = "",
) -> CoverageReport:
    """
    分钟行情（严格网格 OHLCV）校验。

    Args:
        frame: 索引为 tz-aware UTC 时间的数据帧，列至少包含 OHLCV。
        dataset: 数据集定义（必须声明 `grid`）。
        request: 本次请求，提供 `[start, end_exclusive)`。
        scan_complete: 采集端是否完成了完整扫描。False 表示扫描被提前终止
                       （预算耗尽 / 提前空页），此时即使缺口为 0 也只能标记为未验证。
        scan_note: 扫描说明，进入覆盖报告 note。

    Returns:
        CoverageReport。

    Raises:
        TimeContractError: 时间轴违约（不在网格上 / 重复 / 非升序）。单页越界由来源适配器 Fail Fast。
        ValueContractError: 数值违约（非有限、价格区间关系错误、成交量为负）。
    """
    context = f"{dataset.dataset_id}"
    if dataset.grid is None:
        raise TimeContractError(f"{context}: 该数据集未声明严格网格，不能使用网格校验。")
    interval = dataset.grid.interval

    index = frame.index
    assert_on_grid(index, interval=interval, context=context)
    assert_strictly_increasing_unique(index, context=context)

    missing_columns = [col for col in OHLCV_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueContractError(f"{context}: 缺少必需 OHLCV 列 {missing_columns}。")

    # 数值合同：所有值必须有限；价格必须为正；high/low 必须正确包住 open/close；volume 非负。
    for column in OHLCV_COLUMNS:
        series = frame[column]
        if not np.isfinite(series.to_numpy(dtype="float64")).all():
            bad = series[~np.isfinite(series.to_numpy(dtype="float64"))]
            raise ValueContractError(
                f"{context}: 列 {column!r} 存在非有限值，示例索引: {[str(i) for i in bad.index[:5]]}"
            )
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype="float64")
    if (prices <= 0).any():
        rows = frame.index[(prices <= 0).any(axis=1)]
        raise ValueContractError(
            f"{context}: 存在非正价格，示例索引: {[str(i) for i in rows[:5]]}"
        )
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    open_ = frame["open"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    bad_range = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
    if bad_range.any():
        rows = frame.index[bad_range]
        raise ValueContractError(
            f"{context}: high/low 未正确包住 open/close，示例索引: {[str(i) for i in rows[:5]]}"
        )
    volume = frame["volume"].to_numpy(dtype="float64")
    if (volume < 0).any():
        rows = frame.index[volume < 0]
        raise ValueContractError(
            f"{context}: 成交量为负，示例索引: {[str(i) for i in rows[:5]]}"
        )

    # 覆盖只在请求范围内统计：合并了更早历史的正式数据不应因为“比请求更全”而被判违约。
    # 越界记录本身仍要被看见——单页越界由来源适配器直接 Fail Fast（见 sources/*），
    # 这里把范围外的行计入 extra_count 并在 note 中说明，避免它们悄悄参与覆盖统计。
    in_range_mask = (index >= request.start) & (index < request.end_exclusive)
    in_range = index[in_range_mask]
    extra_count = int((~in_range_mask).sum())

    grid = expected_grid(
        start=request.start, end_exclusive=request.end_exclusive, interval=interval
    )
    missing = grid.difference(in_range)
    head, interior, tail, classify_note = _classify_missing(missing=missing, actual=in_range)

    notes = [note for note in (scan_note, classify_note) if note]
    if len(missing) == 0 and scan_complete:
        status = COVERAGE_COMPLETE
    elif len(missing) == 0 and not scan_complete:
        # 扫描被提前终止时，“没发现缺口”不构成完整性证据。
        status = COVERAGE_UNVERIFIED
        notes.append("采集扫描未完成，缺口为 0 不能作为完整性证据。")
    else:
        status = COVERAGE_PARTIAL

    if extra_count:
        notes.append(f"正式数据另有 {extra_count} 条记录落在本次请求范围之外（更早/更晚历史，不参与覆盖统计）。")

    return CoverageReport(
        status=status,
        checked_start=request.start,
        checked_end_exclusive=request.end_exclusive,
        actual_start=None if len(in_range) == 0 else in_range[0],
        actual_end=None if len(in_range) == 0 else in_range[-1],
        rows=int(len(in_range)),
        missing_head=head,
        missing_interior=interior,
        missing_tail=tail,
        missing_examples=tuple(str(ts) for ts in missing[:MISSING_EXAMPLE_LIMIT]),
        extra_count=extra_count,
        scan_complete=bool(scan_complete),
        note=" | ".join(notes),
    )


def validate_irregular_series(
    frame: pd.DataFrame,
    *,
    dataset: Dataset,
    request: DataRequest,
    scan_complete: bool = True,
    scan_note: str = "",
    required_columns: Optional[Sequence[str]] = None,
) -> CoverageReport:
    """
    非规则序列（低频指标 / 资金费率等事件流）的校验。

    为什么不套网格：
        这类来源天然存在非发布日（周末、假期、结算规则变化）。用分钟网格判定完整性
        会把“正常的非发布日”误报成缺失；反过来，没有验证过来源日历时也不能宣称完整。
        因此这里只做“时间轴合法 + 范围内 + 字段齐全”的硬校验，覆盖状态默认 `unverified`。

    Args:
        frame: 索引为 tz-aware UTC 时间的数据帧。
        dataset: 数据集定义；`calendar_verified=True` 时才允许给出 complete。
        request: 本次请求。
        scan_complete: 采集端是否完成完整扫描。
        scan_note: 扫描说明。
        required_columns: 必须存在且不得有缺失值的列；默认取 `dataset.value_columns`。

    Returns:
        CoverageReport。

    Raises:
        TimeContractError: 时间轴违约（非 tz-aware / 重复 / 非升序）。
        ValueContractError: 必需字段缺失或含空值。
    """
    context = f"{dataset.dataset_id}"
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TimeContractError(f"{context}: 时间索引必须是 tz-aware 的 DatetimeIndex。")
    assert_strictly_increasing_unique(index, context=context)

    columns = list(required_columns) if required_columns is not None else list(dataset.value_columns)
    missing_columns = [col for col in columns if col not in frame.columns]
    if missing_columns:
        raise ValueContractError(f"{context}: 缺少必需字段 {missing_columns}（字段缺失 != 记录缺失）。")
    for column in columns:
        if frame[column].isna().any():
            rows = frame.index[frame[column].isna()]
            raise ValueContractError(
                f"{context}: 字段 {column!r} 存在空值，示例索引: {[str(i) for i in rows[:5]]}；"
                "字段缺失必须显式处理，不允许静默填充。"
            )

    # 与网格分支同理：覆盖只统计请求范围内的记录，范围外历史计入 extra_count。
    in_range_mask = (index >= request.start) & (index < request.end_exclusive)
    in_range = index[in_range_mask]
    extra_count = int((~in_range_mask).sum())

    notes = [note for note in (scan_note,) if note]
    if not dataset.calendar_verified:
        notes.append("来源发布日历未验证，覆盖标记为未验证；要求严格证明完整的调用应拒绝。")
        status = COVERAGE_UNVERIFIED
    elif not scan_complete:
        notes.append("采集扫描未完成，覆盖标记为未验证。")
        status = COVERAGE_UNVERIFIED
    else:
        status = COVERAGE_COMPLETE

    if extra_count:
        notes.append(f"正式数据另有 {extra_count} 条记录落在本次请求范围之外（更早/更晚历史，不参与覆盖统计）。")

    return CoverageReport(
        status=status,
        checked_start=request.start,
        checked_end_exclusive=request.end_exclusive,
        actual_start=None if len(in_range) == 0 else in_range[0],
        actual_end=None if len(in_range) == 0 else in_range[-1],
        rows=int(len(in_range)),
        extra_count=extra_count,
        scan_complete=bool(scan_complete),
        note=" | ".join(notes),
    )
