# -*- coding: utf-8 -*-
"""
Binance USDⓈ-M 永续资金费率采集适配器（WP1 的首个日频代表）。

功能或目的：
    实现 Issue #5 §4.4 / §5 对资金费率的要求，并修正基线实现的三处问题：

    1. **所有页保持固定截止。** 基线在“升序翻页”分支里执行 `params.pop("endTime")`，
       于是第二页开始就不再受本次任务截止约束，会一路追到最新数据。本实现每页都带
       固定 `endTime`，并在分页结束后再次核对结果范围。
    2. **先原始记录、后聚合。** 基线在每页内就把 `fundingTime` 截成日期再 `ensure_daily`
       求均值，分页重叠会让同一条结算重复参与日均。本实现先落原始结算记录
       （主键 = fundingTime），去重后再由 `transforms.daily_mean_by_utc_date` 生成日频视图。
    3. **不依赖 `startTime=0` 的哨兵行为。** K 线与资金费率对 `startTime=0` 的处理不能
       相互推导；本实现永远发送由请求推导出的显式正值 startTime，`0` 从不出现在参数里。

调用关系：
    - `rbt_fin.registry.build_funding_dataset()` 绑定本模块的 `acquire_funding_records`。
    - 日频视图由 `rbt_fin.transforms.daily_mean_by_utc_date` 生成（复用同一份基础数据）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..contracts import AcquireContext, AcquireResult
from ..errors import (
    BusinessStatusError,
    ConfigError,
    PaginationError,
    ResponseSchemaError,
)
from ..timeutils import from_epoch_ms, require_utc, to_epoch_ms
from ..transforms import daily_mean_by_utc_date
from ..validation import dedupe_by_time_primary_key, parse_strict_float

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# Binance 资金费率端点单页上限。
MAX_LIMIT = 1000

# 落盘的原始记录列。为什么只有 funding_rate：
# markPrice 在历史记录中的可得性尚未经本版本的真实资格验证；
# 未验证的字段不进入 v1 数据合同，避免“声称有、实际靠猜”。
RAW_VALUE_COLUMNS = ("funding_rate",)
RAW_TIME_COLUMN = "funding_time"


def _business_check(payload: Any) -> None:
    """
    区分 HTTP 200 下的业务错误与正常空数组。

    Raises:
        BusinessStatusError: 载荷是 `{"code": ..., "msg": ...}` 错误对象。
        ResponseSchemaError: 载荷既不是数组也不是错误对象。
    """
    if isinstance(payload, list):
        return
    if isinstance(payload, dict) and "code" in payload:
        raise BusinessStatusError(
            f"Binance 资金费率业务错误: code={payload['code']!r}, msg={payload.get('msg')!r}"
        )
    raise ResponseSchemaError(
        f"Binance fundingRate 响应结构异常，期望数组，实际为 {type(payload)!r}"
    )


def _parse_page(rows: Sequence[Any], *, context: str) -> pd.DataFrame:
    """
    解析一页资金费率为原始结算记录。

    Args:
        rows: 端点返回的对象数组，每项需含 `fundingTime` 与 `fundingRate`。
        context: 报错上下文。

    Returns:
        索引为结算时间（tz-aware UTC，名为 `funding_time`）的数据帧。

    Raises:
        ResponseSchemaError: 行不是对象或缺少必需字段（字段不全必须可见，不得静默丢弃）。
    """
    times: List[pd.Timestamp] = []
    rates: List[Any] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ResponseSchemaError(
                f"{context}: 第 {position} 项不是对象，实际为 {type(row)!r}"
            )
        for field_name in ("fundingTime", "fundingRate"):
            if field_name not in row:
                raise ResponseSchemaError(
                    f"{context}: 第 {position} 项缺少必需字段 {field_name!r}，实际键: {sorted(row)}"
                )
        raw_time = row["fundingTime"]
        if isinstance(raw_time, bool) or not isinstance(raw_time, int):
            raise ResponseSchemaError(
                f"{context}: 第 {position} 项 fundingTime 必须是整数毫秒，实际为 {raw_time!r}"
            )
        times.append(from_epoch_ms(raw_time, arg_name=f"{context}.fundingTime[{position}]"))
        rates.append(row["fundingRate"])

    index = pd.DatetimeIndex(times, tz="UTC", name=RAW_TIME_COLUMN)
    return pd.DataFrame(
        {"funding_rate": parse_strict_float(rates, column="funding_rate", context=context)},
        index=index,
    )


def fetch_funding_records(
    *,
    http,
    symbol: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    limit: int,
    context: str,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    分页拉取 `[start, end_exclusive)` 的原始资金费率结算记录。

    分页协议（由本适配器负责，不从 K 线端点推导）：
        - 升序翻页：`startTime` 推进到上一页最晚结算时间 + 1ms。
        - **每页都保持固定 `endTime`**，因此任何一页都不会越过本次任务截止。
        - `startTime` 恒为由请求推导的正值；`0` 的哨兵语义未经本版本验证，不使用。
        - 越界记录、重复页、游标不推进都立即 Fail Fast。
        - 空页只终止扫描并标记未完成，不当作“已到最新”。

    Args:
        http: `rbt_fin.http.HttpClient`。
        symbol: 交易对。
        start: 左闭端点（tz-aware UTC）。
        end_exclusive: 右开端点（tz-aware UTC）。
        limit: 单页条数，1..MAX_LIMIT。
        context: 报错上下文。

    Returns:
        `(frame, scan_complete, scan_note)`；frame 为去重排序后的原始结算记录。

    Raises:
        ConfigError: limit 非法。
        PaginationError: 越界记录、重复页或游标不推进。
    """
    if not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
        raise ConfigError(f"limit 必须是 1..{MAX_LIMIT} 的整数，实际为 {limit!r}")
    start_ts = require_utc(start, arg_name=f"{context}.start")
    end_ts = require_utc(end_exclusive, arg_name=f"{context}.end_exclusive")

    start_ms = to_epoch_ms(start_ts, arg_name=f"{context}.start")
    # 端点 endTime 为包含语义；对外右开边界减 1 毫秒即为本次任务的固定截止。
    end_ms = to_epoch_ms(end_ts, arg_name=f"{context}.end_exclusive") - 1
    # startTime 永远为正：0 在该端点上的含义未经本版本验证，不作为“自最早开始”的哨兵使用。
    cursor_ms = max(1, start_ms)

    frames: List[pd.DataFrame] = []
    seen: set = set()
    previous_max_ms: Optional[int] = None
    scan_complete = True
    scan_note = ""

    while cursor_ms <= end_ms:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "startTime": cursor_ms,
            "endTime": end_ms,  # 每页固定截止；这里正是基线被修复的地方。
            "limit": limit,
        }
        payload = http.get_json(
            url=FUNDING_URL, params=params, context=context, business_check=_business_check
        )
        if len(payload) == 0:
            scan_complete = False
            scan_note = (
                f"提前空页：startTime={cursor_ms} 返回空数组，已终止扫描；"
                "空响应不构成区间完整性证据。"
            )
            break

        page = _parse_page(payload, context=context)
        page_index = page.index
        out_of_range = page_index[(page_index < start_ts) | (page_index >= end_ts)]
        if len(out_of_range) > 0:
            raise PaginationError(
                f"{context}: 端点返回 {len(out_of_range)} 条越界记录（本次截止 {end_ts.isoformat()}），"
                f"示例: {[str(t) for t in out_of_range[:3]]}"
            )

        page_set = set(page_index)
        if page_set and page_set.issubset(seen):
            raise PaginationError(
                f"{context}: 收到完全重复的分页（{len(page_set)} 条全部已见），终止并报告未完成。"
            )
        page_max_ms = to_epoch_ms(page_index.max(), arg_name=f"{context}.page_max")
        if previous_max_ms is not None and page_max_ms <= previous_max_ms:
            raise PaginationError(
                f"{context}: 分页游标未推进（上一页 {previous_max_ms}，本页 {page_max_ms}），"
                "终止并报告未完成。"
            )

        frames.append(page)
        seen.update(page_set)
        previous_max_ms = page_max_ms
        cursor_ms = page_max_ms + 1

        if len(payload) < limit:
            # 末页：本页不足一整页，说明端点在固定截止内已无更多记录。
            break

    if frames:
        # 分页重叠允许存在，但同一结算时间只能保留一条 —— 否则日均会被重复加权；
        # 同时间不同费率则是真实冲突，必须暴露。
        combined = dedupe_by_time_primary_key(
            pd.concat(frames), value_columns=RAW_VALUE_COLUMNS, context=context
        )
    else:
        combined = pd.DataFrame(
            {"funding_rate": pd.Series(dtype="float64")},
            index=pd.DatetimeIndex([], tz="UTC", name=RAW_TIME_COLUMN),
        )

    # 分页结束后对结果范围再检查一次：即使每页都带了 endTime，也不把端点行为当作证据。
    if len(combined) and (combined.index.min() < start_ts or combined.index.max() >= end_ts):
        raise PaginationError(
            f"{context}: 合并结果范围 [{combined.index.min()}, {combined.index.max()}] "
            f"超出请求范围 [{start_ts}, {end_ts})。"
        )
    return combined, scan_complete, scan_note


def acquire_funding_records(context: AcquireContext) -> AcquireResult:
    """
    数据集级采集入口：保留缓存中的既有结算记录，只补齐缓存尾部之后的区间。

    为什么用“缓存最晚记录之后”作为增量起点：
        资金费率是事件流而不是规则网格，无法用网格差集定位缺口；已结算记录不会改写，
        因此从缓存最晚结算时间 + 1ms 开始补齐是安全且请求数最小的方式。
        若调用方请求的范围早于缓存起点，则回退为对整段请求范围重新扫描。

    Args:
        context: 采集上下文。

    Returns:
        AcquireResult：`frame` 为“缓存 ∪ 本次新取”的原始结算记录。
    """
    dataset = context.dataset
    request = context.request
    symbol = dataset.params["symbol"]
    limit = dataset.params["limit"]

    cached = context.cached_frame
    fetch_start = request.start
    # force 模式跳过增量优化，直接重扫整段请求范围。
    if not context.force_refresh and cached is not None and len(cached):
        cached_max = cached.index.max()
        if cached.index.min() <= request.start and cached_max >= request.start:
            # 缓存已覆盖请求头部：只补齐尾部（+1ms 避免重复取同一条结算）。
            fetch_start = cached_max + pd.Timedelta(milliseconds=1)
    if fetch_start >= request.end_exclusive:
        return AcquireResult(
            frame=cached,
            scan_complete=True,
            scan_note="缓存已覆盖请求范围尾部，本次未发起网络请求。",
            requests_used=0,
        )

    if context.http is None:
        raise ConfigError(
            f"{dataset.dataset_id}: 缓存未覆盖请求范围且未提供 HTTP 客户端，无法采集。"
        )
    before = context.http.requests_used
    fetched, scan_complete, scan_note = fetch_funding_records(
        http=context.http,
        symbol=symbol,
        start=fetch_start,
        end_exclusive=request.end_exclusive,
        limit=limit,
        context=f"{dataset.dataset_id}[{fetch_start.isoformat()},{request.end_exclusive.isoformat()})",
    )
    pieces = [piece for piece in ([cached] if cached is not None else []) + [fetched] if len(piece)]
    if pieces:
        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fetched
    return AcquireResult(
        frame=merged,
        scan_complete=scan_complete,
        scan_note=scan_note,
        requests_used=context.http.requests_used - before,
    )


def funding_daily_mean(records: pd.DataFrame) -> pd.DataFrame:
    """
    由原始结算记录生成旧的日均资金费率视图。

    Args:
        records: `acquire_funding_records` 产出的原始记录（已按结算时间去重排序）。

    Returns:
        以 UTC 日期为索引、含 `funding_rate` 的日频数据帧。
    """
    return daily_mean_by_utc_date(
        records, value_columns=RAW_VALUE_COLUMNS, context="binance_funding_daily_mean"
    )
