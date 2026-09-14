# -*- coding: utf-8 -*-
"""
Bybit V5 资金费率历史采集适配器。

功能或目的：
    按 Issue #5 §5 对 Bybit 的三条专项要求实现，并修正基线的两处问题：

    1. **不依赖 `startTime=0` 的哨兵行为。** 基线用 `start_ms=0` 表示"自最早起取"。
       该语义在本版本未经验证，且与 Binance 的同名参数不能互相推导。
       本实现首页**不传 startTime**、只带固定 `endTime`，随后用
       `endTime = 本页最早时间 - 1ms` 向历史推进；有明确下界时在每页校验并终止。

    2. **`retCode != 0` 立即失败。** 基线只取 `result.list`，业务错误会表现为"没有数据"
       后正常结束——凭据失效或参数错误于是变成了"该来源今天没有记录"。

    与 Binance 资金费率一致的部分：先落**原始结算记录**（主键 = 结算时间），
    去重之后再由 `transforms.daily_mean_by_utc_date` 生成日均视图。

调用关系：
    `rbt_fin.registry.build_bybit_funding_dataset()` 绑定本模块的 `acquire_bybit_funding`。
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
from ..validation import dedupe_by_time_primary_key, parse_strict_float

FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"

# Bybit V5 该端点的单页上限。
MAX_LIMIT = 200

RAW_TIME_COLUMN = "funding_time"
RAW_VALUE_COLUMNS: Tuple[str, ...] = ("funding_rate",)

# 单次采集允许的最大翻页次数。与请求预算共同构成"有限终止"保证：
# 预算是全局的，这里是单个数据集内的上界，防止一次采集独吞整批预算。
MAX_PAGES = 400


def _business_check(payload: Any) -> None:
    """
    Bybit 的业务状态在 `retCode` 上：`0` 才是成功。

    Raises:
        BusinessStatusError: `retCode != 0`。
        ResponseSchemaError: 载荷结构不符合预期。
    """
    if not isinstance(payload, dict):
        raise ResponseSchemaError(f"Bybit 响应必须是对象，实际为 {type(payload).__name__}")
    if "retCode" not in payload:
        raise ResponseSchemaError(f"Bybit 响应缺少 retCode，实际键: {sorted(payload)}")
    if payload["retCode"] != 0:
        raise BusinessStatusError(
            f"Bybit 业务错误: retCode={payload['retCode']!r}, retMsg={payload.get('retMsg')!r}"
        )


def _parse_page(payload: Any, *, context: str) -> pd.DataFrame:
    """
    解析一页资金费率为原始结算记录。

    Args:
        payload: 已通过业务状态检查的完整响应。
        context: 报错上下文。

    Returns:
        索引为结算时间（tz-aware UTC）的数据帧。

    Raises:
        ResponseSchemaError: `result.list` 结构或字段不符合预期。
    """
    result = payload["result"] if isinstance(payload.get("result"), dict) else None
    if result is None or not isinstance(result.get("list"), list):
        raise ResponseSchemaError(f"{context}: 响应缺少 result.list 数组")
    rows: Sequence[Any] = result["list"]
    times: List[pd.Timestamp] = []
    rates: List[Any] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ResponseSchemaError(f"{context}: 第 {position} 项不是对象")
        for field in ("fundingRateTimestamp", "fundingRate"):
            if field not in row:
                raise ResponseSchemaError(
                    f"{context}: 第 {position} 项缺少必需字段 {field!r}，实际键: {sorted(row)}"
                )
        stamp = row["fundingRateTimestamp"]
        # 实测该字段是**毫秒字符串**；接受字符串但拒绝非整数文本，保持时间精度合同。
        if isinstance(stamp, str):
            if not stamp.strip().isdigit():
                raise ResponseSchemaError(
                    f"{context}: 第 {position} 项 fundingRateTimestamp={stamp!r} 不是整数毫秒文本"
                )
            stamp = int(stamp)
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise ResponseSchemaError(
                f"{context}: 第 {position} 项 fundingRateTimestamp 必须是整数毫秒，实际为 {stamp!r}"
            )
        times.append(from_epoch_ms(stamp, arg_name=f"{context}.fundingRateTimestamp[{position}]"))
        rates.append(row["fundingRate"])

    index = pd.DatetimeIndex(times, tz="UTC", name=RAW_TIME_COLUMN)
    return pd.DataFrame(
        {"funding_rate": parse_strict_float(rates, column="funding_rate", context=context)},
        index=index,
    )


def fetch_bybit_funding(
    *,
    http,
    symbol: str,
    category: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    limit: int,
    context: str,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    向历史翻页拉取 `[start, end_exclusive)` 的原始资金费率结算记录。

    分页协议：
        - 首页只带固定 `endTime`，**不传 startTime**（该端点对 0 的处理未经本版本验证）。
        - 后续页用 `endTime = 本页最早结算时间 - 1ms` 向更早推进。
        - 每页都显式带 `endTime`，因此任何一页都不会越过本次任务截止。
        - 重复页、游标不推进、越过下界、页数上限都终止；空页只标记扫描未完成。

    Args:
        http: `rbt_fin.http.HttpClient`。
        symbol: 交易对。
        category: 市场类别（'linear' 等）。
        start: 左闭端点（tz-aware UTC）。
        end_exclusive: 右开端点（tz-aware UTC）。
        limit: 单页条数，1..MAX_LIMIT。
        context: 报错上下文。

    Returns:
        `(frame, scan_complete, scan_note)`。

    Raises:
        ConfigError: limit 非法。
        PaginationError: 重复页、游标不推进或页数上限耗尽。
    """
    if not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
        raise ConfigError(f"limit 必须是 1..{MAX_LIMIT} 的整数，实际为 {limit!r}")
    start_ts = require_utc(start, arg_name=f"{context}.start")
    end_ts = require_utc(end_exclusive, arg_name=f"{context}.end_exclusive")
    start_ms = to_epoch_ms(start_ts, arg_name=f"{context}.start")
    cursor_end_ms = to_epoch_ms(end_ts, arg_name=f"{context}.end_exclusive") - 1

    frames: List[pd.DataFrame] = []
    seen: set = set()
    previous_min_ms: Optional[int] = None
    scan_complete = True
    scan_note = ""

    for page_number in range(MAX_PAGES):
        params: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
            "endTime": cursor_end_ms,
        }
        payload = http.get_json(
            url=FUNDING_URL, params=params, context=context, business_check=_business_check
        )
        page = _parse_page(payload, context=context)
        if len(page) == 0:
            scan_complete = False
            scan_note = (
                f"提前空页：endTime={cursor_end_ms} 返回空列表，已终止向历史的扫描；"
                "空响应不构成区间完整性证据。"
            )
            break

        page_set = set(page.index)
        if page_set.issubset(seen):
            raise PaginationError(
                f"{context}: 收到完全重复的分页（{len(page_set)} 条全部已见），终止并报告未完成。"
            )
        page_min_ms = to_epoch_ms(page.index.min(), arg_name=f"{context}.page_min")
        if previous_min_ms is not None and page_min_ms >= previous_min_ms:
            raise PaginationError(
                f"{context}: 分页游标未向历史推进（上一页最早 {previous_min_ms}，"
                f"本页最早 {page_min_ms}），终止并报告未完成。"
            )

        # 只保留落在请求范围内的记录：越过下界的部分属于本次不需要的更早历史。
        in_range = page[(page.index >= start_ts) & (page.index < end_ts)]
        if len(in_range):
            frames.append(in_range)
        seen.update(page_set)
        previous_min_ms = page_min_ms

        if page_min_ms <= start_ms:
            break  # 已经翻到请求下界之前，正常结束
        cursor_end_ms = page_min_ms - 1
    else:
        raise PaginationError(
            f"{context}: 已达单次采集页数上限 {MAX_PAGES}，终止并报告未完成。"
        )

    if frames:
        combined = dedupe_by_time_primary_key(
            pd.concat(frames), value_columns=RAW_VALUE_COLUMNS, context=context
        )
    else:
        combined = pd.DataFrame(
            {"funding_rate": pd.Series(dtype="float64")},
            index=pd.DatetimeIndex([], tz="UTC", name=RAW_TIME_COLUMN),
        )
    return combined, scan_complete, scan_note


def acquire_bybit_funding(context: AcquireContext) -> AcquireResult:
    """
    数据集级采集入口：保留缓存中的既有结算记录，只补齐缓存尾部之后的区间。

    Args:
        context: 采集上下文。

    Returns:
        AcquireResult。

    Raises:
        ConfigError: 缓存不足却未提供 HTTP 客户端。
    """
    dataset = context.dataset
    request = context.request
    cached = context.cached_frame
    fetch_start = request.start
    if not context.force_refresh and cached is not None and len(cached):
        cached_max = cached.index.max()
        if cached.index.min() <= request.start and cached_max >= request.start:
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
    fetched, scan_complete, scan_note = fetch_bybit_funding(
        http=context.http,
        symbol=dataset.params["symbol"],
        category=dataset.params["category"],
        start=fetch_start,
        end_exclusive=request.end_exclusive,
        limit=dataset.params["limit"],
        context=f"{dataset.dataset_id}[{fetch_start.isoformat()},{request.end_exclusive.isoformat()})",
    )
    pieces = [piece for piece in ([cached] if cached is not None else []) + [fetched] if len(piece)]
    merged = (
        dedupe_by_time_primary_key(
            pd.concat(pieces), value_columns=RAW_VALUE_COLUMNS, context=dataset.dataset_id
        )
        if pieces
        else fetched
    )
    return AcquireResult(
        frame=merged,
        scan_complete=scan_complete,
        scan_note=scan_note,
        requests_used=context.http.requests_used - before,
    )
