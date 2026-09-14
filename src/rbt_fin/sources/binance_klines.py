# -*- coding: utf-8 -*-
"""
Binance K 线（分钟行情）采集适配器。

功能或目的：
    实现 Issue #5 §4.2 的分钟行情合同，并修正 trading 旧实现的安全缺口：
        - 固定任务截止：所有页都保持本次 `end_exclusive`，不会因翻页而追到“现在”。
        - 只取已收盘周期：右开边界由调用方用 `timeutils.complete_end_exclusive` 固定。
        - 提前空页只代表本次响应为空，必须标记扫描未完成，由覆盖校验去判头/内/尾。
        - 重复页、游标不推进、越界记录立即 Fail Fast，不假报完整。
        - 严格解析数值，绝不 `coerce + dropna` 静默删行。
        - 增量采集：只请求缓存未覆盖的连续缺口段，冷启动 / 追尾 / 补洞共用同一机制。

调用关系：
    - `rbt_fin.registry.build_klines_dataset()` 绑定本模块的 `acquire_klines`。
    - 底层请求走 `rbt_fin.http.HttpClient`，时间语义走 `rbt_fin.timeutils`。
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
from ..timeutils import (
    expected_grid,
    from_epoch_ms,
    parse_interval,
    require_utc,
    to_epoch_ms,
)
from ..validation import dedupe_by_time_primary_key, parse_strict_float

# 市场 -> 端点。分钟行情的首批生产承诺是 USDⓈ-M 期货成交价（trading 当前所用）。
#
# `spot_data_mirror` 是 Binance 官方公开只读行情镜像（data-api.binance.vision），
# 协议形状与 `api.binance.com/api/v3/klines` 完全一致、无需鉴权。
# 它存在的唯一理由是：在 `fapi.binance.com` / `api.binance.com` 被地域限制（HTTP 451）
# 的环境里，仍能对**同一条代码路径**做真实 HTTP 资格验证。
# 它的数据身份是 spot，**不能**冒充 USDⓈ-M 期货已验证。
KLINES_ENDPOINTS = {
    "usdm_futures": "https://fapi.binance.com/fapi/v1/klines",
    "spot": "https://api.binance.com/api/v3/klines",
    "spot_data_mirror": "https://data-api.binance.vision/api/v3/klines",
    # 三类参考价格。它们与成交价 K 线共用同一套数组形状与分页协议，因此复用同一条取数路径；
    # 但语义完全不同：`volume` 恒为占位 "0"，绝不能当成成交量对外（Issue #5 §5）。
    "usdm_index_price": "https://fapi.binance.com/fapi/v1/indexPriceKlines",
    "usdm_mark_price": "https://fapi.binance.com/fapi/v1/markPriceKlines",
    "usdm_premium_index": "https://fapi.binance.com/fapi/v1/premiumIndexKlines",
}

# 各端点用来指定标的的参数名。指数价用 `pair`，其余用 `symbol`——
# 这一位差异不能靠"看起来都一样"推导，必须显式声明。
KLINES_SYMBOL_PARAM = {
    "usdm_index_price": "pair",
}
DEFAULT_SYMBOL_PARAM = "symbol"

# Binance klines 单页上限。
MAX_LIMIT = 1500

KLINE_COLUMNS = ("open", "high", "low", "close", "volume")


def _business_check(payload: Any) -> None:
    """
    Binance 的业务错误在 HTTP 200 下表现为 `{"code": -1121, "msg": "..."}`。

    为什么单独检查：
        HTTP 状态与 JSON 业务状态是两件事；只看 200 会把“交易对不存在”当成空数据。

    Raises:
        BusinessStatusError: 载荷是错误对象。
        ResponseSchemaError: 载荷既不是列表也不是错误对象。
    """
    if isinstance(payload, list):
        return
    if isinstance(payload, dict) and "code" in payload:
        raise BusinessStatusError(
            f"Binance 业务错误: code={payload['code']!r}, msg={payload.get('msg')!r}"
        )
    raise ResponseSchemaError(f"Binance klines 响应结构异常，期望数组，实际为 {type(payload)!r}")


def _parse_page(rows: Sequence[Any], *, context: str) -> pd.DataFrame:
    """
    解析一页 K 线为标准 OHLCV 数据帧。

    Args:
        rows: 端点返回的行数组；每行至少 6 个元素
              （open_time, open, high, low, close, volume, ...）。
        context: 报错上下文。

    Returns:
        索引为 tz-aware UTC 开盘时间、列为 OHLCV 的数据帧。

    Raises:
        ResponseSchemaError: 行结构或 open_time 类型不符合预期。
        ValueContractError: 数值无法严格解析（由 `parse_strict_float` 抛出）。
    """
    open_times: List[pd.Timestamp] = []
    columns: Dict[str, List[Any]] = {name: [] for name in KLINE_COLUMNS}
    for position, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ResponseSchemaError(
                f"{context}: 第 {position} 行结构不符合 K 线合同（需至少 6 个字段），实际为 {row!r}"
            )
        raw_open_time = row[0]
        if isinstance(raw_open_time, bool) or not isinstance(raw_open_time, int):
            raise ResponseSchemaError(
                f"{context}: 第 {position} 行 open_time 必须是整数毫秒，实际为 {raw_open_time!r}"
            )
        open_times.append(from_epoch_ms(raw_open_time, arg_name=f"{context}.open_time[{position}]"))
        for offset, name in enumerate(KLINE_COLUMNS, start=1):
            columns[name].append(row[offset])

    index = pd.DatetimeIndex(open_times, tz="UTC", name="datetime")
    data = {
        name: parse_strict_float(values, column=name, context=context)
        for name, values in columns.items()
    }
    return pd.DataFrame(data, index=index)


def _missing_segments(
    *, grid: pd.DatetimeIndex, have: pd.DatetimeIndex, interval: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    把“期望网格减去已有时间戳”的结果切成若干连续缺口段。

    为什么不逐点请求：
        冷启动、追尾更新与补洞在本质上是同一件事——补齐缺口段。切成连续段后
        一次请求可以覆盖成百上千个点，同时天然限制请求数。

    Args:
        grid: 请求范围的期望网格。
        have: 已有（缓存）时间戳。
        interval: 固定长度周期。

    Returns:
        `[(segment_start, segment_end_exclusive), ...]`，按时间升序。
    """
    missing = grid.difference(have)
    if len(missing) == 0:
        return []
    delta = parse_interval(interval)
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    seg_start = missing[0]
    prev = missing[0]
    for current in missing[1:]:
        if current - prev != delta:
            segments.append((seg_start, prev + delta))
            seg_start = current
        prev = current
    segments.append((seg_start, prev + delta))
    return segments


def fetch_klines_range(
    *,
    http,
    market: str,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    limit: int,
    context: str,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    拉取 `[start, end_exclusive)` 的 K 线，并如实报告扫描是否完成。

    Args:
        http: `rbt_fin.http.HttpClient`。
        market: KLINES_ENDPOINTS 中的市场键。
        symbol: 交易对，例如 'BTCUSDT'。
        interval: 固定长度周期，例如 '1m'。
        start: 左闭端点（tz-aware UTC，必须落在网格上）。
        end_exclusive: 右开端点（tz-aware UTC，必须落在网格上）。
        limit: 单页条数，1..MAX_LIMIT。
        context: 报错/说明上下文。

    Returns:
        `(frame, scan_complete, scan_note)`。

    Raises:
        ConfigError: 市场或 limit 非法。
        PaginationError: 重复页、游标不推进或返回越界记录。
    """
    if market not in KLINES_ENDPOINTS:
        raise ConfigError(f"未知 market={market!r}，可用: {sorted(KLINES_ENDPOINTS)}")
    if not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
        raise ConfigError(f"limit 必须是 1..{MAX_LIMIT} 的整数，实际为 {limit!r}")
    url = KLINES_ENDPOINTS[market]
    start_ts = require_utc(start, arg_name=f"{context}.start")
    end_ts = require_utc(end_exclusive, arg_name=f"{context}.end_exclusive")
    delta = parse_interval(interval)

    # Binance 的 endTime 是包含语义；本库对外统一右开，因此在端点边界减 1 毫秒。
    # 这个转换只发生在适配器内部，公共层永远看到 `[start, end_exclusive)`。
    end_time_ms = to_epoch_ms(end_ts, arg_name=f"{context}.end_exclusive") - 1

    frames: List[pd.DataFrame] = []
    seen: set = set()
    cursor = start_ts
    previous_max: Optional[pd.Timestamp] = None
    scan_complete = True
    scan_note = ""

    while cursor < end_ts:
        params = {
            KLINES_SYMBOL_PARAM.get(market, DEFAULT_SYMBOL_PARAM): symbol,
            "interval": interval,
            "startTime": to_epoch_ms(cursor, arg_name=f"{context}.cursor"),
            "endTime": end_time_ms,
            "limit": limit,
        }
        payload = http.get_json(
            url=url, params=params, context=context, business_check=_business_check
        )
        if len(payload) == 0:
            # 提前空页：只证明本次响应为空。区间是否真的到此为止，交给覆盖校验判定。
            scan_complete = False
            scan_note = (
                f"提前空页：请求 startTime={cursor.isoformat()} 返回空数组，"
                "已终止扫描；空响应不构成区间完整性证据。"
            )
            break

        page = _parse_page(payload, context=context)
        page_index = page.index
        out_of_range = page_index[(page_index < start_ts) | (page_index >= end_ts)]
        if len(out_of_range) > 0:
            raise PaginationError(
                f"{context}: 端点返回 {len(out_of_range)} 条越界记录（本次截止为 "
                f"{end_ts.isoformat()}），示例: {[str(t) for t in out_of_range[:3]]}"
            )

        page_set = set(page_index)
        if page_set and page_set.issubset(seen):
            raise PaginationError(
                f"{context}: 收到完全重复的分页（{len(page_set)} 条全部已见），终止并报告未完成。"
            )
        page_max = page_index.max()
        if previous_max is not None and page_max <= previous_max:
            raise PaginationError(
                f"{context}: 分页游标未推进（上一页最大 {previous_max}，本页最大 {page_max}），"
                "终止并报告未完成。"
            )

        frames.append(page)
        seen.update(page_set)
        previous_max = page_max
        cursor = page_max + delta

    if frames:
        # 多页重叠是允许的（端点边界为包含语义），但同一开盘时间必须给出相同内容；
        # 不同内容意味着端点或翻页出了问题，必须暴露而不是悄悄取一条。
        combined = dedupe_by_time_primary_key(
            pd.concat(frames), value_columns=KLINE_COLUMNS, context=context
        )
    else:
        combined = pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in KLINE_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
        )
    return combined, scan_complete, scan_note


def acquire_klines(context: AcquireContext) -> AcquireResult:
    """
    数据集级采集入口：只补齐缓存未覆盖的缺口段，并保留缓存中更早的历史。

    Args:
        context: 采集上下文（含 dataset/request/http/cached_frame）。

    Returns:
        AcquireResult：`frame` 为“缓存 ∪ 本次新取”的完整数据。

    Raises:
        ConfigError: 数据集未声明网格，或 offline 上下文缺少 http 却仍需联网。
    """
    dataset = context.dataset
    request = context.request
    if dataset.grid is None:
        raise ConfigError(f"{dataset.dataset_id}: K 线数据集必须声明 grid。")
    interval = dataset.grid.interval
    params = dataset.params
    market = params["market"]
    symbol = params["symbol"]
    limit = params["limit"]

    cached = context.cached_frame
    # force 模式下把“已有”视为空：必须真的重新向来源取数，否则无法发现来源修订。
    have = (
        pd.DatetimeIndex([], tz="UTC", name="datetime")
        if (context.force_refresh or cached is None)
        else cached.index
    )
    grid = expected_grid(start=request.start, end_exclusive=request.end_exclusive, interval=interval)
    segments = _missing_segments(grid=grid, have=have, interval=interval)

    if not segments:
        # 缓存已覆盖请求范围的每个网格点：零网络请求，但仍要走一次完整校验。
        return AcquireResult(
            frame=cached if cached is not None else pd.DataFrame(),
            scan_complete=True,
            scan_note="缓存已覆盖请求范围全部网格点，本次未发起网络请求。",
            requests_used=0,
        )

    if context.http is None:
        raise ConfigError(
            f"{dataset.dataset_id}: 缓存未覆盖请求范围且未提供 HTTP 客户端，无法采集。"
        )

    before = context.http.requests_used
    fetched: List[pd.DataFrame] = []
    scan_complete = True
    notes: List[str] = []
    for seg_start, seg_end in segments:
        frame, segment_complete, note = fetch_klines_range(
            http=context.http,
            market=market,
            symbol=symbol,
            interval=interval,
            start=seg_start,
            end_exclusive=seg_end,
            limit=limit,
            context=f"{dataset.dataset_id}[{seg_start.isoformat()},{seg_end.isoformat()})",
        )
        fetched.append(frame)
        scan_complete = scan_complete and segment_complete
        if note:
            notes.append(note)

    pieces = [piece for piece in ([cached] if cached is not None else []) + fetched if len(piece)]
    if pieces:
        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in KLINE_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
        )
    return AcquireResult(
        frame=merged,
        scan_complete=scan_complete,
        scan_note=" | ".join(notes),
        requests_used=context.http.requests_used - before,
    )
