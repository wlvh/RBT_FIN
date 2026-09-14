# -*- coding: utf-8 -*-
"""
Binance 三类参考价格（指数价 / 标记价 / 溢价指数）日频采集适配器。

功能或目的：
    这三个端点与成交价 K 线共用数组形状与分页协议，因此取数直接复用
    `binance_klines.fetch_klines_range`；本模块只负责两件它们**特有**的事：

    1. **占位字段不冒充成交量。** 三个端点返回的 `volume` 恒为 "0"，是协议占位，
       不是"当天没有成交"。基线把它按 OHLCV 原样带出来，下游看到的就是一列假成交量。
       这里只保留 `close`，并按价格类型命名（`index_close` / `mark_close` / `premium_close`）。

    2. **分页，并如实记录实际覆盖。** 基线只发一次请求、`limit=1500` 就当成全部历史
       （Issue #5 §5："不把第一批数据称为全部历史"）。这里走与行情同一套分页与
       提前空页语义，覆盖由公共校验按日网格判定。

    另外：溢价指数**可以为负**，因此它不能走要求价格为正的 OHLCV 校验，
    而是走单值网格校验（`validation.validate_strict_grid_series`）。

调用关系：
    `rbt_fin.registry.build_reference_price_dataset()` 绑定本模块的 `acquire_reference_price`。
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ..contracts import AcquireContext, AcquireResult
from ..errors import ConfigError
from ..timeutils import expected_grid, parse_interval
from .binance_klines import _missing_segments, fetch_klines_range


def acquire_reference_price(context: AcquireContext) -> AcquireResult:
    """
    采集参考价格日频序列：只补缓存未覆盖的缺口段，并只保留 close。

    Args:
        context: 采集上下文；`dataset.params` 提供 market/symbol/interval/limit/value_column。

    Returns:
        AcquireResult：`frame` 为"缓存 ∪ 本次新取"，只含单个价格列。

    Raises:
        ConfigError: 数据集未声明网格，或缓存不足却未提供 HTTP 客户端。
    """
    dataset = context.dataset
    request = context.request
    if dataset.grid is None:
        raise ConfigError(f"{dataset.dataset_id}: 参考价格数据集必须声明 grid。")
    interval = dataset.grid.interval
    params = dataset.params
    value_column = params["value_column"]

    cached = context.cached_frame
    have = (
        pd.DatetimeIndex([], tz="UTC", name=dataset.time_column)
        if (context.force_refresh or cached is None)
        else cached.index
    )
    grid = expected_grid(start=request.start, end_exclusive=request.end_exclusive, interval=interval)
    segments = _missing_segments(grid=grid, have=have, interval=interval)
    if not segments:
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
        page, segment_complete, note = fetch_klines_range(
            http=context.http,
            market=params["market"],
            symbol=params["symbol"],
            interval=interval,
            start=seg_start,
            end_exclusive=seg_end,
            limit=params["limit"],
            context=f"{dataset.dataset_id}[{seg_start.isoformat()},{seg_end.isoformat()})",
        )
        # 只保留 close 并按价格类型命名：volume/open/high/low 在这些端点上没有成交语义。
        fetched.append(page[["close"]].rename(columns={"close": value_column}))
        scan_complete = scan_complete and segment_complete
        if note:
            notes.append(note)

    pieces = [piece for piece in ([cached] if cached is not None else []) + fetched if len(piece)]
    if pieces:
        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = dataset.time_column
    else:
        merged = pd.DataFrame(
            {value_column: pd.Series(dtype="float64")},
            index=pd.DatetimeIndex([], tz="UTC", name=dataset.time_column),
        )
    return AcquireResult(
        frame=merged,
        scan_complete=scan_complete,
        scan_note=" | ".join(notes),
        requests_used=context.http.requests_used - before,
    )
