# -*- coding: utf-8 -*-
"""
迁入 trading 的七项行情护栏（保留测试意图，并补上已确认的反例）。

对照来源：
    trading `tests/unit/test_binance_data_handler_pipeline_guards.py`
    （基线 cb1fe04071ed516abd57b7999072cba5b11f85e3）。

每个用例都注明它在旧实现里守护的是什么风险，以及在 rbt_fin 中由哪个入口承担。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from rbt_fin.errors import (
    ConfigError,
    ContractError,
    ResponseSchemaError,
    ValueContractError,
)
from rbt_fin.sources.binance_klines import fetch_klines_range
from rbt_fin.timeutils import complete_end_exclusive, parse_interval
from rbt_fin.validation import dedupe_by_time_primary_key

from tests.conftest import FakeResponse, kline_row, make_client, minute_klines, to_ms

OHLCV = ("open", "high", "low", "close", "volume")


def _frame(index, closes):
    """构造一份合法 OHLCV（high/low 正确包住 open/close），只让 close 变化。"""
    return pd.DataFrame(
        {
            "open": [1.0] * len(closes),
            "high": [max(1.0, c) + 1.0 for c in closes],
            "low": [min(1.0, c) - 1.0 for c in closes],
            "close": list(closes),
            "volume": [0.0] * len(closes),
        },
        index=index,
    )


def test_guard1_identical_ohlcv_at_different_timestamps_is_kept():
    """
    护栏 1：不同时间戳但 OHLCV 完全相同的真实 bar 不得被删除。

    旧风险：按 OHLCV 值去重会误删真实 K 线，让回测输入出现看不见的缺口。
    新归属：`validation.dedupe_by_time_primary_key` 只按时间主键去重。
    """
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00:00", "2026-01-01 00:01:00"], tz="UTC", name="datetime"
    )
    frame = _frame(index, [1.0, 1.0])
    out = dedupe_by_time_primary_key(frame, value_columns=OHLCV, context="guard1")
    assert len(out) == 2
    assert list(out.index) == list(index)


def test_guard2_same_timestamp_same_content_is_deduped():
    """
    护栏 2（新默认）：同一时间戳、内容相同 -> 去重。

    注意与旧实现的差别：旧实现对“同时间不同内容”采用 keep-last 静默覆盖；
    新默认冲突策略改为显式拒绝，见 `test_guard2b_*`。
    """
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    index = pd.DatetimeIndex([ts, ts, pd.Timestamp("2026-01-01 00:01:00", tz="UTC")],
                             tz="UTC", name="datetime")
    frame = _frame(index, [1.0, 1.0, 3.0])
    out = dedupe_by_time_primary_key(frame, value_columns=OHLCV, context="guard2")
    assert out.index.is_unique
    assert len(out) == 2


def test_guard2b_same_timestamp_different_content_is_rejected():
    """
    护栏 2 的新默认冲突策略：同一时间戳、内容不同 -> 拒绝，不静默 keep-last。

    这是本次相对旧实现的行为变更：未经解释的行情冲突不允许覆盖正式数据。
    """
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    index = pd.DatetimeIndex([ts, ts], tz="UTC", name="datetime")
    frame = _frame(index, [1.0, 2.0])
    with pytest.raises(ContractError, match="内容不同"):
        dedupe_by_time_primary_key(frame, value_columns=OHLCV, context="guard2b")


def test_guard3_end_boundary_excludes_current_unfinished_bar():
    """
    护栏 3：右开边界对齐到当前未完成 bar 的开盘时间，排除未收盘 bar。

    与旧实现的等价关系：旧实现返回“当前 bar 开盘时间 - 1ms”作为包含式 endTime；
    本库统一右开语义，二者在端点上完全等价（适配器内部减 1ms）。
    """
    now = dt.datetime(2026, 3, 2, 10, 35, 45, tzinfo=dt.timezone.utc)
    end_exclusive = complete_end_exclusive(now_utc=now, interval="1m")
    assert end_exclusive == pd.Timestamp("2026-03-02 10:35:00", tz="UTC")
    legacy_inclusive_end = end_exclusive - pd.Timedelta(milliseconds=1)
    assert legacy_inclusive_end == pd.Timestamp("2026-03-02 10:34:59.999", tz="UTC")


def test_guard4_unfillable_gap_is_reported_not_silently_accepted(cache_root, fixed_now):
    """
    护栏 4：缺口补不上时必须失败，不得返回“看似完整”的数据。

    新归属：缺口补齐不再是独立的 `fill_discontinuity`，而是公共流程的一部分——
    采集只请求缺失段，补不回来时覆盖校验报告缺口，严格请求在 pipeline 处失败。
    """
    from rbt_fin.contracts import DataRequest
    from rbt_fin.errors import CoverageError
    from rbt_fin.pipeline import acquire_dataset
    from rbt_fin.registry import build_klines_dataset

    # 端点只在第一次请求时给出 00:00，其后一律空页：00:01/00:02 的缺口永远补不上。
    def handler(url, params):
        if params["startTime"] == to_ms("2026-01-01 00:00"):
            return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:00", count=1))
        return FakeResponse(status_code=200, payload=[])

    client, _ = make_client(handler)
    with pytest.raises(CoverageError, match="严格请求要求完整覆盖"):
        acquire_dataset(
            dataset=build_klines_dataset(),
            request=DataRequest(
                start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                end_exclusive=pd.Timestamp("2026-01-01 00:03", tz="UTC"),
            ),
            store_root=cache_root,
            now_utc=fixed_now,
            http=client,
        )


def test_guard5_weekly_interval_is_rejected():
    """
    护栏 5：周线不能走 epoch floor 对齐，必须 Fail Fast。

    旧风险：周线锚点由交易所定义，epoch 取整会错锚并返回“看似合理”的时间。
    """
    with pytest.raises(ConfigError, match="不支持 epoch floor 对齐"):
        parse_interval("1w")
    with pytest.raises(ConfigError, match="不支持 epoch floor 对齐"):
        complete_end_exclusive(
            now_utc=dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc), interval="1M"
        )


def test_guard6_naive_datetime_is_rejected_at_entry():
    """
    护栏 6：naive datetime 必须在入口处 Fail Fast。

    旧风险：naive 值在 `.timestamp()` 时按本地时区解释，造成静默时间漂移。
    """
    client, _ = make_client(lambda url, params: FakeResponse(status_code=200, payload=[]))
    aware = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    naive = pd.Timestamp("2026-01-01 00:00")

    with pytest.raises(ConfigError, match="必须是 tz-aware"):
        fetch_klines_range(
            http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
            start=naive, end_exclusive=aware, limit=1500, context="guard6",
        )
    with pytest.raises(ConfigError, match="必须是 tz-aware"):
        fetch_klines_range(
            http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
            start=aware, end_exclusive=naive, limit=1500, context="guard6",
        )


def test_guard7_malformed_row_fails_fast_instead_of_partial_result():
    """
    护栏 7：解析阶段出错必须抛错，不允许返回半成品。

    旧风险：旧实现在 `except Exception` 里 `break`，把已取到的部分数据当作全部返回。
    """
    pages = [
        minute_klines(start="2026-01-01 00:00", count=1),
        [[to_ms("2026-01-01 00:01"), "only_two_cols"]],
    ]

    def handler(url, params):
        return FakeResponse(status_code=200, payload=pages.pop(0))

    client, _ = make_client(handler)
    with pytest.raises(ResponseSchemaError, match="结构不符合 K 线合同"):
        fetch_klines_range(
            http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:03", tz="UTC"),
            limit=1, context="guard7",
        )


def test_guard7b_non_numeric_value_is_not_silently_dropped():
    """
    护栏 7 的补充反例：坏数值不得被 `coerce + dropna` 静默删行。

    旧实现用 `pd.to_numeric(errors='coerce')` + `dropna()`，一行坏数据会变成一个
    看不见的缺口；新实现在解析处直接失败。
    """
    bad = kline_row(open_time_ms=to_ms("2026-01-01 00:00"), open_=1, high=2, low=0.5, close=1.5, volume=1)
    bad[4] = "not-a-number"

    def handler(url, params):
        return FakeResponse(status_code=200, payload=[bad])

    client, _ = make_client(handler)
    with pytest.raises(ValueContractError, match="无法解析为数值"):
        fetch_klines_range(
            http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:01", tz="UTC"),
            limit=1, context="guard7b",
        )
