# -*- coding: utf-8 -*-
"""
时间与覆盖：三个必须保留的直接反例 A/B，以及头/内/尾缺失、越界、未完成周期、时区输入。

反例 A（Issue #5 §8）：`00:00:00.500000`、`00:01:00.500000` 在严格分钟合同下拒绝。
反例 B（Issue #5 §8）：要求 `[00:00,00:05)`，仅取得 00:00、00:01 后空页；必须报告尾部未完成。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from rbt_fin.contracts import (
    COVERAGE_COMPLETE,
    COVERAGE_PARTIAL,
    COVERAGE_UNVERIFIED,
    DataRequest,
)
from rbt_fin.errors import ConfigError, CoverageError, PaginationError, TimeContractError
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_klines_dataset
from rbt_fin.sources.binance_klines import fetch_klines_range
from rbt_fin.timeutils import complete_end_exclusive, require_utc
from rbt_fin.validation import validate_strict_grid_ohlcv

from tests.conftest import FakeResponse, kline_row, make_client, minute_klines, to_ms


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """构造一份合法 OHLCV 数据帧，价格随行递增以便定位。"""
    base = [30000.0 + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "open": base,
            "high": [p + 5 for p in base],
            "low": [p - 5 for p in base],
            "close": [p + 1 for p in base],
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def _request(start: str, end: str, **kwargs) -> DataRequest:
    return DataRequest(
        start=pd.Timestamp(start, tz="UTC"), end_exclusive=pd.Timestamp(end, tz="UTC"), **kwargs
    )


# ----------------------------- 反例 A -----------------------------

def test_counterexample_a_half_second_offsets_are_rejected():
    """
    反例 A：半秒偏移必须被拒绝。

    旧实现只检查 `.second == 0`，`00:00:00.500000` 的秒位正好是 0，会被放行。
    新实现以纳秒对 interval 取模，一次覆盖秒 / 微秒 / 纳秒三级精度。
    """
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00:00.500000", "2026-01-01 00:01:00.500000"], tz="UTC", name="datetime"
    )
    dataset = build_klines_dataset()
    with pytest.raises(TimeContractError, match="未落在 interval=1m 网格上"):
        validate_strict_grid_ohlcv(
            _ohlcv(index), dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05")
        )


def test_counterexample_a_sub_second_variants_all_rejected():
    """反例 A 的三级精度补充：微秒与纳秒偏移同样必须被拒绝。"""
    dataset = build_klines_dataset()
    for offset in ("2026-01-01 00:00:00.000001", "2026-01-01 00:00:00.000000001"):
        index = pd.DatetimeIndex([offset], tz="UTC", name="datetime")
        with pytest.raises(TimeContractError, match="未落在 interval=1m 网格上"):
            validate_strict_grid_ohlcv(
                _ohlcv(index), dataset=dataset,
                request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
            )


# ----------------------------- 反例 B -----------------------------

def test_counterexample_b_early_empty_page_reports_tail_incomplete(cache_root, fixed_now):
    """
    反例 B：要求 `[00:00,00:05)`，端点给出 00:00、00:01 后返回空页。

    断言：
        - 覆盖状态为 partial，且缺失全部落在**尾部**（3 条），不是头部或内部。
        - 扫描被标记为未完成，说明里明确写出“空响应不构成完整性证据”。
        - 严格请求因此失败，正式数据不会被发布。
    """
    def handler(url, params):
        if params["startTime"] == to_ms("2026-01-01 00:00"):
            return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:00", count=2))
        return FakeResponse(status_code=200, payload=[])

    client, session = make_client(handler)
    dataset = build_klines_dataset()
    request = _request("2026-01-01 00:00", "2026-01-01 00:05")

    frame, scan_complete, note = fetch_klines_range(
        http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
        start=request.start, end_exclusive=request.end_exclusive, limit=1500, context="B",
    )
    assert len(frame) == 2
    assert scan_complete is False
    assert "提前空页" in note

    coverage = validate_strict_grid_ohlcv(
        frame, dataset=dataset, request=request, scan_complete=scan_complete, scan_note=note
    )
    assert coverage.status == COVERAGE_PARTIAL
    assert (coverage.missing_head, coverage.missing_interior, coverage.missing_tail) == (0, 0, 3)
    assert coverage.missing_examples[0].startswith("2026-01-01 00:02")

    # 端到端：严格请求必须失败，且不留下正式数据。
    client2, _ = make_client(handler)
    with pytest.raises(CoverageError, match="tail=3"):
        acquire_dataset(
            dataset=dataset, request=request, store_root=cache_root, now_utc=fixed_now, http=client2
        )
    assert not (cache_root / dataset.dataset_id / "manifest.json").exists()


# ----------------------------- 头 / 内 / 尾 -----------------------------

@pytest.mark.parametrize(
    "present,expected",
    [
        (["00:02", "00:03", "00:04"], (2, 0, 0)),          # 头部缺失
        (["00:00", "00:01", "00:04"], (0, 2, 0)),          # 内部缺失
        (["00:00", "00:01", "00:02"], (0, 0, 2)),          # 尾部缺失
        (["00:00", "00:02", "00:04"], (0, 2, 0)),          # 交错内部缺失
    ],
)
def test_missing_is_classified_into_head_interior_tail(present, expected):
    """
    缺失必须按头 / 内 / 尾分类报告：三者的处置方式完全不同。
    """
    index = pd.DatetimeIndex([f"2026-01-01 {t}" for t in present], tz="UTC", name="datetime")
    coverage = validate_strict_grid_ohlcv(
        _ohlcv(index), dataset=build_klines_dataset(),
        request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
    )
    assert coverage.status == COVERAGE_PARTIAL
    assert (coverage.missing_head, coverage.missing_interior, coverage.missing_tail) == expected


def test_full_grid_with_complete_scan_is_complete():
    """网格无缺口且扫描完成 -> complete。这是严格请求唯一能通过的状态。"""
    index = pd.DatetimeIndex(
        [f"2026-01-01 00:0{i}" for i in range(5)], tz="UTC", name="datetime"
    )
    coverage = validate_strict_grid_ohlcv(
        _ohlcv(index), dataset=build_klines_dataset(),
        request=_request("2026-01-01 00:00", "2026-01-01 00:05"), scan_complete=True,
    )
    assert coverage.status == COVERAGE_COMPLETE
    assert coverage.missing_total == 0


def test_no_gap_but_incomplete_scan_is_unverified():
    """
    扫描未完成时，“没发现缺口”不构成完整性证据 -> unverified。

    这是提前空页条款最容易被忽略的一面：缺口计数为 0 也不能放行严格请求。
    """
    index = pd.DatetimeIndex(
        [f"2026-01-01 00:0{i}" for i in range(5)], tz="UTC", name="datetime"
    )
    coverage = validate_strict_grid_ohlcv(
        _ohlcv(index), dataset=build_klines_dataset(),
        request=_request("2026-01-01 00:00", "2026-01-01 00:05"), scan_complete=False,
    )
    assert coverage.status == COVERAGE_UNVERIFIED
    assert coverage.missing_total == 0
    assert "不能作为完整性证据" in coverage.note


# ----------------------------- 越界记录 -----------------------------

def test_out_of_range_page_records_fail_fast():
    """端点返回越界记录（越过本次任务截止）必须立即失败，不得悄悄纳入。"""
    def handler(url, params):
        # 第 3 条越过 [00:00, 00:02) 的截止。
        return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:00", count=3))

    client, _ = make_client(handler)
    with pytest.raises(PaginationError, match="越界记录"):
        fetch_klines_range(
            http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:02", tz="UTC"),
            limit=1500, context="oob",
        )


def test_records_outside_request_range_count_as_extra_not_missing():
    """
    正式数据中比请求更宽的历史计入 extra_count，不影响覆盖判定。

    这是 trading 冷启动规则的直接依赖：已有更早的合格历史不应因为窄窗口请求被裁掉。
    """
    index = pd.DatetimeIndex(
        [f"2026-01-01 00:0{i}" for i in range(6)], tz="UTC", name="datetime"
    )
    coverage = validate_strict_grid_ohlcv(
        _ohlcv(index), dataset=build_klines_dataset(),
        request=_request("2026-01-01 00:01", "2026-01-01 00:05"),
    )
    assert coverage.status == COVERAGE_COMPLETE
    assert coverage.extra_count == 2  # 00:00 与 00:05
    assert coverage.rows == 4


# ----------------------------- 未完成周期 -----------------------------

def test_unfinished_period_is_excluded_by_end_boundary():
    """
    未完成周期必须被右开边界排除：10:35:45 时，10:35 这根 bar 还在形成。
    """
    now = dt.datetime(2026, 3, 2, 10, 35, 45, tzinfo=dt.timezone.utc)
    end_exclusive = complete_end_exclusive(now_utc=now, interval="1m")
    index = pd.DatetimeIndex(["2026-03-02 10:34:00", "2026-03-02 10:35:00"], tz="UTC", name="datetime")
    coverage = validate_strict_grid_ohlcv(
        _ohlcv(index), dataset=build_klines_dataset(),
        request=DataRequest(start=pd.Timestamp("2026-03-02 10:34", tz="UTC"), end_exclusive=end_exclusive),
    )
    # 10:35 落在请求范围之外 -> 计入 extra，不参与覆盖；10:34 完整。
    assert coverage.status == COVERAGE_COMPLETE
    assert coverage.rows == 1
    assert coverage.extra_count == 1


# ----------------------------- 时区输入 -----------------------------

def test_utc_non_utc_and_naive_inputs():
    """
    UTC / 非 UTC / naive 三类输入的处置：
        - UTC：原样接受。
        - 非 UTC（带时区）：转换为 UTC，指向同一时刻。
        - naive：在接口处拒绝。
    """
    utc = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert require_utc(utc, arg_name="t") == utc

    tokyo = pd.Timestamp("2026-01-01 09:00", tz="Asia/Tokyo")
    assert require_utc(tokyo, arg_name="t") == utc

    with pytest.raises(ConfigError, match="必须是 tz-aware"):
        require_utc(pd.Timestamp("2026-01-01 00:00"), arg_name="t")
    with pytest.raises(ConfigError, match="必须是 tz-aware"):
        require_utc(dt.datetime(2026, 1, 1), arg_name="t")
