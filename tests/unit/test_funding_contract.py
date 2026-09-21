# -*- coding: utf-8 -*-
"""
Binance 资金费率：固定截止、先去重后聚合、startTime 语义、日内乱序。

本文件同时是“基线三处缺陷已修复”的证据：
    1. 升序翻页时 `endTime` 被移除 -> 现在每页都带固定截止。
    2. 每页内先截成日期再求均值，分页重叠重复加权 -> 现在先落原始结算记录再聚合。
    3. 依赖 `startTime=0` 的哨兵行为 -> 现在永远发送由请求推导的正值。
"""

from __future__ import annotations

import pandas as pd
import pytest

from rbt_fin.contracts import COVERAGE_UNVERIFIED, DataRequest
from rbt_fin.errors import PaginationError, ResponseSchemaError
from rbt_fin.registry import build_funding_dataset
from rbt_fin.sources.binance_funding import fetch_funding_records, funding_daily_mean
from rbt_fin.transforms import daily_mean_by_utc_date
from rbt_fin.validation import validate_irregular_series

from tests.conftest import FakeResponse, funding_row, make_client, to_ms

SYMBOL = "BTCUSDT"


def _fetch(handler, *, start: str, end: str, limit: int = 2, **client_kwargs):
    client, session = make_client(handler, **client_kwargs)
    frame, complete, note = fetch_funding_records(
        http=client, symbol=SYMBOL,
        start=pd.Timestamp(start, tz="UTC"), end_exclusive=pd.Timestamp(end, tz="UTC"),
        limit=limit, context="funding",
    )
    return frame, complete, note, session


def test_every_page_keeps_fixed_end_time():
    """
    基线缺陷 1 的直接反例：升序翻页时 `endTime` 必须**仍然存在**且保持不变。

    旧实现在升序分支执行 `params.pop("endTime", None)`，于是第二页起不再受任务截止约束。
    """
    pages = {
        to_ms("2026-01-01 00:00"): [
            funding_row(funding_time_ms=to_ms("2026-01-01 00:00"), rate=0.0001),
            funding_row(funding_time_ms=to_ms("2026-01-01 08:00"), rate=0.0002),
        ],
        to_ms("2026-01-01 08:00") + 1: [
            funding_row(funding_time_ms=to_ms("2026-01-01 16:00"), rate=0.0003),
            funding_row(funding_time_ms=to_ms("2026-01-02 00:00"), rate=0.0004),
        ],
    }

    def handler(url, params):
        return FakeResponse(status_code=200, payload=pages.get(params["startTime"], []))

    frame, _, _, session = _fetch(handler, start="2026-01-01 00:00", end="2026-01-02 08:00")
    expected_end = to_ms("2026-01-02 08:00") - 1
    assert len(session.calls) >= 2
    for _url, params in session.calls:
        assert "endTime" in params, "每页都必须带固定截止，否则会一路追到最新数据"
        assert params["endTime"] == expected_end
    assert len(frame) == 4


def test_start_time_is_always_positive_and_never_zero():
    """
    基线缺陷 3：`startTime=0` 在该端点上的含义未经本版本验证，绝不作为哨兵使用。

    即使请求范围从 epoch 0 开始，发送的 startTime 也必须是正值。
    """
    def handler(url, params):
        return FakeResponse(status_code=200, payload=[])

    _, _, _, session = _fetch(handler, start="1970-01-01 00:00", end="2026-01-01 00:00")
    assert session.calls, "应至少发出一次请求"
    for _url, params in session.calls:
        assert "startTime" in params
        assert params["startTime"] >= 1


def test_overlapping_pages_do_not_double_weight_daily_mean():
    """
    基线缺陷 2 的直接反例：分页重叠不得让同一条结算重复参与日均。

    构造：两页重叠 00:00 这一条。若先聚合后合并（旧行为），00:00 会被计两次，
    日均从 (1+2+3)/3 = 2.0 漂移为 (1+1+2+3)/4 = 1.75。
    """
    day = "2026-01-01"
    page1 = [
        funding_row(funding_time_ms=to_ms(f"{day} 00:00"), rate=0.0001),
        funding_row(funding_time_ms=to_ms(f"{day} 08:00"), rate=0.0002),
    ]
    page2 = [
        # 与上一页重叠一条（同时间、同内容），再补 16:00。
        funding_row(funding_time_ms=to_ms(f"{day} 08:00"), rate=0.0002),
        funding_row(funding_time_ms=to_ms(f"{day} 16:00"), rate=0.0003),
    ]
    pages = {to_ms(f"{day} 00:00"): page1, to_ms(f"{day} 08:00") + 1: page2}

    def handler(url, params):
        return FakeResponse(status_code=200, payload=pages.get(params["startTime"], []))

    frame, _, _, _ = _fetch(handler, start=f"{day} 00:00", end="2026-01-02 00:00")
    assert len(frame) == 3, "原始结算记录必须先按事件主键去重"

    daily = funding_daily_mean(frame)
    # 手算标准答案（不由被测函数生成）：(0.0001 + 0.0002 + 0.0003) / 3 = 0.0002
    assert len(daily) == 1
    assert daily["funding_rate"].iloc[0] == pytest.approx(0.0002, abs=1e-12)


def test_daily_mean_requires_deduped_input():
    """
    聚合函数自身拒绝未去重的输入：把“先去重”变成可执行的前置条件，而不是口头约定。
    """
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 00:00"], tz="UTC", name="funding_time"
    )
    frame = pd.DataFrame({"funding_rate": [0.0001, 0.0001]}, index=index)
    with pytest.raises(Exception, match="重复事件主键"):
        daily_mean_by_utc_date(frame, value_columns=["funding_rate"], context="dup")


def test_intraday_out_of_order_input_is_rejected_before_aggregation():
    """
    日内乱序：聚合必须按**原始时间**排序，而不是依赖行的偶然排列。

    这里让转换函数直接拒绝未排序输入，把风险挡在“取到当天任意一条”之前。
    """
    index = pd.DatetimeIndex(
        ["2026-01-01 16:00", "2026-01-01 00:00", "2026-01-01 08:00"], tz="UTC", name="funding_time"
    )
    frame = pd.DataFrame({"funding_rate": [0.3, 0.1, 0.2]}, index=index)
    with pytest.raises(Exception, match="按原始时间升序"):
        daily_mean_by_utc_date(frame, value_columns=["funding_rate"], context="unsorted")
    # 排序后结果是手算的 (0.1 + 0.2 + 0.3) / 3 = 0.2
    ordered = frame.sort_index()
    daily = daily_mean_by_utc_date(ordered, value_columns=["funding_rate"], context="sorted")
    assert daily["funding_rate"].iloc[0] == pytest.approx(0.2, abs=1e-12)


def test_missing_required_field_fails_fast():
    """字段不全（缺 fundingRate）必须失败：字段缺失不是记录缺失。"""
    def handler(url, params):
        return FakeResponse(
            status_code=200,
            payload=[{"symbol": SYMBOL, "fundingTime": to_ms("2026-01-01 00:00")}],
        )

    with pytest.raises(ResponseSchemaError, match="缺少必需字段"):
        _fetch(handler, start="2026-01-01 00:00", end="2026-01-02 00:00")


def test_out_of_range_record_fails_fast():
    """端点返回越过任务截止的记录必须立即失败。"""
    def handler(url, params):
        return FakeResponse(
            status_code=200,
            payload=[funding_row(funding_time_ms=to_ms("2026-01-03 00:00"), rate=0.0001)],
        )

    with pytest.raises(PaginationError, match="越界记录"):
        _fetch(handler, start="2026-01-01 00:00", end="2026-01-02 00:00")


def test_coverage_is_unverified_because_settlement_calendar_is_not_verified():
    """
    资金费率是事件流：结算日历未验证 -> 覆盖最高只能是 unverified，
    要求严格证明完整的调用必须被拒绝，而不是靠“看起来每 8 小时一条”自我说服。
    """
    dataset = build_funding_dataset()
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:00"], tz="UTC", name="funding_time"
    )
    frame = pd.DataFrame({"funding_rate": [0.1, 0.2, 0.3]}, index=index)
    coverage = validate_irregular_series(
        frame, dataset=dataset,
        request=DataRequest(
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-02", tz="UTC"),
        ),
    )
    assert coverage.status == COVERAGE_UNVERIFIED
    assert "日历未验证" in coverage.note
