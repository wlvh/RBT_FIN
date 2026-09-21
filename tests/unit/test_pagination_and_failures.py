# -*- coding: utf-8 -*-
"""
分页与失败：多页重叠、重复页、无进展、提前空页、预算耗尽、HTTP/业务失败、字段不全。

共同断言：全部**有限终止**，且任何一种失败都不会被伪装成“正常空数据 / 已完整”。
"""

from __future__ import annotations

import pandas as pd
import pytest

from rbt_fin.errors import (
    BudgetExhaustedError,
    BusinessStatusError,
    HttpStatusError,
    PaginationError,
    ResponseSchemaError,
)
from rbt_fin.sources.binance_klines import fetch_klines_range
from rbt_fin.timeutils import assert_on_grid

from tests.conftest import FakeResponse, kline_row, make_client, minute_klines, to_ms

START = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def _fetch(handler, *, end: str = "2026-01-01 00:10", limit: int = 2, **client_kwargs):
    client, session = make_client(handler, **client_kwargs)
    frame, complete, note = fetch_klines_range(
        http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
        start=START, end_exclusive=pd.Timestamp(end, tz="UTC"), limit=limit, context="page",
    )
    return frame, complete, note, session, client


def test_non_nanosecond_index_resolution_is_still_on_grid():
    """
    回归：pandas 2.0 起 DatetimeIndex 可为秒/毫秒分辨率。

    `assert_on_grid` 若直接用 `asi8` 取模，会把 `10:34:00`（秒分辨率）误判为不在分钟网格上。
    这是实现期实际踩到的一个真问题，固化为回归用例。
    """
    for unit in ("s", "ms", "us", "ns"):
        index = pd.DatetimeIndex(["2026-03-02 10:34:00"], tz="UTC").as_unit(unit)
        assert_on_grid(index, interval="1m", context=f"unit={unit}")


def test_multi_page_overlap_is_deduped_without_loss():
    """
    多页重叠：端点 endTime 为包含语义，相邻页天然可能重复一条。

    断言：重叠部分按时间主键去重，且不丢失任何一分钟。
    """
    pages = {
        to_ms("2026-01-01 00:00"): minute_klines(start="2026-01-01 00:00", count=3),
        # 第二页从 00:02 重新给出（与上一页重叠一条），再补 00:03、00:04。
        to_ms("2026-01-01 00:03"): minute_klines(start="2026-01-01 00:02", count=3)[1:],
    }

    def handler(url, params):
        start_ms = params["startTime"]
        return FakeResponse(status_code=200, payload=pages.get(start_ms, []))

    frame, complete, note, session, _ = _fetch(handler, end="2026-01-01 00:05", limit=3)
    assert list(frame.index) == list(
        pd.date_range("2026-01-01 00:00", periods=5, freq="1min", tz="UTC")
    )
    assert frame.index.is_unique


def test_duplicate_page_terminates_and_reports_incomplete():
    """重复页（整页都已见过）必须终止并报错，不能无限翻页。"""
    def handler(url, params):
        return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:00", count=2))

    with pytest.raises(PaginationError, match="完全重复的分页"):
        _fetch(handler)


def test_cursor_not_advancing_terminates():
    """
    游标不推进：端点忽略 startTime 并返回更早的数据。

    断言：立即终止，绝不出现“看似在翻页实则原地打转”的死循环。
    """
    calls = {"n": 0}

    def handler(url, params):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:02", count=2))
        # 第二页给出更早的数据：最大时间戳反而回退了。
        return FakeResponse(status_code=200, payload=minute_klines(start="2026-01-01 00:00", count=2))

    with pytest.raises(PaginationError, match="游标未推进"):
        _fetch(handler)


def test_budget_exhausted_terminates_and_is_not_complete():
    """
    请求预算耗尽：必须以 BudgetExhaustedError 终止，而不是返回已取到的一部分并自称完整。
    """
    def handler(url, params):
        start_ms = params["startTime"]
        offset = (start_ms - to_ms("2026-01-01 00:00")) // 60_000
        return FakeResponse(
            status_code=200,
            payload=minute_klines(start=f"2026-01-01 00:{offset:02d}", count=1),
        )

    with pytest.raises(BudgetExhaustedError, match="请求预算已耗尽"):
        _fetch(handler, end="2026-01-01 00:10", limit=1, max_requests=3)


def test_http_error_is_not_silent_empty():
    """不可重试的 HTTP 错误必须立即失败，不得当作空数据。"""
    def handler(url, params):
        return FakeResponse(status_code=400, payload={"code": -1121, "msg": "Invalid symbol."})

    with pytest.raises(HttpStatusError, match="HTTP 400"):
        _fetch(handler)


def test_retryable_status_is_retried_then_fails():
    """
    限流状态按有限重试处理；重试耗尽后必须失败，不以无限重试绕过。
    """
    def handler(url, params):
        return FakeResponse(status_code=429, payload={"code": -1003, "msg": "Too many requests"})

    with pytest.raises(HttpStatusError, match="HTTP 429 重试耗尽"):
        _fetch(handler, max_retries=2)


def test_business_error_under_http_200_is_detected():
    """HTTP 200 但载荷是业务错误对象：必须按失败处理（状态与业务状态分别检查）。"""
    def handler(url, params):
        return FakeResponse(status_code=200, payload={"code": -1121, "msg": "Invalid symbol."})

    with pytest.raises(BusinessStatusError, match="业务错误"):
        _fetch(handler)


def test_malformed_json_is_detected():
    """响应不是合法 JSON：报结构错误，不返回空数据。"""
    def handler(url, params):
        return FakeResponse(status_code=200, text="<html>maintenance</html>")

    with pytest.raises(ResponseSchemaError, match="不是合法 JSON"):
        _fetch(handler)


def test_incomplete_row_fields_are_detected():
    """行字段不全（少于 6 列）必须失败；缺字段不是缺记录。"""
    def handler(url, params):
        return FakeResponse(status_code=200, payload=[[to_ms("2026-01-01 00:00"), "1", "2", "0.5"]])

    with pytest.raises(ResponseSchemaError, match="需至少 6 个字段"):
        _fetch(handler)


def test_non_integer_open_time_is_detected():
    """open_time 必须是整数毫秒；浮点/字符串会带来静默精度问题，直接拒绝。"""
    row = kline_row(open_time_ms=to_ms("2026-01-01 00:00"), open_=1, high=2, low=0.5, close=1.5, volume=1)
    row[0] = float(row[0])

    def handler(url, params):
        return FakeResponse(status_code=200, payload=[row])

    with pytest.raises(ResponseSchemaError, match="open_time 必须是整数毫秒"):
        _fetch(handler)


def test_every_page_keeps_the_task_cutoff():
    """
    每页都必须带本次任务的固定截止（endTime），否则翻着翻着就追到“现在”了。
    """
    pages = {
        to_ms("2026-01-01 00:00"): minute_klines(start="2026-01-01 00:00", count=2),
        to_ms("2026-01-01 00:02"): minute_klines(start="2026-01-01 00:02", count=2),
    }

    def handler(url, params):
        return FakeResponse(status_code=200, payload=pages.get(params["startTime"], []))

    frame, complete, note, session, _ = _fetch(handler, end="2026-01-01 00:04", limit=2)
    expected_end_ms = to_ms("2026-01-01 00:04") - 1
    assert len(session.calls) >= 2
    for _url, params in session.calls:
        assert params["endTime"] == expected_end_ms
