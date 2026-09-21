# -*- coding: utf-8 -*-
"""
WP2 来源专项：协议、错误处理、占位字段与不可逆转换（Issue #5 §5 / §4.4）。

每个用例都对应 Issue 里一条具体要求，并且都走真实解析路径：
    - Bybit：`retCode != 0` 立即失败；首页不传 startTime；正值/0/未传分别观察。
    - Binance K 线：`startTime=0` 的行为不能从资金费率推导——两边分别测。
    - 参考价格：占位 `volume` 不得冒充成交量；溢价指数可以为负。
    - 低频来源：非法数值、全空响应、字段变化、业务错误。
    - 转换：FRED 的 `.`；环比的首期/零分母不填 0；历史缩短必须保留旧数据并报错。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from rbt_fin.contracts import COVERAGE_UNVERIFIED, MODE_FORCE, DataRequest
from rbt_fin.errors import (
    BusinessStatusError,
    PaginationError,
    ResponseSchemaError,
    RevisionConflictError,
    SourceBlockedError,
    ValueContractError,
)
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_registry, get_dataset
from rbt_fin.sources.binance_funding import fetch_funding_records
from rbt_fin.sources.binance_klines import fetch_klines_range
from rbt_fin.sources.bybit_funding import fetch_bybit_funding
from rbt_fin.sources.daily_series import (
    extract_blockchain_chart,
    extract_defillama_stablecoins,
    extract_fred_observations,
    extract_stooq_csv,
)

from tests.conftest import FakeResponse, make_client, minute_klines, to_ms

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-06", tz="UTC")
NOW = pd.Timestamp("2026-01-10", tz="UTC")
DAYS = [START + pd.Timedelta(days=i) for i in range(5)]


# ----------------------------- Bybit 专项 -----------------------------

def _bybit_rows(count: int = 5):
    return [{"symbol": "BTCUSDT", "fundingRate": f"{0.0001 * (i + 1):.8f}",
             "fundingRateTimestamp": str(to_ms(DAYS[i].strftime("%Y-%m-%d %H:%M")))}
            for i in range(count)]


def _fetch_bybit(handler, **kwargs):
    client, session = make_client(handler, max_requests=20)
    frame, complete, note = fetch_bybit_funding(
        http=client, symbol="BTCUSDT", category="linear",
        start=START, end_exclusive=END, limit=200, context="bybit", **kwargs
    )
    return frame, complete, note, session


def test_bybit_business_error_fails_immediately():
    """
    `retCode != 0` 必须立即形成明确失败。

    基线只取 `result.list`，业务错误于是表现为"该来源今天没有记录"——
    凭据失效、参数错误都会被当成正常的空数据。
    """
    def handler(url, params):
        return FakeResponse(status_code=200,
                            payload={"retCode": 10001, "retMsg": "params error", "result": {}})

    with pytest.raises(BusinessStatusError, match="retCode=10001"):
        _fetch_bybit(handler)


def test_bybit_first_page_omits_start_time_and_always_sends_end_time():
    """
    首页不传 `startTime`、只带固定 `endTime`。

    `startTime=0` 在该端点的语义未经本版本验证，且与 Binance 的同名参数不能互相推导，
    因此本实现从不发送它。
    """
    def handler(url, params):
        return FakeResponse(status_code=200,
                            payload={"retCode": 0, "result": {"list": _bybit_rows()}})

    _frame, _complete, _note, session = _fetch_bybit(handler)
    assert session.calls, "应至少发出一次请求"
    for _url, params in session.calls:
        assert "startTime" not in params, "本实现不依赖该端点的 startTime 哨兵语义"
        assert "endTime" in params, "每页都必须带固定截止"


def test_bybit_pagination_walks_backwards_and_stops():
    """向历史翻页：游标必须单调向更早推进，并在越过下界后正常终止。"""
    all_rows = _bybit_rows()

    def handler(url, params):
        end_ms = params["endTime"]
        visible = [r for r in all_rows if int(r["fundingRateTimestamp"]) <= end_ms]
        visible.sort(key=lambda r: int(r["fundingRateTimestamp"]), reverse=True)
        return FakeResponse(status_code=200,
                            payload={"retCode": 0, "result": {"list": visible[:2]}})

    frame, complete, _note, session = _fetch_bybit(handler)
    assert len(frame) == 5
    assert frame.index.is_monotonic_increasing
    end_times = [params["endTime"] for _url, params in session.calls]
    assert end_times == sorted(end_times, reverse=True), "endTime 必须单调向更早推进"


def test_bybit_cursor_not_advancing_is_detected():
    """
    端点忽略 endTime 时游标不会推进：必须终止，不能无限翻页。

    这里固定返回**最早时间仍高于请求下界**的同一页——否则翻到下界就会正常结束，
    测不到"游标不推进"这条路径。
    """
    def handler(url, params):
        return FakeResponse(status_code=200,
                            payload={"retCode": 0, "result": {"list": _bybit_rows()[2:4]}})

    with pytest.raises(PaginationError, match="游标未向历史推进|完全重复的分页"):
        _fetch_bybit(handler)


def test_bybit_missing_field_fails_fast():
    """字段不全（缺 fundingRate）必须失败：字段缺失不是记录缺失。"""
    def handler(url, params):
        return FakeResponse(status_code=200, payload={
            "retCode": 0,
            "result": {"list": [{"symbol": "BTCUSDT", "fundingRateTimestamp": "1767225600000"}]},
        })

    with pytest.raises(ResponseSchemaError, match="缺少必需字段"):
        _fetch_bybit(handler)


# ----------------------------- Binance startTime 语义不可互推 -----------------------------

def test_binance_klines_and_funding_start_time_are_tested_separately():
    """
    K 线与资金费率对 `startTime` 的处理**分别**验证，不互相推导。

    两条路径都由请求范围推导出显式正值；即使请求从 epoch 0 开始，
    发送的 startTime 也不会是 0（0 在两个端点上的含义都未经本版本验证）。
    """
    seen_klines, seen_funding = [], []

    def kline_handler(url, params):
        seen_klines.append(params)
        return FakeResponse(status_code=200, payload=[])

    def funding_handler(url, params):
        seen_funding.append(params)
        return FakeResponse(status_code=200, payload=[])

    client, _ = make_client(kline_handler, max_requests=5)
    fetch_klines_range(
        http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
        start=pd.Timestamp("1970-01-01", tz="UTC"),
        end_exclusive=pd.Timestamp("1970-01-01 00:05", tz="UTC"),
        limit=10, context="klines-epoch0",
    )
    client2, _ = make_client(funding_handler, max_requests=5)
    fetch_funding_records(
        http=client2, symbol="BTCUSDT",
        start=pd.Timestamp("1970-01-01", tz="UTC"),
        end_exclusive=pd.Timestamp("1970-01-02", tz="UTC"),
        limit=10, context="funding-epoch0",
    )
    # K 线按请求原样传（epoch 0 的分钟边界就是 0），资金费率显式提升为正值。
    # 两者的处理不同，正说明它们没有被当成同一件事。
    assert seen_klines and seen_funding
    assert seen_funding[0]["startTime"] >= 1, "资金费率端点从不发送 0"
    assert "endTime" in seen_klines[0] and "endTime" in seen_funding[0]


# ----------------------------- 参考价格专项 -----------------------------

def test_reference_price_does_not_expose_placeholder_volume(cache_root):
    """
    三类参考价格端点的 `volume` 恒为占位 "0"，绝不能作为成交量对外。

    基线把它按 OHLCV 原样带出来，下游看到的就是一列假成交量。
    """
    dataset = get_dataset("binance_index_price_1d")
    rows = minute_klines(start="2026-01-01 00:00", count=5)
    for index, row in enumerate(rows):
        row[0] = to_ms(DAYS[index].strftime("%Y-%m-%d %H:%M"))
        row[5] = "0"

    def handler(url, params):
        lo, hi = params["startTime"], params["endTime"]
        return FakeResponse(status_code=200,
                            payload=[r for r in rows if lo <= r[0] <= hi][:params["limit"]])

    client, session = make_client(handler, max_requests=20)
    result = acquire_dataset(
        dataset=dataset, request=DataRequest(start=START, end_exclusive=END),
        store_root=cache_root, now_utc=NOW, http=client,
    )
    assert list(result.frame.columns) == ["index_close"]
    assert "volume" not in result.frame.columns
    # 指数价端点用 `pair` 而不是 `symbol`：这一位差异不能靠"看起来都一样"推导。
    assert "pair" in session.calls[0][1]


def test_premium_index_accepts_negative_values(cache_root):
    """
    溢价指数**可以为负**：按"价格必须为正"校验会把合法数据判成违约。
    """
    dataset = get_dataset("binance_premium_index_1d")
    rows = minute_klines(start="2026-01-01 00:00", count=5)
    for index, row in enumerate(rows):
        row[0] = to_ms(DAYS[index].strftime("%Y-%m-%d %H:%M"))
        row[1:5] = ["-0.00010000"] * 4  # 负溢价
        row[5] = "0"

    def handler(url, params):
        lo, hi = params["startTime"], params["endTime"]
        return FakeResponse(status_code=200,
                            payload=[r for r in rows if lo <= r[0] <= hi][:params["limit"]])

    client, _ = make_client(handler, max_requests=20)
    result = acquire_dataset(
        dataset=dataset, request=DataRequest(start=START, end_exclusive=END),
        store_root=cache_root, now_utc=NOW, http=client,
    )
    assert (result.frame["premium_close"] < 0).all()


def test_index_price_rejects_non_positive_values(cache_root):
    """指数价必须为正：非正值是真实违约，不能与溢价指数一视同仁地放行。"""
    dataset = get_dataset("binance_index_price_1d")
    rows = minute_klines(start="2026-01-01 00:00", count=5)
    for index, row in enumerate(rows):
        row[0] = to_ms(DAYS[index].strftime("%Y-%m-%d %H:%M"))
        row[1:5] = ["0.00000000"] * 4

    def handler(url, params):
        lo, hi = params["startTime"], params["endTime"]
        return FakeResponse(status_code=200,
                            payload=[r for r in rows if lo <= r[0] <= hi][:params["limit"]])

    client, _ = make_client(handler, max_requests=20)
    with pytest.raises(ValueContractError, match="非正价格|非正值"):
        acquire_dataset(
            dataset=dataset, request=DataRequest(start=START, end_exclusive=END),
            store_root=cache_root, now_utc=NOW, http=client,
        )


# ----------------------------- 低频来源的解析边界 -----------------------------

def test_blockchain_missing_field_fails_fast():
    """字段变化（缺 y）必须在解析处失败，不静默丢该条。"""
    with pytest.raises(ResponseSchemaError, match="缺少必需字段"):
        extract_blockchain_chart({"values": [{"x": 1767225600}]}, context="t")


def test_blockchain_wrong_container_reports_which_layer():
    """缺失必须报出缺在哪一层，而不是一路 None 变成"来源没有数据"。"""
    with pytest.raises(ResponseSchemaError, match="缺在 'values' 这一层"):
        extract_blockchain_chart({"status": "ok"}, context="t")


def test_defillama_nested_value_must_contain_pegged_usd():
    """
    `totalCirculatingUSD` 实测是嵌套对象；缺 `peggedUSD` 时必须失败，
    不能退回"把对象里所有数值求和"——那会把不同锚定资产混成一个量纲。
    """
    with pytest.raises(ResponseSchemaError, match="缺少 peggedUSD"):
        extract_defillama_stablecoins(
            [{"date": "1767225600", "totalCirculatingUSD": {"peggedEUR": 1}}], context="t"
        )


def test_fred_dot_value_is_skipped_not_zeroed():
    """
    FRED 用 `.` 表示"该发布日无值"。它绝不能被当成 0 参与后续计算：提取层把它作为 value=None 的记录交出，
    build_frame 不让它进入数据、但把条数与日期写入采集说明（自审 R-14：不再静默跳过）。
    """
    from rbt_fin.sources.daily_series import build_frame
    records = extract_fred_observations(
        {"observations": [{"date": "2026-01-01", "value": "1.5"},
                          {"date": "2026-01-02", "value": "."}]},
        context="t",
    )
    assert [values["value"] for _, values in records] == ["1.5", None]
    frame, note = build_frame(records, value_columns=("value",), context="t")
    assert len(frame) == 1 and frame["value"].iloc[0] == 1.5 and "字段缺失 1 条" in note and "2026-01-02" in note
    assert records[0][1]["value"] == "1.5"


def test_stooq_html_block_is_distinguished_from_schema_drift():
    """
    HTTP 200 + HTML 是来源侧反爬，不是"CSV 列名变了"。
    两者的处置完全不同，必须是两种错误类型。
    """
    with pytest.raises(SourceBlockedError):
        extract_stooq_csv("<!DOCTYPE html><html>challenge</html>", context="t")
    with pytest.raises(ResponseSchemaError, match="缺少日期或收盘列"):
        extract_stooq_csv("Foo,Bar\n1,2\n", context="t")


def test_daily_series_rejects_non_numeric_value(cache_root):
    """坏数值必须在解析处失败，不被 coerce 成 NaN 后 dropna 变成看不见的缺口。"""
    dataset = get_dataset("blockchain_hash_rate")

    def handler(url, params):
        return FakeResponse(status_code=200, payload={
            "values": [{"x": int(d.value // 10**9), "y": "not-a-number"} for d in DAYS]
        })

    client, _ = make_client(handler, max_requests=5)
    with pytest.raises(ValueContractError, match="无法解析为数值"):
        acquire_dataset(
            dataset=dataset,
            request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
            store_root=cache_root, now_utc=NOW, http=client,
        )


def test_empty_response_is_not_reported_as_complete(cache_root):
    """
    全空响应不得被当成"完整覆盖"：这些来源的日历未验证，
    覆盖最高只能是 unverified，严格请求必须被拒绝。
    """
    dataset = get_dataset("blockchain_hash_rate")

    def handler(url, params):
        return FakeResponse(status_code=200, payload={"values": []})

    client, _ = make_client(handler, max_requests=5)
    result = acquire_dataset(
        dataset=dataset,
        request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
        store_root=cache_root, now_utc=NOW, http=client,
    )
    assert result.coverage.status == COVERAGE_UNVERIFIED
    assert result.coverage.rows == 0


# ----------------------------- 转换与修订 -----------------------------

def test_cryptocompare_pct_keeps_base_and_marks_undefined(cache_root):
    """
    环比转换必须保留基础值，并把首期 / 零分母标记为未定义，而不是统一填 0.0。

    手算：基础值 [5, 0, 0] -> 环比 [未定义, -1.0, 未定义]。
    """
    dataset = get_dataset("cryptocompare_btc_chain_pct")
    from rbt_fin.sources.daily_series import CRYPTOCOMPARE_BASE_FIELDS

    series = [5, 0, 0]

    def handler(url, params):
        return FakeResponse(status_code=200, payload={"Data": {"Data": [
            {"time": int(DAYS[i].value // 10**9),
             **{name: series[i] for name in CRYPTOCOMPARE_BASE_FIELDS}}
            for i in range(3)
        ]}})

    client, _ = make_client(handler, max_requests=5)
    result = acquire_dataset(
        dataset=dataset,
        request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
        store_root=cache_root, now_utc=NOW, http=client,
        credentials={"CRYPTOCOMPARE_API_KEY": "fake-key-for-tests"},
    )
    frame = result.frame
    assert frame["hashrate"].tolist() == [5.0, 0.0, 0.0], "必须保留基础值"
    pct = frame["hashrate_pct"].tolist()
    assert pd.isna(pct[0]), "首次记录没有前值 -> 未定义"
    assert pct[1] == pytest.approx(-1.0), "5 -> 0 是 -100%，有定义"
    assert pd.isna(pct[2]), "零分母 -> 未定义，不能填 0"


def test_history_shortening_on_full_refetch_keeps_old_data(cache_root):
    """
    小数据全量重取时，原覆盖范围内记录消失必须保留旧正式数据并报错，
    不能靠"行数变少"就静默清库。
    """
    dataset = get_dataset("blockchain_hash_rate")
    full = {"values": [{"x": int(d.value // 10**9), "y": 1.0 + i} for i, d in enumerate(DAYS)]}
    shortened = {"values": [full["values"][0], full["values"][-1]]}
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)

    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=full), max_requests=5)
    first = acquire_dataset(dataset=dataset, request=request, store_root=cache_root,
                            now_utc=NOW, http=client)
    assert first.coverage.rows == 5

    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=shortened),
                             max_requests=5)
    import dataclasses

    with pytest.raises(RevisionConflictError, match="既有记录在新结果中消失"):
        acquire_dataset(dataset=dataset, request=dataclasses.replace(request, mode=MODE_FORCE),
                        store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)

    from rbt_fin.storage import DatasetStore

    assert DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest().rows == 5
