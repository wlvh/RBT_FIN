# -*- coding: utf-8 -*-
"""
全注册表参数化测试：每个注册项都走 acquire -> persist -> load -> cache-hit。

Issue #5 §8「全注册表」组要求：
    25 项各有 acquire → persist → load → cache-hit 测试；
    按**实际采集函数**共享固定 HTTP 响应；FRED 别名映射和参数一致。

这里按采集函数族共享一份固定协议响应（形状取自实测，见 `sources/daily_series.py` 顶部注释），
让真实的解析 / 规范化 / 校验 / 内容寻址存储全部运行，而不是替换成永远正确的 DataFrame。
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List

import pandas as pd
import pytest

from rbt_fin.contracts import (
    COVERAGE_COMPLETE,
    COVERAGE_UNVERIFIED,
    MODE_OFFLINE,
    DataRequest,
)
from rbt_fin.errors import ConfigError
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import (
    BLOCKCHAIN_CHARTS,
    DATASET_ALIASES,
    FRED_SERIES,
    build_registry,
    get_dataset,
    list_registered_ids,
)
from rbt_fin.sources.daily_series import CRYPTOCOMPARE_BASE_FIELDS

from tests.conftest import FakeResponse, make_client, minute_klines, to_ms

NOW = pd.Timestamp("2026-01-10 00:00", tz="UTC")
START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-06", tz="UTC")
DAYS = [START + pd.Timedelta(days=i) for i in range(5)]

# 每个需要凭据的数据集在测试里使用的**虚构**凭据。
# 绝不使用任何真实 key：测试凭据只用于走通鉴权分支。
FAKE_CREDENTIALS = {
    "FRED_API_KEY": "fake-fred-key-for-tests",
    "GLASSNODE_API_KEY": "fake-glassnode-key-for-tests",
    "CRYPTOCOMPARE_API_KEY": "fake-cryptocompare-key-for-tests",
}


def _epoch_seconds(stamp: pd.Timestamp) -> int:
    return int(stamp.value // 1_000_000_000)


def _daily_payload(extractor: str) -> Any:
    """按提取器返回一份固定的协议载荷（形状取自实测响应）。"""
    if extractor == "blockchain_chart":
        return {"status": "ok", "values": [{"x": _epoch_seconds(d), "y": 1.0 + i}
                                           for i, d in enumerate(DAYS)]}
    if extractor == "fear_greed":
        # timestamp 与 value 实测都是字符串。
        return {"data": [{"timestamp": str(_epoch_seconds(d)), "value": str(40 + i)}
                         for i, d in enumerate(DAYS)]}
    if extractor == "wikipedia_pageviews":
        return {"items": [{"timestamp": d.strftime("%Y%m%d") + "00", "views": 1000 + i}
                          for i, d in enumerate(DAYS)]}
    if extractor == "defillama_stablecoins":
        # totalCirculatingUSD 实测是嵌套对象，不是数字。
        return [{"date": str(_epoch_seconds(d)),
                 "totalCirculatingUSD": {"peggedUSD": 1_000_000 + i}} for i, d in enumerate(DAYS)]
    if extractor == "fred_observations":
        # 第 3 条用 '.'：FRED 用它表示"该发布日无值"，必须被显式跳过而不是静默变 0。
        return {"observations": [
            {"date": d.strftime("%Y-%m-%d"), "value": "." if i == 2 else str(1.5 + i)}
            for i, d in enumerate(DAYS)
        ]}
    if extractor == "glassnode_metric":
        return [{"t": _epoch_seconds(d), "v": 900000 + i} for i, d in enumerate(DAYS)]
    if extractor == "cryptocompare_chain":
        return {"Response": "Success", "Data": {"Data": [
            {"time": _epoch_seconds(d), **{name: 10 + i for name in CRYPTOCOMPARE_BASE_FIELDS}}
            for i, d in enumerate(DAYS)
        ]}}
    raise AssertionError(f"未覆盖的提取器: {extractor}")


def _build_session_for(dataset):
    """为一个数据集构造它自己那族协议的受控端点。"""
    params = dataset.params

    if "extractor" in params:
        if params["extractor"] == "stooq_csv":
            text = "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie\n" + "\n".join(
                f"{d.strftime('%Y-%m-%d')},1,2,0.5,{2000 + i}" for i, d in enumerate(DAYS)
            )

            def handler(url, request_params):
                return FakeResponse(status_code=200, text=text)

            return handler
        payload = _daily_payload(params["extractor"])

        def handler(url, request_params):
            return FakeResponse(status_code=200, payload=payload)

        return handler

    if "value_column" in params:  # 参考价格：与 K 线同形状，volume 为占位 "0"
        rows = minute_klines(start="2026-01-01 00:00", count=5)
        for index, row in enumerate(rows):
            row[0] = to_ms(DAYS[index].strftime("%Y-%m-%d %H:%M"))
            row[5] = "0"  # 占位成交量：必须不被当成成交量带出

        def handler(url, request_params):
            lo, hi = request_params["startTime"], request_params["endTime"]
            return FakeResponse(status_code=200,
                                payload=[r for r in rows if lo <= r[0] <= hi][:request_params["limit"]])

        return handler

    if "category" in params:  # Bybit：降序列表 + retCode
        rows = [{"symbol": params["symbol"], "fundingRate": f"{0.0001 * (i + 1):.8f}",
                 "fundingRateTimestamp": str(to_ms(d.strftime("%Y-%m-%d %H:%M")))}
                for i, d in enumerate(DAYS)]

        def handler(url, request_params):
            end_ms = request_params["endTime"]
            visible = [r for r in rows if int(r["fundingRateTimestamp"]) <= end_ms]
            visible.sort(key=lambda r: int(r["fundingRateTimestamp"]), reverse=True)
            return FakeResponse(status_code=200,
                                payload={"retCode": 0, "retMsg": "OK",
                                         "result": {"list": visible[:request_params["limit"]]}})

        return handler

    if dataset.dataset_id.endswith("_raw"):  # Binance 资金费率：升序数组
        rows = [{"symbol": params["symbol"], "fundingTime": to_ms(d.strftime("%Y-%m-%d %H:%M")),
                 "fundingRate": f"{0.0001 * (i + 1):.8f}"} for i, d in enumerate(DAYS)]

        def handler(url, request_params):
            lo, hi = request_params["startTime"], request_params["endTime"]
            return FakeResponse(status_code=200,
                                payload=[r for r in rows if lo <= r["fundingTime"] <= hi][:request_params["limit"]])

        return handler

    # 分钟行情
    rows = minute_klines(start="2026-01-01 00:00", count=60 * 24 * 5)

    def handler(url, request_params):
        lo, hi = request_params["startTime"], request_params["endTime"]
        return FakeResponse(status_code=200,
                            payload=[r for r in rows if lo <= r[0] <= hi][:request_params["limit"]])

    return handler


def _local_csv(tmp_path, dataset) -> Dict[str, str]:
    """为本地导入数据集准备一份用户 CSV。"""
    path = tmp_path / f"{dataset.dataset_id}.csv"
    path.write_text(
        "date,value\n" + "\n".join(f"{d.isoformat()},{1.0 + i}" for i, d in enumerate(DAYS)) + "\n",
        encoding="utf-8",
    )
    return {"csv_path": str(path)}


ALL_IDS = list_registered_ids()


@pytest.mark.parametrize("dataset_id", ALL_IDS)
def test_every_registered_id_acquires_persists_loads_and_hits_cache(dataset_id, cache_root, tmp_path):
    """
    每个注册标识：acquire -> persist -> load -> cache-hit 全程走真实代码。

    第二次请求必须**零网络**：把 HTTP 客户端换成"一被调用就失败"的替身，
    因此"缓存命中没有取数"是被断言出来的，而不是看日志推断的。
    """
    dataset = get_dataset(dataset_id)
    request = DataRequest(
        start=START, end_exclusive=END,
        # 低频来源的发布日历未验证，覆盖最高只到 unverified；严格请求会被拒绝。
        require_full_coverage=dataset.calendar_verified,
    )
    local_inputs = _local_csv(tmp_path, dataset) if not dataset.requires_network else {}
    client, session = (make_client(_build_session_for(dataset), max_requests=60)
                       if dataset.requires_network else (None, None))

    first = acquire_dataset(
        dataset=dataset, request=request, store_root=cache_root, now_utc=NOW,
        http=client, credentials=FAKE_CREDENTIALS, local_inputs=local_inputs,
    )
    assert first.used_cache is False
    assert first.coverage.rows > 0, f"{dataset_id}: 首次采集应取得记录"
    assert first.coverage.status in (COVERAGE_COMPLETE, COVERAGE_UNVERIFIED)

    # 落盘可读回：manifest 指向的内容按 schema 正确解析，行数一致。
    from rbt_fin.storage import DatasetStore

    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    assert manifest is not None
    reloaded = store.read_frame(manifest, verify_content=True)
    assert list(reloaded.columns) == list(dataset.value_columns)

    # cache-hit：offline 模式必然零网络，且必须能满足同一请求。
    second = acquire_dataset(
        dataset=dataset,
        request=DataRequest(start=START, end_exclusive=END, mode=MODE_OFFLINE,
                            require_full_coverage=dataset.calendar_verified),
        store_root=cache_root, now_utc=NOW, http=None, local_inputs=local_inputs,
    )
    assert second.used_cache is True
    assert second.content_sha256 == first.content_sha256


def test_all_baseline_dataset_ids_are_preserved():
    """
    基线的 25 个 `dataset_id` 必须全部仍可解析——原 id 的概览用途不能丢。
    """
    baseline = {
        "binance_funding_btcusdt", "binance_index_price_1d", "binance_premium_index_1d",
        "binance_mark_price_1d", "bybit_funding_btcusdt",
        *(f"blockchain_{chart.replace('-', '_')}" for chart in BLOCKCHAIN_CHARTS),
        "altme_fng", "wikipedia_bitcoin_pageviews", "defillama_stablecoins_total",
        "macro_dxy", "macro_spx", "macro_us10y", "macro_vix",
        "fred_t10y2y", "fred_dgs10", "fred_dtwexbgs",
        "macro_gold", "glassnode_active_addresses_btc", "cryptocompare_btc_chain_pct",
        "binance_metrics_local",
    }
    assert len(baseline) == 25
    missing = sorted(name for name in baseline if name not in set(ALL_IDS))
    assert missing == [], f"基线 dataset_id 丢失: {missing}"


# ----------------------------- FRED 别名与零重复下载 -----------------------------

@pytest.mark.parametrize("alias,base", sorted(DATASET_ALIASES.items()))
def test_alias_resolves_to_same_dataset_and_params(alias, base):
    """别名与基础名必须解析到同一份定义：同一份参数、同一个缓存目录。"""
    assert get_dataset(alias).dataset_id == base
    assert dict(get_dataset(alias).params) == dict(get_dataset(base).params)
    assert dict(get_dataset(alias).identity) == dict(get_dataset(base).identity)


def test_duplicate_fred_series_does_not_download_twice(cache_root):
    """
    重复的 FRED 序列不重复下载：第二个注册名的请求必须零网络。

    基线里 DGS10 与 DTWEXBGS 各有两个注册名，各自独立发请求。
    """
    base_id = "macro_us10y"          # DGS10 的基础注册名
    alias_id = "fred_dgs10"          # 同一序列的另一个注册名
    dataset = get_dataset(base_id)
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)

    client, session = make_client(_build_session_for(dataset), max_requests=20)
    acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW,
                    http=client, credentials=FAKE_CREDENTIALS)
    requests_after_first = len(session.calls)
    assert requests_after_first >= 1

    def _must_not_be_called(url, params):
        raise AssertionError("同一 FRED 序列的第二个注册名不得重复下载")

    alias_client, alias_session = make_client(_must_not_be_called, max_requests=20)
    result = acquire_dataset(
        dataset=get_dataset(alias_id), request=request, store_root=cache_root,
        now_utc=NOW + dt.timedelta(seconds=1), http=alias_client, credentials=FAKE_CREDENTIALS,
    )
    assert result.used_cache is True
    assert alias_session.calls == []


# ----------------------------- 凭据边界 -----------------------------

@pytest.mark.parametrize(
    "dataset_id,credential_env",
    [("macro_us10y", "FRED_API_KEY"),
     ("glassnode_active_addresses_btc", "GLASSNODE_API_KEY"),
     ("cryptocompare_btc_chain_pct", "CRYPTOCOMPARE_API_KEY")],
)
def test_missing_credential_fails_fast_instead_of_reporting_zero_missing(
    dataset_id, credential_env, cache_root
):
    """
    缺凭据必须 Fail Fast，绝不能表现成"零缺失"的正常输出。

    也顺带证明本库没有任何内置备用 key：不提供凭据就是取不到数。
    """
    dataset = get_dataset(dataset_id)
    client, _ = make_client(_build_session_for(dataset), max_requests=10)
    with pytest.raises(ConfigError, match=credential_env):
        acquire_dataset(
            dataset=dataset,
            request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
            store_root=cache_root, now_utc=NOW, http=client, credentials={},
        )


def test_credential_value_never_enters_params_or_manifest(cache_root):
    """凭据值不得进入 dataset.params 或 manifest —— 只有环境变量**名字**会被记录。"""
    dataset = get_dataset("macro_us10y")
    client, _ = make_client(_build_session_for(dataset), max_requests=20)
    acquire_dataset(
        dataset=dataset,
        request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
        store_root=cache_root, now_utc=NOW, http=client, credentials=FAKE_CREDENTIALS,
    )
    from rbt_fin.storage import DatasetStore

    manifest = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest()
    serialized = manifest.to_json_bytes().decode("utf-8")
    for secret in FAKE_CREDENTIALS.values():
        assert secret not in serialized
    assert "FRED_API_KEY" in serialized, "凭据的环境变量名应当被记录，便于排查"
