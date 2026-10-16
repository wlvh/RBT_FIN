#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data freshness probe
--------------------
对 build_registry() 中的 19 个数据源各发一次最小化 API 请求，
记录每个源当前可见的最新 UTC 日期 + 当前 UTC 时间戳。
设计目标：
- 单次运行 < 60 秒、HTTP 流量 < 5 MB，适合每小时跑一次（routine / launchd 均可）。
- 每条观测一行 JSON 写到 stdout，由调用方追加到 data_freshness_probe.jsonl。
- 用于跟踪每个源的发布滞后模式：UTC 0:00 转日时 Binance 何时出现新一日；
  blockchain.com / wikipedia / altme 隔多久才补齐前一日；
  FRED 各序列与 LBMA PM fix 各自的发布窗口。

输出字段：
- observed_at_utc:        探针触发的 UTC 时刻（ISO8601 秒精度）。
- current_utc_date:       探针触发当下的 UTC 日期。
- dataset_id:             与主 registry 一致的稳定 ID。
- last_date:              该源当前可见的最新 UTC 日期；失败为 null。
- rows:                   探针窗口内返回的记录数（用于排查 API 半故障）。
- lag_days:               current_utc_date - last_date，单位天；失败为 null。
- includes_current_day:   last_date == current_utc_date，对 24/7 类源最有信号。
- day_convention:         源对“一天”的语义：utc / et-trading / london-trading。静态。
- release_utc_hour_prior: 当日数据通常 UTC 何时可见的先验估计（0-23）。静态。
                          这只是先验，分析阶段会用实证分布修正。
- ok:                     探针是否成功。
- error:                  失败时的异常类型与消息；成功为 null。

注意：lag_days 是按 current_utc_date 直接相减的“UTC 视角滞后”。
对 day_convention != "utc" 的源（FRED / LBMA），需结合 release_utc_hour_prior
才能算出真实的“业务日发布滞后”，否则会有 8-15 小时的“假滞后”。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from data_layer_unified_opt import http_get, load_env_from_dotenv, to_utc_date

# 探针回溯窗口：14 天足以覆盖 FRED/LBMA 的最长发布滞后，且响应数据量很小。
PROBE_LOOKBACK_DAYS = 14

# 每个源对“一天”的静态语义 + 先验发布时点。
# release_utc_hour_prior 是“当日业务数据通常 UTC 几点变得可拉到”的初值（0-23）；
# 阶段二的实证分析会用真实分布替换这些先验值。
DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    # 24/7 自然日类：UTC 0:00 起开始累积，UTC 24:00 才完整。release_utc=23 表示“当日 23:00 后基本不再变”。
    "binance_funding_btcusdt":      {"day_convention": "utc",            "release_utc_hour_prior": 23},
    "binance_index_price_1d":       {"day_convention": "utc",            "release_utc_hour_prior": 23},
    "binance_premium_index_1d":     {"day_convention": "utc",            "release_utc_hour_prior": 23},
    # blockchain.com 多数指标当日数据 UTC 0:00 后约几小时才稳定，挑保守值 6。
    "blockchain_hash_rate":         {"day_convention": "utc",            "release_utc_hour_prior": 6},
    "blockchain_difficulty":        {"day_convention": "utc",            "release_utc_hour_prior": 6},
    "blockchain_mempool_size":      {"day_convention": "utc",            "release_utc_hour_prior": 0},
    "blockchain_n_transactions":    {"day_convention": "utc",            "release_utc_hour_prior": 6},
    "blockchain_transaction_fees":  {"day_convention": "utc",            "release_utc_hour_prior": 6},
    "blockchain_n_unique_addresses": {"day_convention": "utc",           "release_utc_hour_prior": 6},
    "altme_fng":                    {"day_convention": "utc",            "release_utc_hour_prior": 0},
    "wikipedia_bitcoin_pageviews":  {"day_convention": "utc",            "release_utc_hour_prior": 1},
    "defillama_stablecoins_total":  {"day_convention": "utc",            "release_utc_hour_prior": 1},
    "coinmetrics_btc_chain_pct":    {"day_convention": "utc",            "release_utc_hour_prior": 2},
    # 美东交易日（FRED）：日期标 ET 交易日，但发布约在 ET 16:00-16:30 ≈ UTC 20:30-22:00（夏令时）。
    # H.15 系列（DGS10/VIX）通常次日 H.15 release 才发，因此“对应日”可能滞后 1 个自然日。
    "macro_dxy":   {"day_convention": "et-trading",     "release_utc_hour_prior": 22},
    "macro_spx":   {"day_convention": "et-trading",     "release_utc_hour_prior": 22},
    "macro_us10y": {"day_convention": "et-trading",     "release_utc_hour_prior": 22},
    "macro_vix":   {"day_convention": "et-trading",     "release_utc_hour_prior": 22},
    "fred_t10y2y": {"day_convention": "et-trading",     "release_utc_hour_prior": 22},
    # 伦敦交易日（LBMA Gold PM auction）：fix 公布于 15:00 BST = UTC 14:00（夏） / 15:00 GMT（冬）。
    "macro_gold":  {"day_convention": "london-trading", "release_utc_hour_prior": 15},
}

ProbeResult = Tuple[Optional[str], int]
ProbeFn = Callable[[], ProbeResult]


def now_utc() -> pd.Timestamp:
    """当前 UTC 时间，保留时区。"""
    ts = pd.Timestamp.utcnow()
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def probe_window_start(ref: pd.Timestamp, lookback_days: int = PROBE_LOOKBACK_DAYS) -> pd.Timestamp:
    """探针窗口起点：当前 UTC 日 00:00 回溯 lookback_days 天。"""
    return ref.normalize() - pd.Timedelta(days=lookback_days)


def _last_date_iso(dates: List[pd.Timestamp]) -> Optional[str]:
    """从一组 UTC 时间戳中取最新日期，返回 ISO 字符串；空列表返回 None。"""
    if not dates:
        return None
    return max(dates).date().isoformat()


def probe_binance_funding(start: pd.Timestamp) -> ProbeResult:
    """Binance BTCUSDT 永续资金费率：startTime + 单页足够覆盖 14 天内 ~42 条 8h 记录。"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": "BTCUSDT", "limit": 1000, "startTime": int(start.timestamp() * 1000)}
    data = http_get(url=url, params=params).json()
    dates = [to_utc_date(item["fundingTime"], unit="ms") for item in data] if data else []
    return _last_date_iso(dates), len(data) if data else 0


def probe_binance_kline(endpoint: str, param_key: str, start: pd.Timestamp) -> ProbeResult:
    """Binance 日 K 线（指数价 / 溢价指数）：interval=1d，14 天 ~14 条。"""
    url = f"https://fapi.binance.com{endpoint}"
    params = {param_key: "BTCUSDT", "interval": "1d", "limit": 50, "startTime": int(start.timestamp() * 1000)}
    data = http_get(url=url, params=params).json()
    dates = [to_utc_date(row[0], unit="ms") for row in data] if data else []
    return _last_date_iso(dates), len(data) if data else 0


def probe_blockchain_chart(chart_id: str) -> ProbeResult:
    """Blockchain.com Charts：API 仅支持 timespan 粒度，30days 是最贴近 14 天的可用值。"""
    url = f"https://api.blockchain.info/charts/{chart_id}"
    params = {"timespan": "30days", "format": "json", "sampled": "false"}
    vals = http_get(url=url, params=params).json().get("values", [])
    dates = [to_utc_date(v["x"], unit="s") for v in vals]
    return _last_date_iso(dates), len(vals)


def probe_altme_fng() -> ProbeResult:
    """Alternative.me Fear & Greed：limit=30 取最近 30 天。"""
    data = http_get(url="https://api.alternative.me/fng/", params={"limit": 30, "format": "json"}).json().get("data", [])
    dates = [to_utc_date(int(item["timestamp"]), unit="s") for item in data]
    return _last_date_iso(dates), len(data)


def probe_wikipedia(start: pd.Timestamp, today: pd.Timestamp) -> ProbeResult:
    """Wikipedia Bitcoin pageviews：起止日期通过 URL path 精确传入。"""
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/all-agents/Bitcoin/daily/"
        f"{start.strftime('%Y%m%d')}/{today.strftime('%Y%m%d')}"
    )
    items = http_get(url=url).json().get("items", [])
    dates = [pd.to_datetime(item["timestamp"], format="%Y%m%d%H", utc=True).normalize() for item in items]
    return _last_date_iso(dates), len(items)


def probe_defillama_stable() -> ProbeResult:
    """DefiLlama 稳定币：API 无时间过滤参数，全量约 50 KB，本地取 max。"""
    data = http_get(url="https://stablecoins.llama.fi/stablecoincharts/all").json()
    dates: List[pd.Timestamp] = []
    for item in data:
        ts = item.get("date")
        if ts is None:
            continue
        dates.append(to_utc_date(ts, unit="s"))
    return _last_date_iso(dates), len(dates)


def probe_fred(series_id: str, start: pd.Timestamp) -> ProbeResult:
    """FRED 序列：observation_start 控制窗口；过滤掉 value='.' 这类未发布占位。"""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.date().isoformat(),
    }
    obs = http_get(url=url, params=params).json().get("observations", [])
    valid = [o for o in obs if o.get("value", ".") != "."]
    dates = [pd.to_datetime(o["date"], utc=True).normalize() for o in valid]
    return _last_date_iso(dates), len(valid)


def probe_lbma_gold() -> ProbeResult:
    """LBMA Gold PM：全量 JSON 约 30 KB，本地取 max。"""
    data = http_get(url="https://prices.lbma.org.uk/json/gold_pm.json").json()
    dates: List[pd.Timestamp] = []
    for item in data:
        if not isinstance(item, dict) or "d" not in item or "v" not in item:
            continue
        v = item["v"]
        if not isinstance(v, list) or not v or v[0] is None:
            continue
        dates.append(pd.to_datetime(item["d"], utc=True).normalize())
    return _last_date_iso(dates), len(dates)


def probe_coinmetrics(start: pd.Timestamp) -> ProbeResult:
    """Coin Metrics Community：start_time 控制窗口；探针只取 2 个轻量字段够判滞后。"""
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": "AdrActCnt,HashRate",
        "frequency": "1d",
        "page_size": 100,
        "start_time": start.date().isoformat(),
    }
    rows = http_get(url=url, params=params).json().get("data", [])
    dates = [pd.to_datetime(row["time"], utc=True).normalize() for row in rows]
    return _last_date_iso(dates), len(rows)


def build_probes(start: pd.Timestamp, today: pd.Timestamp) -> List[Tuple[str, ProbeFn]]:
    """与 data_layer_unified_opt.build_registry() 顺序对齐的探针列表。"""
    probes: List[Tuple[str, ProbeFn]] = []

    probes.append(("binance_funding_btcusdt", lambda: probe_binance_funding(start)))
    probes.append(("binance_index_price_1d", lambda: probe_binance_kline("/fapi/v1/indexPriceKlines", "pair", start)))
    probes.append(("binance_premium_index_1d", lambda: probe_binance_kline("/fapi/v1/premiumIndexKlines", "symbol", start)))

    for cid in ["hash-rate", "difficulty", "mempool-size", "n-transactions", "transaction-fees", "n-unique-addresses"]:
        dataset_id = f"blockchain_{cid.replace('-', '_')}"
        probes.append((dataset_id, lambda cid=cid: probe_blockchain_chart(cid)))

    probes.append(("altme_fng", probe_altme_fng))
    probes.append(("wikipedia_bitcoin_pageviews", lambda: probe_wikipedia(start, today)))
    probes.append(("defillama_stablecoins_total", probe_defillama_stable))

    fred_series = [
        ("DTWEXBGS", "macro_dxy"),
        ("SP500", "macro_spx"),
        ("DGS10", "macro_us10y"),
        ("VIXCLS", "macro_vix"),
        ("T10Y2Y", "fred_t10y2y"),
    ]
    for series_id, dataset_id in fred_series:
        probes.append((dataset_id, lambda sid=series_id: probe_fred(sid, start)))

    probes.append(("macro_gold", probe_lbma_gold))
    probes.append(("coinmetrics_btc_chain_pct", lambda: probe_coinmetrics(start)))

    return probes


def main() -> int:
    """对每个源跑一次探针，每条观测以单行 JSON 写到 stdout（无表头、无逗号）。"""
    load_env_from_dotenv()

    observed = now_utc()
    today = observed.normalize()
    today_iso = today.date().isoformat()
    start = probe_window_start(observed)

    probes = build_probes(start=start, today=today)

    for dataset_id, probe_fn in probes:
        meta = DATASET_METADATA.get(dataset_id)
        if meta is None:
            # registry 与 metadata 必须保持一一对应；缺映射时 fail fast 而不是悄悄写出半字段。
            raise KeyError(f"DATASET_METADATA missing entry for {dataset_id}")
        record: Dict[str, Any] = {
            "observed_at_utc": observed.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "current_utc_date": today_iso,
            "dataset_id": dataset_id,
            "last_date": None,
            "rows": 0,
            "lag_days": None,
            "includes_current_day": False,
            "day_convention": meta["day_convention"],
            "release_utc_hour_prior": meta["release_utc_hour_prior"],
            "ok": False,
            "error": None,
        }
        try:
            last_date_str, rows = probe_fn()
            record["last_date"] = last_date_str
            record["rows"] = int(rows)
            if last_date_str is not None:
                last_d = pd.to_datetime(last_date_str).date()
                record["lag_days"] = (today.date() - last_d).days
                record["includes_current_day"] = last_d == today.date()
            record["ok"] = True
        except Exception as ex:
            # 单源失败不影响其余源；错误类型 + 消息进 jsonl 供后续审计。
            record["error"] = f"{type(ex).__name__}: {ex}"

        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
