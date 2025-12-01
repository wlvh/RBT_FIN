
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_data_layer.py
-------------------
目的：
- 仅采集 **日或以上级别** 且 **免费**、**不受“仅近30天”限制** 的数据，统一对齐到“日”时间轴；
- 模块化数据层结构（collector/adapter/registry），统一返回包含 ['date', <value_cols...>] 的 DataFrame；
- 程序运行后输出一个 CSV，字段：
  [dataset_id, category, endpoint, earliest_date, sample_start, sample_end, sample_rows, na_cells_300, na_ratio_300, remark]

约定：
- 只使用免费 API；需要免费 API key 的（如 FRED）从环境变量读取（FRED_KEY），无则跳过并标注 TODO；
- 避免 30 天窗口类端点（如 Binance taker 多空比、GLS 等）；
- 子日频数据（如资金费率 8h）会聚合为日频（默认 'mean'）；
- 为了安全性/健壮性，网络失败或字段变动时会优雅降级并在 CSV 的 remark 中写明错误/提示；

TODO（保留未来可扩展的免费接口占位）：
- Etherscan: ERC20 稳定币供给的**逐日**历史（需要 getblocknobytime → tokensupplyhistory 映射；属于免费接口，但请求量较大，后续加入批量限流与缓存）。
- TRONSCAN: USDT(TRON) 的历史供给与活跃度（需确认历史端点稳定性）。
- Google Trends: 官方无公共 API；可考虑合法合规方式接入或仅保留说明。
- Github 开发活跃度：/stats/contributors 提供**周**频历史，后续可加入周频→日频的对齐与聚合。

使用方法：
    python daily_data_layer.py --out datasets_daily_summary.csv --limit 300

依赖：requests, pandas
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

import requests
import pandas as pd

# --------------------------- 基础工具 ---------------------------

DEFAULT_TIMEOUT = (5, 30)
USER_AGENT = "btc-daily-data-layer/0.2 (+research, free-only)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain;q=0.8"})

def http_get(url: str, params: Optional[Dict[str, Any]]=None, timeout=DEFAULT_TIMEOUT, max_retries: int=3, backoff: float=1.6) -> requests.Response:
    last_exc = None
    for i in range(max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            # 部分公共 API 对 429/5xx 比较敏感
            if r.status_code in (429, 418, 500, 502, 503, 504, 520, 525, 526, 527):
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")

def to_utc_date(ts: Any, unit: str="ms") -> pd.Timestamp:
    """将时间戳/字符串转换为 UTC 日期（按天对齐）。"""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return pd.NaT
    if isinstance(ts, (int, float)):
        if unit == "ms":
            t = pd.to_datetime(int(ts), unit="ms", utc=True)
        elif unit == "s":
            t = pd.to_datetime(int(ts), unit="s", utc=True)
        else:
            t = pd.to_datetime(ts, utc=True)
    else:
        t = pd.to_datetime(ts, utc=True)
    # 统一到日期（UTC）
    return pd.Timestamp(t.date(), tz="UTC")

def ensure_daily(df: pd.DataFrame, date_col: str="date", agg: str="mean") -> pd.DataFrame:
    """确保 DataFrame 为日频；若更高频，则按日聚合（默认 mean）。"""
    if df.empty:
        return df
    if date_col not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": date_col})
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols + [date_col]]
    if non_numeric_cols:
        # 非数值列丢弃（可按需扩展：one-hot/label）
        df = df.drop(columns=non_numeric_cols, errors="ignore")
    if not numeric_cols:
        # 无数值列，返回仅含日期（当作占位，后续缺失会很高）
        return df[[date_col]].drop_duplicates().sort_values(date_col)
    # 如果已经唯一日粒度直接返回，否则聚合
    if df.groupby(date_col).size().max() == 1:
        return df[[date_col] + numeric_cols].sort_values(date_col)
    if agg == "mean":
        out = df.groupby(date_col, as_index=False)[numeric_cols].mean()
    elif agg == "sum":
        out = df.groupby(date_col, as_index=False)[numeric_cols].sum()
    elif agg == "last":
        out = df.sort_values(date_col).groupby(date_col, as_index=False)[numeric_cols].last()
    else:
        out = df.groupby(date_col, as_index=False)[numeric_cols].mean()
    return out.sort_values(date_col)

def sample_first_n_days(df: pd.DataFrame, n: int=300, date_col: str="date") -> pd.DataFrame:
    """从最早日期起取前 n 天（缺天则以自然日对齐并不填充）。"""
    if df.empty:
        return df
    start = pd.to_datetime(df[date_col].min(), utc=True)
    end = start + pd.Timedelta(days=n-1)
    cal = pd.DataFrame({date_col: pd.date_range(start, end, freq="D", tz="UTC")})
    return cal.merge(df, on=date_col, how="left")

def na_cells_info(df: pd.DataFrame, date_col: str="date") -> Tuple[int, float]:
    """计算缺失单元格数与占比（相对于数值列 * 行数）。"""
    if df.empty:
        return 0, 0.0
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        # 没有值列，视为全缺失
        return len(df) * 1, 1.0
    na_cells = df[value_cols].isna().sum().sum()
    denom = len(df) * len(value_cols)
    ratio = float(na_cells) / float(denom) if denom else 0.0
    return int(na_cells), ratio

# --------------------------- 数据规范 ---------------------------

@dataclass
class Dataset:
    dataset_id: str
    category: str
    endpoint: str
    params: Dict[str, Any]
    collector: Callable[..., pd.DataFrame]
    agg: str = "mean"  # 子日频如何聚合到日频
    free: bool = True
    only_30d: bool = False
    remark: str = ""   # 额外说明（会写入 CSV）

# --------------------------- Collectors ---------------------------
# 仅选择“日或以上级别 & 非30天窗口 & 免费”的端点

# A. Binance（USDⓈ-M）- 指数/标记价K线（1d），资金费率（聚合为日频）
def binance_index_klines_daily(pair: str="BTCUSDT", interval: str="1d", limit: int=1500, start_ms: Optional[int]=None, end_ms: Optional[int]=None) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/indexPriceKlines"
    params = {"pair": pair, "interval": interval, "limit": limit}
    if start_ms: params["startTime"] = int(start_ms)
    if end_ms: params["endTime"] = int(end_ms)
    r = http_get(url, params=params)
    data = r.json()
    # kline: [open_time, open, high, low, close, ...]
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"]
    df = pd.DataFrame(data, columns=cols[:len(data[0])] if data else cols)
    if df.empty:
        return df
    df = df[["open_time","open","high","low","close"]]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","close"]].rename(columns={"close":"index_close"})

def binance_premium_index_klines_daily(symbol: str="BTCUSDT", interval: str="1d", limit: int=1500, start_ms: Optional[int]=None, end_ms: Optional[int]=None) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms: params["startTime"] = int(start_ms)
    if end_ms: params["endTime"] = int(end_ms)
    r = http_get(url, params=params)
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"]
    df = pd.DataFrame(data, columns=cols[:len(data[0])] if data else cols)
    if df.empty:
        return df
    df = df[["open_time","close"]]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date","close"]].rename(columns={"close":"premium_index_close"})

def binance_funding_rate_daily(symbol: str="BTCUSDT", limit: int=1000, start_ms: Optional[int]=None, end_ms: Optional[int]=None, agg: str="mean") -> pd.DataFrame:
    # 8h 资金费率 → 日频聚合
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if start_ms: params["startTime"] = int(start_ms)
    if end_ms: params["endTime"] = int(end_ms)
    out = []
    cursor_start = start_ms
    while True:
        r = http_get(url, params=params)
        lst = r.json()
        if not lst:
            break
        tmp = pd.DataFrame(lst)
        tmp["date"] = tmp["fundingTime"].apply(lambda x: to_utc_date(x, unit="ms"))
        tmp["fundingRate"] = pd.to_numeric(tmp["fundingRate"], errors="coerce")
        out.append(tmp[["date","fundingRate"]])
        if len(lst) < limit or (end_ms and tmp["date"].max().value >= pd.to_datetime(end_ms, unit="ms", utc=True).value):
            break
        # 滚动窗口：以上次最后一条的时间 + 1 ms 继续
        last_ts = int(lst[-1]["fundingTime"])
        params["startTime"] = last_ts + 1
    if not out:
        return pd.DataFrame(columns=["date","fundingRate"])
    df = pd.concat(out, ignore_index=True)
    return ensure_daily(df, date_col="date", agg=agg)

# B. OKX / Bybit / Deribit
def okx_funding_rate_daily(instId: str="BTC-USDT-SWAP", limit: int=100, max_pages: int=500, agg: str="mean") -> pd.DataFrame:
    url = "https://www.okx.com/api/v5/public/funding-rate-history"
    params = {"instId": instId, "limit": limit}
    out = []
    before = None
    pages = 0
    while pages < max_pages:
        if before:
            params["before"] = before
        r = http_get(url, params=params)
        data = r.json().get("data", [])
        if not data:
            break
        df = pd.DataFrame(data)
        if "fundingTime" not in df.columns:
            break
        df["date"] = df["fundingTime"].apply(lambda x: to_utc_date(x, unit="ms"))
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        out.append(df[["date","fundingRate"]])
        # 翻页：OKX 使用 before/after；data 通常按时间倒序返回
        min_ts = min([int(x["fundingTime"]) for x in data])
        before = str(min_ts)
        pages += 1
        # 提前停止：如果已经够 300 天就可以退出
        if out and len(pd.concat(out).groupby("date").size()) >= 300:
            break
    if not out:
        return pd.DataFrame(columns=["date","fundingRate"])
    df_all = pd.concat(out, ignore_index=True)
    return ensure_daily(df_all, date_col="date", agg=agg)

def bybit_funding_rate_daily(category: str="linear", symbol: str="BTCUSDT", limit: int=200, max_pages: int=200, agg: str="mean") -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/history-fund-rate"
    params = {"category": category, "symbol": symbol, "limit": limit}
    out = []
    cursor = None
    pages = 0
    while pages < max_pages:
        if cursor:
            params["cursor"] = cursor
        r = http_get(url, params=params)
        j = r.json().get("result") or r.json()
        data = j.get("list", [])
        if not data:
            break
        df = pd.DataFrame(data)
        if "fundingRateTimestamp" not in df.columns:
            break
        df["date"] = df["fundingRateTimestamp"].apply(lambda x: to_utc_date(x, unit="ms"))
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        out.append(df[["date","fundingRate"]])
        cursor = j.get("nextPageCursor")
        pages += 1
        if cursor in (None, "", "null"):
            # 到头
            break
        if out and len(pd.concat(out).groupby("date").size()) >= 300:
            break
    if not out:
        return pd.DataFrame(columns=["date","fundingRate"])
    df_all = pd.concat(out, ignore_index=True)
    return ensure_daily(df_all, date_col="date", agg=agg)

def deribit_historical_volatility_daily(currency: str="BTC") -> pd.DataFrame:
    url = "https://www.deribit.com/api/v2/public/get_historical_volatility"
    r = http_get(url, params={"currency": currency})
    # result: [[ts_ms, hv], ...]
    res = r.json().get("result", [])
    df = pd.DataFrame(res, columns=["ts","hv"])
    if df.empty:
        return df
    df["date"] = df["ts"].apply(lambda x: to_utc_date(x, unit="ms"))
    df["hv"] = pd.to_numeric(df["hv"], errors="coerce")
    return df[["date","hv"]]

# C. 链上（Blockchain.com Charts）
def blockchain_chart_daily(name: str="hash-rate", timespan: str="all") -> pd.DataFrame:
    url = f"https://api.blockchain.info/charts/{name}"
    r = http_get(url, params={"timespan": timespan, "format": "json"})
    j = r.json()
    vals = j.get("values", [])
    df = pd.DataFrame(vals)
    if df.empty:
        return df
    df["date"] = df["x"].apply(lambda x: to_utc_date(x, unit="s"))
    df["value"] = pd.to_numeric(df["y"], errors="coerce")
    return df[["date","value"]]

# D. 情绪（Alternative.me Fear & Greed）
def altme_fng_daily(limit: int=0) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {}
    if limit:
        params["limit"] = int(limit)
    else:
        params["limit"] = 0  # 0 = all
    r = http_get(url, params=params)
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = df["timestamp"].apply(lambda x: to_utc_date(int(x), unit="s"))
    df["index"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date","index"]]

# E. 关注度（Wikipedia Pageviews）
def wikipedia_pageviews_daily(article: str="Bitcoin", project: str="en.wikipedia.org", access: str="all-access", agents: str="all-agents", start: str="20150701", end: Optional[str]=None) -> pd.DataFrame:
    if end is None:
        end = dt.datetime.utcnow().strftime("%Y%m%d")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agents}/{article}/daily/{start}/{end}"
    r = http_get(url)
    items = r.json().get("items", [])
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", utc=True).dt.normalize()
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    return df[["date","views"]]

# F. 宏观（FRED）
def fred_series_daily(series_id: str, api_key: Optional[str]=None, observation_start: Optional[str]=None) -> pd.DataFrame:
    api_key = api_key or os.getenv("FRED_KEY")
    if not api_key:
        # 无 key 情况下，返回空并在 remark 标注 TODO
        return pd.DataFrame()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if observation_start:
        params["observation_start"] = observation_start
    r = http_get(url, params=params)
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # FRED 中有很多 '.'（缺失）值，已转为 NaN
    return df[["date","value"]]

# G. 稳定币代理（CoinGecko：USDT 市值/供给代理）
def coingecko_market_cap_daily(coin_id: str="tether", vs_currency: str="usd", days: str="max") -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = http_get(url, params={"vs_currency": vs_currency, "days": days})
    j = r.json()
    caps = j.get("market_caps", [])
    df = pd.DataFrame(caps, columns=["ts","market_cap"])
    if df.empty:
        return df
    df["date"] = df["ts"].apply(lambda x: to_utc_date(x, unit="ms"))
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    return df[["date","market_cap"]]

# --------------------------- 注册表 ---------------------------

def build_registry() -> List[Dataset]:
    reg: List[Dataset] = []

    # 衍生品（Binance/OKX/Bybit/Deribit）
    reg.append(Dataset(
        dataset_id="binance_funding_btcusdt",
        category="衍生品/资金费率",
        endpoint="https://fapi.binance.com/fapi/v1/fundingRate",
        params={"symbol":"BTCUSDT","limit":1000},
        collector=lambda **kw: binance_funding_rate_daily(symbol="BTCUSDT", limit=1000, **kw),
        agg="mean",
        remark="聚合: 8h→日均; 参数={'symbol':'BTCUSDT','limit':1000}"
    ))
    reg.append(Dataset(
        dataset_id="binance_index_price_1d",
        category="衍生品/指数价K线",
        endpoint="https://fapi.binance.com/fapi/v1/indexPriceKlines",
        params={"pair":"BTCUSDT","interval":"1d","limit":1500},
        collector=lambda **kw: binance_index_klines_daily(pair="BTCUSDT", interval="1d", limit=1500, **kw),
        agg="last",
        remark="字段使用 close 作为 index_close; 参数={'pair':'BTCUSDT','interval':'1d','limit':1500}"
    ))
    reg.append(Dataset(
        dataset_id="binance_premium_index_1d",
        category="衍生品/溢价指数K线",
        endpoint="https://fapi.binance.com/fapi/v1/premiumIndexKlines",
        params={"symbol":"BTCUSDT","interval":"1d","limit":1500},
        collector=lambda **kw: binance_premium_index_klines_daily(symbol="BTCUSDT", interval="1d", limit=1500, **kw),
        agg="last",
        remark="字段使用 close 作为 premium_index_close; 参数={'symbol':'BTCUSDT','interval':'1d','limit':1500}"
    ))
    reg.append(Dataset(
        dataset_id="okx_funding_btcusdt",
        category="跨交易所/资金费率",
        endpoint="https://www.okx.com/api/v5/public/funding-rate-history",
        params={"instId":"BTC-USDT-SWAP","limit":100},
        collector=lambda **kw: okx_funding_rate_daily(instId="BTC-USDT-SWAP", limit=100, **kw),
        agg="mean",
        remark="聚合: 8h→日均; 参数={'instId':'BTC-USDT-SWAP','limit':100}"
    ))
    reg.append(Dataset(
        dataset_id="bybit_funding_btcusdt",
        category="跨交易所/资金费率",
        endpoint="https://api.bybit.com/v5/market/history-fund-rate",
        params={"category":"linear","symbol":"BTCUSDT","limit":200},
        collector=lambda **kw: bybit_funding_rate_daily(category="linear", symbol="BTCUSDT", limit=200, **kw),
        agg="mean",
        remark="聚合: 8h→日均; 参数={'category':'linear','symbol':'BTCUSDT','limit':200}"
    ))
    reg.append(Dataset(
        dataset_id="deribit_hv_btc",
        category="期权/历史波动率",
        endpoint="https://www.deribit.com/api/v2/public/get_historical_volatility",
        params={"currency":"BTC"},
        collector=lambda **kw: deribit_historical_volatility_daily(currency="BTC"),
        agg="last",
        remark="Deribit 官方 HV（日频）; 参数={'currency':'BTC'}"
    ))

    # 链上（全为日频）
    for chart_id, cname in [
        ("hash-rate", "链上/算力"),
        ("difficulty", "链上/难度"),
        ("n-transactions", "链上/链上交易数"),
        ("transaction-fees", "链上/总手续费(USD)")
    ]:
        reg.append(Dataset(
            dataset_id=f"blockchain_{chart_id.replace('-', '_')}",
            category=cname,
            endpoint=f"https://api.blockchain.info/charts/{chart_id}",
            params={"timespan":"all","format":"json"},
            collector=lambda chart_id=chart_id, **kw: blockchain_chart_daily(name=chart_id, timespan="all"),
            agg="last",
            remark=f"Blockchain.com Charts; 参数={{'timespan':'all','format':'json'}}"
        ))

    # 情绪与关注度
    reg.append(Dataset(
        dataset_id="altme_fng",
        category="情绪/恐惧与贪婪",
        endpoint="https://api.alternative.me/fng/",
        params={"limit":0},
        collector=lambda **kw: altme_fng_daily(limit=0),
        agg="last",
        remark="limit=0 返回全历史"
    ))
    reg.append(Dataset(
        dataset_id="wikipedia_bitcoin_pageviews",
        category="关注度/Wikipedia",
        endpoint="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agents}/{article}/daily/{start}/{end}",
        params={"project":"en.wikipedia.org","access":"all-access","agents":"all-agents","article":"Bitcoin","start":"20150701","end":"<today>"},
        collector=lambda **kw: wikipedia_pageviews_daily(article="Bitcoin", project="en.wikipedia.org", access="all-access", agents="all-agents", start="20150701"),
        agg="sum",
        remark="自 2015-07-01 起提供逐日 PV; 参数见上"
    ))

    # 宏观（FRED，需免费 API KEY）
    for sid, cname in [
        ("DGS10", "宏观/美国10年国债收益率"),
        ("T10Y2Y", "宏观/期限利差(10Y-2Y)"),
        ("VIXCLS", "宏观/VIX"),
        ("DTWEXBGS", "宏观/广义美元指数")
    ]:
        reg.append(Dataset(
            dataset_id=f"fred_{sid.lower()}",
            category=cname,
            endpoint="https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "file_type": "json", "api_key": "${FRED_KEY}"},
            collector=lambda sid=sid, **kw: fred_series_daily(series_id=sid),
            agg="last",
            remark="需要免费 FRED_KEY 环境变量；若缺失将标记 TODO"
        ))

    # 稳定币（CoinGecko：USDT 市值/供给代理，免费）
    reg.append(Dataset(
        dataset_id="coingecko_usdt_market_cap",
        category="稳定币/USDT 市值代理",
        endpoint="https://api.coingecko.com/api/v3/coins/tether/market_chart",
        params={"vs_currency":"usd","days":"max"},
        collector=lambda **kw: coingecko_market_cap_daily(coin_id="tether", vs_currency="usd", days="max"),
        agg="last",
        remark="使用 market_caps 作为 USDT 供给的代理；免费但限流"
    ))

    # ---- TODO 占位（免费 API，后续完善，不参与本次抓取） ----
    # - Etherscan tokensupplyhistory: 逐日供应量（需 getblocknobytime + tokensupplyhistory；免费，但实现较重）。
    # - TRONSCAN 稳定币供给历史：待确认稳定可用的历史接口。
    return reg

# --------------------------- 主流程 ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets_daily_summarygpt50pro.csv", help="输出 CSV 路径")
    ap.add_argument("--limit", type=int, default=300, help="从最早日期起抽取的日样本长度（默认 300 天）")
    args = ap.parse_args()

    reg = build_registry()
    rows = []

    for ds in reg:
        earliest_date = None
        sample_df = pd.DataFrame()
        err = None
        # 拉取 & 统一为日频并抽取前 N 天
        try:
            raw = ds.collector()
            daily = ensure_daily(raw, date_col="date", agg=ds.agg)
            if daily.empty:
                earliest_date = None
                sample_df = pd.DataFrame()
            else:
                earliest_date = pd.to_datetime(daily["date"].min(), utc=True).date().isoformat()
                sample_df = sample_first_n_days(daily, n=args.limit, date_col="date")
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
            daily = pd.DataFrame()
            sample_df = pd.DataFrame()

        na_cells, na_ratio = na_cells_info(sample_df, date_col="date")
        sample_rows = int(len(sample_df)) if not sample_df.empty else 0
        sample_start = sample_df["date"].min().date().isoformat() if sample_rows else None
        sample_end = sample_df["date"].max().date().isoformat() if sample_rows else None

        remark_parts = [ds.remark or ""]
        if err:
            if "fred_" in ds.dataset_id and "API key" in (err or ""):
                remark_parts.append("TODO: 设置 FRED_KEY 环境变量后再运行。")
            remark_parts.append(f"error={err}")
        # 将 API 参数写入备注
        remark_parts.append(f"params={json.dumps(ds.params, ensure_ascii=False)}")
        remark = " | ".join([p for p in remark_parts if p])

        rows.append({
            "dataset_id": ds.dataset_id,
            "category": ds.category,
            "endpoint": ds.endpoint,
            "earliest_date": earliest_date,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "sample_rows": sample_rows,
            "na_cells_300": na_cells,
            "na_ratio_300": round(na_ratio, 6),
            "remark": remark
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["category","dataset_id"]).reset_index(drop=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows to {args.out}")
    # 额外输出：建议把 CSV 打开检查是否存在长期空值或异常高缺失的源

if __name__ == "__main__":
    main()
