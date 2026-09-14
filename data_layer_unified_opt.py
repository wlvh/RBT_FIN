#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified daily data layer
------------------------
- 结合现有脚本的优点：模块化 registry、重试/超时、统一日频对齐、缺失度量、错误降级。
- 覆盖端点：Binance/OKX/Bybit/Deribit、Blockchain.com（多指标含 mempool）、CoinGecko、Fear&Greed、
  Wikipedia、DefiLlama 稳定币、FRED（需 key）、Glassnode（需 key）、yfinance 宏观/风险资产。
- 输出 CSV 字段：
  [dataset_id, category, endpoint, earliest_date, sample_start, sample_end,
   sample_rows, na_cells, na_ratio, remark]

刷新语义（重要）：
    **本脚本是会触发刷新的兼容入口。** 每次运行都会对注册表中的每一项真实发起采集，
    因此它既不是只读概览，也不提供缓存/覆盖保证。
    只读概览请使用新入口：`rbt-fin overview --cache-root <dir>`，它不会更新任何来源。

    本文件保留 `--out/--limit` 的原有用途，作为迁移期的兼容入口；
    其各来源将在 WP2 中逐个迁移到 `rbt_fin` 的公共流程。

用法示例：
    python data_layer_unified_opt.py --out datasets_unified_summary.csv --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from io import StringIO
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

# -------------------- 基础配置 --------------------

DEFAULT_TIMEOUT = (5, 30)
USER_AGENT = "btc-daily-unified/0.3 (+research, free-only)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain;q=0.8"})

RETRY_STATUS = {429, 418, 500, 502, 503, 504, 520, 525, 526, 527}

ENV_LOADED = False


def load_env_from_dotenv(path: str = ".env", reload: bool = False) -> None:
    """
    将 .env 中的 key=value 写入进程环境，避免在代码里硬编码密钥。
    参数：
        path: .env 所在路径，默认当前目录。
        reload: 是否强制重新加载（用于测试或显式刷新）。
    返回：
        None，执行后 os.environ 更新。
    """
    global ENV_LOADED
    if ENV_LOADED and not reload:
        return
    if not os.path.exists(path):
        ENV_LOADED = True
        return

    # 逐行解析 key=value，跳过空行与注释。
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key:
                # 仅在未设置时注入，避免覆盖外部环境。
                os.environ.setdefault(key, val)
    ENV_LOADED = True


# 为什么不在模块导入时加载 .env：
# 导入一个模块不应该修改进程环境。.env 的加载改到 `main()`（CLI 入口）里执行，
# 让“谁在什么时候注入了凭据”始终是显式的。


def http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout=DEFAULT_TIMEOUT,
    max_retries: int = 3,
    backoff: float = 1.6,
    headers: Optional[Dict[str, Any]] = None
) -> requests.Response:
    """
    GET with simple retry/backoff for rate limits/5xx。
    支持透传自定义 headers（如 CoinGecko Demo key）。
    """
    last_exc = None
    for i in range(max_retries + 1):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                time.sleep(backoff * (i + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")


def to_utc_date(ts: Any, unit: str = "ms") -> pd.Timestamp:
    """将时间戳/字符串转为 UTC 日期（只保留日期部分）。"""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return pd.NaT
    if isinstance(ts, str):
        ts_strip = ts.strip()
        # 字符串数字直接按给定 unit 转换
        if ts_strip.replace(".", "", 1).isdigit():
            try:
                ts = float(ts_strip)
            except ValueError:
                ts = ts_strip
    if isinstance(ts, (int, float)):
        if unit == "ms":
            dt = pd.to_datetime(int(ts), unit="ms", utc=True)
        elif unit == "s":
            dt = pd.to_datetime(int(ts), unit="s", utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return pd.Timestamp(dt.date(), tz="UTC")


def ensure_daily(df: pd.DataFrame, date_col: str = "date", agg: str = "mean") -> pd.DataFrame:
    """保证日频；子日频按 agg 聚合。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=[date_col])
    if date_col not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": date_col})
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col])
    numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df[[date_col]].drop_duplicates().sort_values(date_col)
    if df.groupby(date_col).size().max() == 1:
        return df[[date_col] + numeric_cols].sort_values(date_col)
    group = df.groupby(date_col, as_index=False)[numeric_cols]
    if agg == "sum":
        out = group.sum()
    elif agg == "last":
        out = df.sort_values(date_col).groupby(date_col, as_index=False)[numeric_cols].last()
    else:
        out = group.mean()
    return out.sort_values(date_col)


def sample_first_n_days(df: pd.DataFrame, n: int = 300, date_col: str = "date") -> pd.DataFrame:
    """从最早日期起抽取前 n 天，保持自然日历以暴露缺失。"""
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=[date_col])
    start = pd.to_datetime(df[date_col].min(), utc=True)
    end = start + pd.Timedelta(days=n - 1)
    cal = pd.DataFrame({date_col: pd.date_range(start, end, freq="D", tz="UTC")})
    return cal.merge(df, on=date_col, how="left")


def na_cells_info(df: pd.DataFrame, date_col: str = "date") -> tuple[int, float]:
    if df is None or df.empty:
        return 0, 0.0
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        return len(df), 1.0
    na_cells = df[value_cols].isna().sum().sum()
    denom = len(df) * len(value_cols)
    return int(na_cells), float(na_cells) / float(denom) if denom else 0.0


# -------------------- 数据集描述 --------------------

@dataclass
class Dataset:
    dataset_id: str
    category: str
    endpoint: str
    params: Dict[str, Any]
    collector: Callable[..., pd.DataFrame]
    agg: str = "mean"
    remark: str = ""


# -------------------- Collectors: 衍生品 --------------------

def binance_kline_daily(endpoint: str, param_key: str, symbol: str, value_col: str, limit: int = 1500) -> pd.DataFrame:
    url = f"https://fapi.binance.com{endpoint}"
    params = {param_key: symbol, "interval": "1d", "limit": limit, "startTime": 0}
    r = http_get(url, params=params)
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["date", value_col])
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"]
    df = pd.DataFrame(data, columns=cols[: len(data[0])])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df[value_col] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", value_col]]


def binance_funding_rate_daily(
    symbol: str = "BTCUSDT",
    limit: int = 1000,
    start_ms: Optional[int] = 1,
    end_ms: Optional[int] = None,
    agg: str = "mean"
) -> pd.DataFrame:
    """
    拉取 Binance 永续资金费率，并自最早时间起分页到 end_ms（如提供）为止，最终聚合为日频。
    注意：接口若返回 fundingTime 降序，自动改用 endTime 向历史回溯，避免仅取最近 limit 条。
    参数：
        symbol: 交易对，示例 "BTCUSDT"。
        limit: 单页最大条数（Binance 接口上限 1000）。
        start_ms: 起始毫秒时间戳，默认为 1，避免 startTime=0 被 Binance 视为未传参。
        end_ms: 结束毫秒时间戳，可选；不传则一直拉到最新。
        agg: 日内聚合方式，默认均值。
    返回：
        包含 date、fundingRate 列的日频 DataFrame，date 为 UTC 日期。
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params: Dict[str, Any] = {"symbol": symbol, "limit": limit}

    # Binance 将 startTime=0 视作缺省，因此将非正值提升为 1，确保从最早时间开始。
    start_time = 1 if start_ms is None or int(start_ms) <= 0 else int(start_ms)
    params["startTime"] = start_time
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    frames: List[pd.DataFrame] = []
    seen_earliest: Optional[int] = None
    order: Optional[str] = None  # 接口实际返回的时间排序，None/asc/desc

    while True:
        # 分页拉取；若接口未按升序返回，会基于首批数据推断排序方向。
        resp = http_get(url, params=params)
        batch = resp.json()
        if not batch:
            break

        # 转日期、数值后累计。
        frame = pd.DataFrame(batch)
        frame["date"] = frame["fundingTime"].apply(lambda ts: to_utc_date(ts, unit="ms"))
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frames.append(frame[["date", "fundingRate"]])

        timestamps = [int(item["fundingTime"]) for item in batch]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        # 推断接口返回顺序，若为降序则改用 endTime 向历史回溯，避免只取最近 N 条。
        if order is None:
            asc_sorted = timestamps == sorted(timestamps)
            desc_sorted = timestamps == sorted(timestamps, reverse=True)
            order = "desc" if desc_sorted else "asc"

        # 如果最早时间未推进，避免死循环（接口可能忽略 startTime 并重复返回最近 N 条）。
        if seen_earliest is not None:
            if order == "desc" and min_ts >= seen_earliest:
                break
            if order != "desc" and min_ts <= seen_earliest:
                break
        seen_earliest = min_ts

        # 末页或达到 end_ms 即停止。
        if len(batch) < limit:
            break
        if end_ms is not None:
            if order == "desc" and min_ts <= int(end_ms):
                break
            if order != "desc" and max_ts >= int(end_ms):
                break

        if order == "desc":
            # 端点若是降序，则用 endTime 向更早时间翻页。
            next_end = min_ts - 1
            if next_end <= start_time:
                break
            params["endTime"] = next_end
        else:
            # 正常升序：起点推进到上一页最后一条之后。
            params.pop("endTime", None)
            params["startTime"] = max_ts + 1

    if not frames:
        return pd.DataFrame(columns=["date", "fundingRate"])

    df = pd.concat(frames, ignore_index=True).sort_values("date")
    return ensure_daily(df, date_col="date", agg=agg)


def bybit_funding_rate_daily(
    symbol: str = "BTCUSDT",
    category: str = "linear",
    limit: int = 200,
    start_ms: Optional[int] = 0,
    agg: str = "mean"
) -> pd.DataFrame:
    """
    拉取 Bybit V5 永续资金费率历史，按日聚合。
    参数：
        symbol: 交易对，示例 "BTCUSDT"。
        category: 市场类别，线性/反向等，默认 "linear"。
        limit: 单页条数，接口上限 200。
        start_ms: 起始毫秒时间戳，默认 0 自最早时间拉取。
        agg: 日内聚合方式，默认均值。
    返回：
        含 date、fundingRate 的日频 DataFrame。
    """
    url = "https://api.bybit.com/v5/market/funding/history"
    params: Dict[str, Any] = {"category": category, "symbol": symbol, "limit": limit}

    # Bybit startTime=0 可以表示自最早起取；小于 0 无意义，转为 0。
    if start_ms is not None:
        start_time = 0 if int(start_ms) < 0 else int(start_ms)
        params["startTime"] = start_time

    frames: List[pd.DataFrame] = []
    seen_earliest: Optional[int] = None
    end_time: Optional[int] = None

    for _ in range(400):
        # Bybit 文档说明可用 endTime 向前翻页；每次用上一页最早时间减一避免重叠。
        if end_time is not None:
            params["endTime"] = end_time

        resp = http_get(url, params=params)
        result = resp.json().get("result", {}) or {}
        batch = result.get("list") or []
        if not batch:
            break

        frame = pd.DataFrame(batch)
        if "fundingRateTimestamp" not in frame.columns or "fundingRate" not in frame.columns:
            break

        # 转 UTC 日期、校验数值并累计。
        frame["date"] = frame["fundingRateTimestamp"].apply(lambda ts: to_utc_date(int(ts), unit="ms"))
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frames.append(frame[["date", "fundingRate"]])

        # 确定本页最早时间并更新翻页游标。
        page_earliest = min(int(item["fundingRateTimestamp"]) for item in batch)
        if seen_earliest is not None and page_earliest >= seen_earliest:
            # 未能继续向前推进，避免死循环。
            break
        seen_earliest = page_earliest
        end_time = page_earliest - 1

    if not frames:
        return pd.DataFrame(columns=["date", "fundingRate"])

    daily = pd.concat(frames, ignore_index=True).sort_values("date")
    return ensure_daily(daily, date_col="date", agg=agg)


# -------------------- Collectors: 链上 --------------------

def blockchain_chart_daily(chart_id: str) -> pd.DataFrame:
    url = f"https://api.blockchain.info/charts/{chart_id}"
    r = http_get(url, params={"timespan": "all", "format": "json"})
    vals = r.json().get("values", [])
    df = pd.DataFrame(vals)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = df["x"].apply(lambda x: to_utc_date(x, unit="s"))
    df["value"] = pd.to_numeric(df["y"], errors="coerce")
    return df[["date", "value"]]


# -------------------- Collectors: 情绪/关注度/流动性 --------------------

def altme_fng_daily(limit: int = 0) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {"limit": limit, "format": "json"}
    r = http_get(url, params=params)
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["date", "index"])
    df["date"] = df["timestamp"].apply(lambda x: to_utc_date(int(x), unit="s"))
    df["index"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "index"]]


def wikipedia_pageviews_daily(article: str = "Bitcoin", project: str = "en.wikipedia.org") -> pd.DataFrame:
    """
    获取指定维基百科条目的日度页面浏览量（PV），覆盖 2015-07-01 至今。
    参数：
        article: 维基百科条目名，默认 "Bitcoin"。
        project: 站点，默认 "en.wikipedia.org"。
    返回：
        含 date、views 的日频 DataFrame，date 为 UTC 日期，views 为整数 PV。
    """
    end = pd.Timestamp.utcnow().strftime("%Y%m%d")
    # 官方 REST 路径：/metrics/pageviews/per-article/{project}/all-access/all-agents/{article}/daily/{start}/{end}
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{article}/daily/20150701/{end}"
    r = http_get(url)
    items = r.json().get("items", [])
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=["date", "views"])
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", utc=True).dt.normalize()
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    return df[["date", "views"]]


def defillama_stablecoins_daily() -> pd.DataFrame:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    r = http_get(url)
    data = r.json()
    records = []
    for item in data:
        ts = item.get("date")
        raw_val = item.get("totalCirculatingUSD") or item.get("totalCirculating")
        val = None
        if isinstance(raw_val, (int, float)):
            val = float(raw_val)
        elif isinstance(raw_val, dict):
            if "peggedUSD" in raw_val and isinstance(raw_val["peggedUSD"], (int, float)):
                val = float(raw_val["peggedUSD"])
            else:
                nums = [float(v) for v in raw_val.values() if isinstance(v, (int, float))]
                val = sum(nums) if nums else None
        if val is None:
            continue
        date = to_utc_date(ts, unit="s")
        records.append({"date": date, "total_circulating_usd": float(val)})
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_circulating_usd"])
    return df.sort_values("date")


# -------------------- Collectors: 宏观 --------------------

def fred_series_daily(series_id: str, api_key_env: str = "FRED_API_KEY") -> pd.DataFrame:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = http_get(url, params=params)
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]]


def yfinance_series_daily(ticker: str) -> pd.DataFrame:
    """
    兼容保留的 yfinance 日频下载（如需代理可改用），默认输出收盘价。
    参数：
        ticker: yfinance 符号。
    返回：
        含 date、value 的日频 DataFrame。
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise RuntimeError("yfinance not installed") from e
    tk = yf.Ticker(ticker)
    df = tk.history(period="max")
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {ticker}")
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").dt.normalize()
    df["value"] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["date", "value"]].sort_values("date")


def stooq_series_daily(ticker: str) -> pd.DataFrame:
    """
    从 Stooq 获取日线收盘价，避免代理需求。
    参数：
        ticker: Stooq 符号（如黄金可用 xauusd）。
    返回：
        含 date、value 的日频 DataFrame。
    """
    url = f"https://stooq.pl/q/d/l/?s={ticker.lower()}&i=d"
    resp = http_get(url)
    csv_text = resp.text
    if not csv_text:
        raise RuntimeError(f"stooq empty for {ticker}")

    df = pd.read_csv(StringIO(csv_text))
    if "Data" not in df.columns:
        raise ValueError("stooq csv date column missing")
    close_col = None
    for cand in ["Zamkniecie", "Close", "Close*"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise ValueError("stooq csv close column missing")

    # Stooq 使用本地化列名，统一转为 UTC 日期与数值。
    df["date"] = pd.to_datetime(df["Data"], utc=True, errors="coerce").dt.normalize()
    df["value"] = pd.to_numeric(df[close_col], errors="coerce")
    out = df[["date", "value"]].dropna(subset=["date"]).sort_values("date")
    return out


# -------------------- Collectors: 本地/需 key --------------------

def glassnode_metric_daily(path: str, asset: str = "BTC", api_key_env: str = "GLASSNODE_API_KEY") -> pd.DataFrame:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set")
    url = f"https://api.glassnode.com{path}"
    params = {"a": asset, "i": "24h", "api_key": api_key}
    r = http_get(url, params=params)
    data = r.json()
    records = []
    for item in data:
        ts = item.get("t")
        value = item.get("v")
        date = to_utc_date(ts, unit="s")
        records.append({"date": date, "value": value})
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    return df.sort_values("date")


def binance_metrics_local(csv_path: str = "crypto_metrics_daily.csv", symbol: str = "BTCUSDT") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("date column missing in metrics csv")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if df.empty:
        return pd.DataFrame(columns=["date"])
    non_value_cols = {"date", "symbol"}
    value_cols = [c for c in df.columns if c not in non_value_cols]
    return df[["date"] + value_cols].sort_values("date")


def cryptocompare_chain_pct(fsym: str = "BTC", limit: int = 2000, api_key_env: str = "CRYPTOCOMPARE_API_KEY") -> pd.DataFrame:
    """
    使用 Cryptocompare 免费链上端点，计算主字段的日度环比（pct_change），按日对齐。
    """
    # 安全：不再保留任何内置备用 key。缺少显式配置时必须 Fail Fast——
    # 凭据失效 / 权限不足绝不能表现为“零缺失”的正常输出（RBT_FIN#5 §5）。
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} not set；CryptoCompare 需要显式环境/配置提供凭据，"
            "本模块不再提供内置备用 key。"
        )
    url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
    params = {"fsym": fsym, "limit": limit}
    headers = {"authorization": f"Apikey {api_key}", "Accept": "application/json"}

    last_exc: Optional[Exception] = None
    for i in range(3):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=(5, 20))
            if r.status_code in RETRY_STATUS:
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (i + 1))
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError("cryptocompare request failed")

    series = data.get("Data", {}).get("Data", [])
    if not series:
        return pd.DataFrame(columns=["date"])
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    df = df.drop(columns=[c for c in df.columns if c not in {
        "date",
        "zero_balance_addresses_all_time",
        "unique_addresses_all_time",
        "new_addresses",
        "active_addresses",
        "transaction_count",
        "transaction_count_all_time",
        "large_transaction_count",
        "average_transaction_value",
        "block_height",
        "hashrate",
        "difficulty",
        "block_time",
        "block_size",
        "current_supply",
    }], errors="ignore")
    numeric_cols = [c for c in df.columns if c != "date"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pct_df = df.sort_values("date").copy()
    pct_df[numeric_cols] = pct_df[numeric_cols].pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return pct_df


# -------------------- 注册表 --------------------

def build_registry() -> List[Dataset]:
    reg: List[Dataset] = []

    # 衍生品
    reg.append(Dataset(
        dataset_id="binance_funding_btcusdt",
        category="derivatives/funding",
        endpoint="https://fapi.binance.com/fapi/v1/fundingRate",
        params={"symbol": "BTCUSDT", "limit": 1000, "startTime": 1},
        collector=lambda **_: binance_funding_rate_daily(symbol="BTCUSDT", limit=1000, start_ms=1),
        agg="mean",
        remark="8h→日均 | startTime>0 自最早端分页，自动适配降序响应"
    ))
    reg.append(Dataset(
        dataset_id="binance_index_price_1d",
        category="derivatives/index_price",
        endpoint="https://fapi.binance.com/fapi/v1/indexPriceKlines",
        params={"pair": "BTCUSDT", "interval": "1d", "limit": 1500},
        collector=lambda **_: binance_kline_daily("/fapi/v1/indexPriceKlines", "pair", "BTCUSDT", "index_close", 1500),
        agg="last",
        remark="指数价 close"
    ))
    reg.append(Dataset(
        dataset_id="binance_premium_index_1d",
        category="derivatives/premium_index",
        endpoint="https://fapi.binance.com/fapi/v1/premiumIndexKlines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1500},
        collector=lambda **_: binance_kline_daily("/fapi/v1/premiumIndexKlines", "symbol", "BTCUSDT", "premium_close", 1500),
        agg="last",
        remark="溢价指数 close"
    ))
    reg.append(Dataset(
        dataset_id="binance_mark_price_1d",
        category="derivatives/mark_price",
        endpoint="https://fapi.binance.com/fapi/v1/markPriceKlines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1500},
        collector=lambda **_: binance_kline_daily("/fapi/v1/markPriceKlines", "symbol", "BTCUSDT", "mark_close", 1500),
        agg="last",
        remark="标记价 close"
    ))
    reg.append(Dataset(
        dataset_id="bybit_funding_btcusdt",
        category="derivatives/funding",
        endpoint="https://api.bybit.com/v5/market/funding/history",
        params={"category": "linear", "symbol": "BTCUSDT", "limit": 200, "startTime": 0},
        collector=lambda **_: bybit_funding_rate_daily(symbol="BTCUSDT", category="linear", limit=200, start_ms=0),
        agg="mean",
        remark="8h→日均 | params={\"category\": \"linear\", \"symbol\": \"BTCUSDT\", \"limit\": 200, \"startTime\": 0}"
    ))

    # 链上：Blockchain.com 各指标
    for cid in ["hash-rate", "difficulty", "mempool-size", "n-transactions", "transaction-fees", "n-unique-addresses"]:
        reg.append(Dataset(
            dataset_id=f"blockchain_{cid.replace('-', '_')}",
            category="onchain/blockchain_com",
            endpoint=f"https://api.blockchain.info/charts/{cid}",
            params={"timespan": "all", "format": "json"},
            collector=lambda cid=cid, **_: blockchain_chart_daily(cid),
            agg="last",
            remark="Blockchain.com Charts 日频 | 早期链上数据稀疏，建模建议 2020-01-01 之后使用"
        ))

    # 情绪/关注度/流动性
    reg.append(Dataset(
        dataset_id="altme_fng",
        category="sentiment",
        endpoint="https://api.alternative.me/fng/",
        params={"limit": 0},
        collector=lambda **_: altme_fng_daily(limit=0),
        agg="last",
        remark="Fear&Greed 全历史"
    ))
    reg.append(Dataset(
        dataset_id="wikipedia_bitcoin_pageviews",
        category="attention",
        endpoint="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article",
        params={"article": "Bitcoin", "project": "en.wikipedia.org"},
        collector=lambda **_: wikipedia_pageviews_daily(article="Bitcoin", project="en.wikipedia.org"),
        agg="sum",
        remark="2015-07-01 起 en.wikipedia Bitcoin 条目 PV"
    ))
    reg.append(Dataset(
        dataset_id="defillama_stablecoins_total",
        category="liquidity/stablecoin",
        endpoint="https://stablecoins.llama.fi/stablecoincharts/all",
        params={},
        collector=lambda **_: defillama_stablecoins_daily(),
        agg="last",
        remark="全稳定币流通总量"
    ))

    # 宏观：FRED（需 key，替代 yfinance，避免代理问题）
    fred_series = [
        ("DTWEXBGS", "macro_dxy", "广义美元指数 DTWEXBGS"),
        ("SP500", "macro_spx", "标普500 SP500"),
        ("DGS10", "macro_us10y", "美国10Y 国债收益率 DGS10"),
        ("VIXCLS", "macro_vix", "VIXCLS"),
        ("T10Y2Y", "fred_t10y2y", "美债 10Y-2Y 利差"),
        ("DGS10", "fred_dgs10", "美国10Y 国债收益率 DGS10"),
        ("DTWEXBGS", "fred_dtwexbgs", "广义美元指数 DTWEXBGS")
    ]
    for sid, name, desc in fred_series:
        reg.append(Dataset(
            dataset_id=name,
            category="macro/fred",
            endpoint="https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "file_type": "json", "api_key": "${FRED_API_KEY}"},
            collector=lambda sid=sid, **_: fred_series_daily(sid),
            agg="last",
            remark=f"{desc} | 需要 FRED_API_KEY"
        ))

    # 宏观：Stooq（无 key，用于黄金现货）
    reg.append(Dataset(
        dataset_id="macro_gold",
        category="macro/stooq",
        endpoint="https://stooq.pl/q/d/l/",
        params={"ticker": "xauusd", "interval": "d"},
        collector=lambda **_: stooq_series_daily("xauusd"),
        agg="last",
        remark="Stooq XAUUSD 收盘价，无需代理/无需 key（FRED 无黄金现货价格）"
    ))

    # Glassnode（需 key）
    reg.append(Dataset(
        dataset_id="glassnode_active_addresses_btc",
        category="onchain/glassnode",
        endpoint="https://api.glassnode.com/v1/metrics/addresses/active_count",
        params={"a": "BTC", "i": "24h", "api_key": "${GLASSNODE_API_KEY}"},
        collector=lambda **_: glassnode_metric_daily("/v1/metrics/addresses/active_count", asset="BTC"),
        agg="last",
        remark="需要 GLASSNODE_API_KEY"
    ))

    # Cryptocompare 链上环比
    reg.append(Dataset(
        dataset_id="cryptocompare_btc_chain_pct",
        category="onchain/cryptocompare",
        endpoint="https://min-api.cryptocompare.com/data/blockchain/histo/day",
        params={"fsym": "BTC", "limit": 2000, "api_key": "${CRYPTOCOMPARE_API_KEY or built-in}"},
        collector=lambda **_: cryptocompare_chain_pct(fsym="BTC", limit=2000),
        agg="mean",
        remark="链上核心字段环比，免费 key 可用 | 字段: transaction_count, new_addresses, active_addresses, average_transaction_value, block_height, hashrate, difficulty, block_time, block_size, current_supply"
    ))

    # 本地 metrics（可选）
    reg.append(Dataset(
        dataset_id="binance_metrics_local",
        category="derivatives/local",
        endpoint="file://crypto_metrics_daily.csv",
        params={"symbol": "BTCUSDT"},
        collector=lambda **_: binance_metrics_local(csv_path="crypto_metrics_daily.csv", symbol="BTCUSDT"),
        agg="last",
        remark="本地 metrics CSV，如缺失会报错"
    ))

    return reg


# -------------------- 主流程 --------------------

def main():
    """
    兼容 CLI 入口：对全部注册项真实采集一次并输出概览 CSV。

    注意：本入口会触发刷新（每项都会发起真实请求），不是只读概览。
    """
    # 在 CLI 入口显式加载 .env：凭据注入是一个明确动作，不是导入副作用。
    load_env_from_dotenv()
    print(
        "[注意] data_layer_unified_opt.py 是会触发刷新的兼容入口：本次运行会对每个注册项"
        "真实发起采集。只读概览请使用 `rbt-fin overview --cache-root <dir>`。"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets_unified_summary.csv", help="输出 CSV 路径")
    ap.add_argument("--limit", type=int, default=300, help="从最早日期起抽取的日样本长度")
    args = ap.parse_args()

    reg = build_registry()
    rows = []

    for ds in reg:
        earliest_date = None
        sample_df = pd.DataFrame()
        err_msg = None
        try:
            raw = ds.collector()
            daily = ensure_daily(raw, date_col="date", agg=ds.agg)
            if not daily.empty:
                earliest_date = pd.to_datetime(daily["date"].min(), utc=True).date().isoformat()
                sample_df = sample_first_n_days(daily, n=args.limit, date_col="date")
            else:
                sample_df = pd.DataFrame(columns=["date"])
        except Exception as ex:
            err_msg = f"{type(ex).__name__}: {ex}"
            sample_df = pd.DataFrame(columns=["date"])

        na_cells, na_ratio = na_cells_info(sample_df, date_col="date")
        sample_rows = int(len(sample_df)) if not sample_df.empty else 0
        sample_start = sample_df["date"].min().date().isoformat() if sample_rows else None
        sample_end = sample_df["date"].max().date().isoformat() if sample_rows else None

        remark_parts = [ds.remark]
        if err_msg:
            remark_parts.append(f"error={err_msg}")
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
            "na_cells": na_cells,
            "na_ratio": round(na_ratio, 6),
            "remark": remark
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["category", "dataset_id"]).reset_index(drop=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
