import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta


@dataclass
class DataSource:
    """
    描述一个“日频数据源”的元信息。
    fetch_func 统一签名：
        fetch_func(sample_size: int) -> (df_sample: DataFrame, earliest_date: 'YYYY-MM-DD')
    其中 df_sample 的索引统一为 'date'（日频 DatetimeIndex）。
    """
    name: str
    category: str
    endpoint: str
    frequency: str  # 'daily', 'weekly', '8h_raw -> daily' 等
    free: bool
    only_recent_30d: bool
    notes: str
    params_example: Dict[str, Any] = field(default_factory=dict)
    implemented: bool = True
    fetch_func: Optional[Callable[[int], Tuple[pd.DataFrame, str]]] = None


# ---------- 具体数据源的抓取函数（已实现的都是日频或以上 & 免费 & 历史>30天） ----------

def fetch_binance_funding_daily(sample_size: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Binance Futures: BTCUSDT USDT-M 永续合约资金费率（原始 8h），聚合为日频：
    - 每日最后一笔 fundingRate 作为该日资金费率。
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/fundingRate"
    symbol = "BTCUSDT"
    limit = 1000

    # 从极早时间开始（startTime=0），Binance 会从最早有数据的地方返回
    params = {
        "symbol": symbol,
        "limit": limit,
        "startTime": 0
    }
    resp = requests.get(base_url + endpoint, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame(), ""

    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df = df.set_index("fundingTime").sort_index()
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

    # 最早可用的日期（UTC）
    earliest_date = df.index[0].normalize()

    # 以日为单位重采样，取每天最后一个 funding
    daily = df["fundingRate"].resample("D").last().to_frame(name="fundingRate")

    # 只取从最早日期起的 sample_size 天
    end_date = earliest_date + timedelta(days=sample_size - 1)
    daily = daily.loc[earliest_date:end_date]

    # 为了暴露缺失：按完整日历重建索引
    full_range = pd.date_range(start=earliest_date, end=end_date, freq="D")
    daily = daily.reindex(full_range)
    daily.index.name = "date"

    return daily, earliest_date.strftime("%Y-%m-%d")


def fetch_blockchain_chart(chart_type: str, sample_size: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Blockchain.com 图表通用抓取函数。
    chart_type 示例：
        - 'n-unique-addresses'
        - 'n-transactions'
        - 'hash-rate'
    返回：
        - 从最早日期起的 sample_size 天日频 data frame（索引为 date）
        - earliest_date 字符串
    """
    # API 需要使用 api.blockchain.info 域名，若用 blockchain.info 会返回 HTML
    base_url = "https://api.blockchain.info/charts/"
    url = base_url + chart_type
    params = {
        "format": "json",
        "timespan": "all"   # 返回全历史
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(), ""

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["x"], unit="s")
    df = df.set_index("date").sort_index()
    df["value"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[["value"]]

    earliest_date = df.index[0].normalize()
    end_date = earliest_date + timedelta(days=sample_size - 1)
    df_sample = df.loc[earliest_date:end_date]

    full_range = pd.date_range(start=earliest_date, end=end_date, freq="D")
    df_sample = df_sample.reindex(full_range)
    df_sample.index.name = "date"

    return df_sample, earliest_date.strftime("%Y-%m-%d")


def fetch_fear_greed(sample_size: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Crypto Fear & Greed Index（日频）。
    来自 alternative.me 免费 API。
    """
    url = "https://api.alternative.me/fng/"
    params = {
        "limit": 0,           # 0 = 全历史
        "format": "json",
        "date_format": "unix"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame(), ""

    df = pd.DataFrame(data)

    # 兼容新旧 timestamp 格式：有时是 Unix 秒，有时是 "DD-MM-YYYY"
    def parse_ts(ts):
        ts_num = pd.to_numeric(ts, errors="coerce")
        if pd.notna(ts_num):
            return pd.to_datetime(int(ts_num), unit="s")
        return pd.to_datetime(str(ts), dayfirst=True, errors="coerce")

    df["date"] = df["timestamp"].apply(parse_ts)
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["value"]]

    earliest_date = df.index[0].normalize()
    end_date = earliest_date + timedelta(days=sample_size - 1)
    df_sample = df.loc[earliest_date:end_date]

    full_range = pd.date_range(start=earliest_date, end=end_date, freq="D")
    df_sample = df_sample.reindex(full_range)
    df_sample.index.name = "date"

    return df_sample, earliest_date.strftime("%Y-%m-%d")


# ---------- 盘点所有数据源，并输出 overview CSV ----------

def evaluate_data_sources(sample_size: int = 300,
                          output_csv: str = "data_sources_overview.csv") -> pd.DataFrame:
    """
    对所有数据源做一次“盘点”：
    - 对每个数据源：
        - 调 fetch_func(sample_size)，得到 (df_sample, earliest_date)
        - 基于前 sample_size 天统计缺失情况
    - 输出 CSV，包含：
        name, category, endpoint, frequency,
        params_example（作为备注，包含调用参数），
        implemented（是否已实现）、earliest_date、missing_summary、notes
    """
    data_sources: List[DataSource] = [
        # 1. Binance Futures: BTCUSDT 资金费率（日频）
        DataSource(
            name="binance_futures_btcusdt_funding",
            category="binance_futures",
            endpoint="https://fapi.binance.com/fapi/v1/fundingRate",
            frequency="8h_raw -> daily",
            free=True,
            only_recent_30d=False,
            notes="USDT-M perpetual funding rate for BTCUSDT, aggregated to daily (last funding of each day).",
            params_example={"symbol": "BTCUSDT", "limit": 1000, "startTime": 0},
            implemented=True,
            fetch_func=fetch_binance_funding_daily
        ),

        # 2. On-chain: Blockchain.com 链上因子（日频）
        DataSource(
            name="blockchain_unique_addresses",
            category="onchain",
            endpoint="https://api.blockchain.info/charts/n-unique-addresses",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="Number of unique Bitcoin addresses used per day from Blockchain.com charts.",
            params_example={"format": "json", "timespan": "all"},
            implemented=True,
            fetch_func=lambda n: fetch_blockchain_chart("n-unique-addresses", n)
        ),
        DataSource(
            name="blockchain_transactions",
            category="onchain",
            endpoint="https://api.blockchain.info/charts/n-transactions",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="Number of confirmed Bitcoin transactions per day from Blockchain.com charts.",
            params_example={"format": "json", "timespan": "all"},
            implemented=True,
            fetch_func=lambda n: fetch_blockchain_chart("n-transactions", n)
        ),
        DataSource(
            name="blockchain_hash_rate",
            category="onchain",
            endpoint="https://api.blockchain.info/charts/hash-rate",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="Estimated Bitcoin hash rate from Blockchain.com charts.",
            params_example={"format": "json", "timespan": "all"},
            implemented=True,
            fetch_func=lambda n: fetch_blockchain_chart("hash-rate", n)
        ),

        # 3. Sentiment: Crypto Fear & Greed（情绪）
        DataSource(
            name="crypto_fear_greed_index",
            category="sentiment",
            endpoint="https://api.alternative.me/fng/",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="Crypto Fear & Greed Index from alternative.me.",
            params_example={"limit": 0, "format": "json", "date_format": "unix"},
            implemented=True,
            fetch_func=fetch_fear_greed
        ),

        # --- 下面这些是“免费但是暂时不实现”的 TODO，先写在 overview 里，方便你之后补齐 ---

        # Binance metrics 日频：你原脚本 get_future_metrics() 已经做了，这里只做登记
        DataSource(
            name="binance_futures_btcusdt_metrics_daily",
            category="binance_futures",
            endpoint="https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip",
            frequency="intraday_raw -> daily",
            free=True,
            only_recent_30d=False,
            notes=(
                "Daily metrics zip for BTCUSDT (open interest, long/short ratios, taker long/short vol ratio). "
                "TODO: reuse your existing get_future_metrics() pipeline here, read crypto_metrics_daily.csv "
                "and返回日频 DataFrame。"
            ),
            params_example={"symbol": "BTCUSDT", "date": "YYYY-MM-DD"},
            implemented=False,
            fetch_func=None
        ),

        # FRED: 10Y 国债收益率（DGS10）—— 需要免费 API key
        DataSource(
            name="fred_dgs10_10y_yield",
            category="macro",
            endpoint="https://api.stlouisfed.org/fred/series/observations",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="US 10-Year Treasury constant maturity rate (DGS10). Requires FRED API key. TODO: implement.",
            params_example={"series_id": "DGS10", "api_key": "YOUR_FRED_KEY", "file_type": "json"},
            implemented=False,
            fetch_func=None
        ),

        # FRED: Nominal Broad U.S. Dollar Index（DTWEXBGS）—— 需要免费 API key
        DataSource(
            name="fred_dtwexbgs_dollar_index",
            category="macro",
            endpoint="https://api.stlouisfed.org/fred/series/observations",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="Nominal Broad U.S. Dollar Index (DTWEXBGS). Requires FRED API key. TODO: implement.",
            params_example={"series_id": "DTWEXBGS", "api_key": "YOUR_FRED_KEY", "file_type": "json"},
            implemented=False,
            fetch_func=None
        ),

        # Alpha Vantage: SPY 日线 —— 需要免费 API key
        DataSource(
            name="alpha_vantage_spy",
            category="macro",
            endpoint="https://www.alphavantage.co/query",
            frequency="daily",
            free=True,
            only_recent_30d=False,
            notes="SPY daily OHLC from Alpha Vantage, as a proxy for US equities. Requires API key. TODO: implement.",
            params_example={"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": "SPY", "apikey": "YOUR_API_KEY"},
            implemented=False,
            fetch_func=None
        ),
    ]

    rows = []
    for ds in data_sources:
        # 过滤掉非免费数据
        if not ds.free:
            continue

        # 过滤掉“只保留最近30天”的数据类型（目前列表里本来就没放）
        if ds.only_recent_30d:
            continue

        row: Dict[str, Any] = {
            "name": ds.name,
            "category": ds.category,
            "endpoint": ds.endpoint,
            "frequency": ds.frequency,
            # 备注里把典型参数拼成 querystring 形式，便于你一眼看到怎么调用
            "params_example": "&".join(f"{k}={v}" for k, v in ds.params_example.items()),
            "implemented": ds.implemented,
            "earliest_date": "",
            "missing_summary": "",
            "notes": ds.notes,
        }

        if not ds.implemented or ds.fetch_func is None:
            row["earliest_date"] = "TODO"
            row["missing_summary"] = "TODO (fetch function not implemented)"
        else:
            try:
                df_sample, earliest_date = ds.fetch_func(sample_size)
                row["earliest_date"] = earliest_date or ""
                if df_sample is None or df_sample.empty:
                    row["missing_summary"] = "no data returned"
                else:
                    total_cells = int(df_sample.size)
                    missing_cells = int(df_sample.isna().sum().sum())
                    missing_pct = (missing_cells / total_cells) if total_cells > 0 else 0.0
                    row["missing_summary"] = (
                        f"rows={len(df_sample)}, cols={df_sample.shape[1]}, "
                        f"missing_cells={missing_cells}, missing_pct={missing_pct:.4f}"
                    )
            except Exception as e:
                row["earliest_date"] = "ERROR"
                row["missing_summary"] = f"error: {e}"

        rows.append(row)

    overview_df = pd.DataFrame(rows)
    overview_df.to_csv(output_csv, index=False)
    return overview_df


if __name__ == "__main__":
    # 本脚本运行时会：
    #  - 遍历所有“free & 非30天限制”的数据源
    #  - 对已有实现的，自动探测 earliest_date，并从该日起抽取 300 天日数据
    #  - 统计缺失值情况
    #  - 输出 data_sources_overview.csv 作为“数据地图”
    evaluate_data_sources(sample_size=300, output_csv="data_sources_overviewaltas.csv")
