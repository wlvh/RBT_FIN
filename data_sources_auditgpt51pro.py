#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源审计脚本（只使用日频及以上 + 免费 + 非 30 天窗口）

功能：
- 模块化管理多个数据源（Binance / Fear&Greed / TODO: FRED / Glassnode）
- 对每个数据源：
    - 探测最早可用日期（earliest_date）
    - 从 earliest_date 起抽取最多 300 行日频数据
    - 统计这 300 行里的缺失值情况
- 输出 data_sources_overview.csv，总结：
    - data_type, endpoint, earliest_date, sample_rows,
      missing_summary, params, notes
"""

import os
import json
import io
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


# ========= 通用配置 =========

UTC = timezone.utc


@dataclass
class DataSourceInfo:
    name: str
    endpoint: str
    earliest_date: Optional[pd.Timestamp]
    sample_rows: int
    missing_summary: str
    params: Dict[str, str]
    notes: str = ""


class BaseDataSource:
    """
    所有数据源的抽象基类：
    - get_sample：返回 (earliest_date, df_sample)
    - build_info：包装成 DataSourceInfo
    """

    name: str
    endpoint: str
    params: Dict[str, str]
    notes: str

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        """
        返回 (earliest_date, df_sample)，df_sample 为日频数据，
        index 为 UTC 日期（normalize 到 00:00）
        """
        raise NotImplementedError

    def build_info(self, n_rows: int = 300) -> DataSourceInfo:
        earliest, df = self.get_sample(n_rows=n_rows)

        # 统一 index 为日期
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{self.name}: df.index 必须是 DatetimeIndex")
        df = df.sort_index()
        # 这里不强制 reindex 为完整日历，只做简单缺失统计
        missing_counts = df.isna().sum()

        missing_summary = "; ".join(
            f"{col}: {int(cnt)}" for col, cnt in missing_counts.items()
        )

        info = DataSourceInfo(
            name=self.name,
            endpoint=self.endpoint,
            earliest_date=earliest,
            sample_rows=len(df),
            missing_summary=missing_summary,
            params=self.params,
            notes=self.notes,
        )
        return info


# ========= Binance 数据源 =========

class BinanceFundingDaily(BaseDataSource):
    """
    Binance USDT-M 永续资金费率，按日重采样（每日最后一笔）。
    Endpoint: /fapi/v1/fundingRate
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.name = f"BinanceFundingDaily_{symbol}"
        self.endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.params = {
            "symbol": symbol,
            "limit": "1000",  # 只需要最早的若干条
            "startTime": "0",  # 从 1970-01-01 开始，API 会返回能给的最早记录
        }
        self.notes = "日频资金费率，按每日最后一笔 funding 重采样；免费且非 30 天限制。"

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        resp = requests.get(self.endpoint, params=self.params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"{self.name}: 未从 API 获得任何数据")

        df = pd.DataFrame(data)
        # fundingTime 是 ms 时间戳
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df.set_index("fundingTime").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # 按日期重采样（每日最后一笔）
        daily = df["fundingRate"].resample("D").last().dropna()
        daily.index = daily.index.tz_convert(UTC)
        earliest = daily.index[0]

        sample = daily.iloc[:n_rows].to_frame(name="fundingRate")
        return earliest, sample


class _BinanceKlinesLikeDaily(BaseDataSource):
    """
    通用：Binance USDT-M Klines 风格的 K 线（premiumIndexKlines / markPriceKlines / indexPriceKlines）
    """

    def __init__(self, name: str, endpoint_path: str, symbol: str = "BTCUSDT", use_pair: bool = False):
        self.symbol = symbol
        self.name = f"{name}_{symbol}"
        self.base_url = "https://fapi.binance.com"
        self.endpoint_path = endpoint_path
        self.endpoint = self.base_url + self.endpoint_path
        # indexPriceKlines 需要 pair 参数，其余使用 symbol
        param_key = "pair" if use_pair else "symbol"
        self.params = {
            param_key: symbol,
            "interval": "1d",
            "startTime": "0",  # 从最早时间开始
            "limit": "1000",   # 只需要最早的若干天
        }
        self.notes = (
            f"{name} 日线 K 线（open, high, low, close），"
            "使用 startTime=0 从最早时间开始拉取；免费且非 30 天限制。"
        )

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        resp = requests.get(self.endpoint, params=self.params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError(f"{self.name}: 未从 API 获得任何数据")

        # Kline 返回是数组列表
        cols = [
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "qav", "numTrades", "takerBaseVol",
            "takerQuoteVol", "ignore"
        ]
        df = pd.DataFrame(raw, columns=cols[: len(raw[0])])
        df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
        df = df.set_index("openTime").sort_index()

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        daily = df[["open", "high", "low", "close"]].copy()
        daily.index = daily.index.tz_convert(UTC)
        earliest = daily.index[0]

        sample = daily.iloc[:n_rows]
        return earliest, sample


class BinancePremiumIndexDaily(_BinanceKlinesLikeDaily):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(
            name="BinancePremiumIndexDaily",
            endpoint_path="/fapi/v1/premiumIndexKlines",
            symbol=symbol,
        )


class BinanceMarkPriceDaily(_BinanceKlinesLikeDaily):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(
            name="BinanceMarkPriceDaily",
            endpoint_path="/fapi/v1/markPriceKlines",
            symbol=symbol,
        )


class BinanceIndexPriceDaily(_BinanceKlinesLikeDaily):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(
            name="BinanceIndexPriceDaily",
            endpoint_path="/fapi/v1/indexPriceKlines",
            symbol=symbol,
            use_pair=True,  # indexPriceKlines 接口要求使用 pair 参数
        )


class BinanceMetricsLocalDaily(BaseDataSource):
    """
    使用你已有脚本生成的 crypto_metrics_daily.csv 作为日频 metrics 数据源。

    原始来源：Binance Vision metrics zip
    https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip

    这里为了简单，直接读取本地 CSV：
    - 必须包含列：symbol, date, 以及你之前计算的 *_close / *_mean 等字段
    """

    def __init__(self, csv_path: str = "crypto_metrics_daily.csv", symbol: str = "BTCUSDT"):
        self.csv_path = csv_path
        self.symbol = symbol
        self.name = f"BinanceMetricsLocalDaily_{symbol}"
        # endpoint 写成原始数据源的说明
        self.endpoint = (
            "https://data.binance.vision/data/futures/um/daily/metrics/"
            f"{symbol}/{symbol}-metrics-YYYY-MM-DD.zip"
        )
        self.params = {
            "symbol": symbol,
            "local_csv": csv_path,
        }
        self.notes = (
            "使用本地 crypto_metrics_daily.csv（由你现有 future_data_collector.py 生成）。"
            "TODO: 如有需要，可改写为直接从 Binance Vision 下载 zip 并聚合。"
        )

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"{self.name}: 本地 CSV {self.csv_path} 不存在，请先运行你的 metrics 生成脚本。"
            )

        df = pd.read_csv(self.csv_path)
        if "date" not in df.columns or "symbol" not in df.columns:
            raise ValueError(f"{self.name}: CSV 中缺少 date 或 symbol 列。")

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df[df["symbol"] == self.symbol].copy()
        if df.empty:
            raise ValueError(f"{self.name}: 在 CSV 中没有找到 symbol={self.symbol} 的数据。")

        df = df.sort_values("date")
        earliest = df["date"].iloc[0]

        df = df.set_index("date")
        # 去掉 symbol 列，保留数值类字段
        df = df.drop(columns=["symbol"], errors="ignore")

        sample = df.iloc[:n_rows]
        return earliest, sample


# ========= 外部免费数据源 =========

class FearGreedDaily(BaseDataSource):
    """
    Crypto Fear & Greed Index（日频）
    Endpoint: https://api.alternative.me/fng/
    """

    def __init__(self):
        self.name = "FearGreedDaily"
        self.endpoint = "https://api.alternative.me/fng/"
        self.params = {
            "limit": "0",   # 返回全历史
            "format": "json",
        }
        self.notes = "Fear & Greed 指数（0-100），日频，完全免费 API。"

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        resp = requests.get(self.endpoint, params=self.params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        data = raw.get("data", [])
        if not data:
            raise ValueError(f"{self.name}: 未从 API 获得任何数据")

        records = []
        for item in data:
            # timestamp 为秒
            ts = int(item["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=UTC)
            date = pd.Timestamp(dt.date(), tz=UTC)
            value = float(item["value"])
            classification = item.get("value_classification", "")
            records.append({"date": date, "value": value, "class": classification})

        df = pd.DataFrame(records).drop_duplicates(subset="date").sort_values("date")
        df = df.set_index("date")
        earliest = df.index[0]

        sample = df.iloc[:n_rows]
        return earliest, sample


# ========= TODO: 免费但需要 API key 的数据源 =========

class FredSeriesDaily(BaseDataSource):
    """
    FRED 宏观时间序列（例如 10Y 国债 DGS10）
    需要环境变量 FRED_API_KEY，否则跳过。
    """

    def __init__(self, series_id: str = "DGS10"):
        self.series_id = series_id
        self.name = f"FRED_{series_id}_Daily"
        self.endpoint = "https://api.stlouisfed.org/fred/series/observations"
        self.params = {
            "series_id": series_id,
            "file_type": "json",
            # observation_start 在 get_sample 里确定
        }
        self.notes = (
            "TODO: 需要设置环境变量 FRED_API_KEY 才能获取数据；"
            "本脚本中默认不启用该数据源。"
        )

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY 未设置，暂不拉取 FRED 数据（TODO）")

        params = self.params.copy()
        params["api_key"] = api_key
        # 从很早的日期开始
        params["observation_start"] = "1960-01-01"

        resp = requests.get(self.endpoint, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        obs = raw.get("observations", [])
        if not obs:
            raise ValueError(f"{self.name}: 未从 FRED 获得数据")

        rows = []
        for o in obs:
            date_str = o["date"]
            value_str = o["value"]
            if value_str == ".":
                value = None
            else:
                value = float(value_str)
            date = pd.Timestamp(date_str, tz=UTC)
            rows.append({"date": date, self.series_id: value})

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        earliest = df.index[0]
        sample = df.iloc[:n_rows]
        return earliest, sample


class GlassnodeMetricDaily(BaseDataSource):
    """
    Glassnode 链上指标（例如 BTC Active Addresses）。
    需要环境变量 GLASSNODE_API_KEY。
    """

    def __init__(self, path: str = "/v1/metrics/addresses/active_count", asset: str = "BTC"):
        self.path = path
        self.asset = asset
        self.name = f"Glassnode_{asset}_{path.replace('/', '_')}_Daily"
        self.base_url = "https://api.glassnode.com"
        self.endpoint = self.base_url + path
        self.params = {
            "a": asset,
            "i": "24h",
            # s / u / api_key 在 get_sample 里补充
        }
        self.notes = (
            "TODO: Glassnode 免费 Basic API，需要 GLASSNODE_API_KEY。"
            "本脚本默认不启用该数据源。"
        )

    def get_sample(self, n_rows: int = 300) -> Tuple[pd.Timestamp, pd.DataFrame]:
        api_key = os.getenv("GLASSNODE_API_KEY")
        if not api_key:
            raise RuntimeError("GLASSNODE_API_KEY 未设置，暂不拉取 Glassnode 数据（TODO）")

        params = self.params.copy()
        params["api_key"] = api_key
        # s 不写时 Glassnode 默认返回全历史，可以直接取
        resp = requests.get(self.endpoint, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError(f"{self.name}: 未从 Glassnode 获得数据")

        rows = []
        for item in raw:
            ts = int(item["t"])
            dt = datetime.fromtimestamp(ts, tz=UTC)
            date = pd.Timestamp(dt.date(), tz=UTC)
            value = item.get("v")
            rows.append({"date": date, "value": value})

        df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
        df = df.set_index("date")
        earliest = df.index[0]
        sample = df.iloc[:n_rows]
        return earliest, sample


# ========= 主程序：生成 CSV 概览 =========

def main():
    # 只启用：Binance + Fear&Greed（完全免费 & 不需要 key）
    data_sources: List[BaseDataSource] = [
        BinanceFundingDaily(symbol="BTCUSDT"),
        BinancePremiumIndexDaily(symbol="BTCUSDT"),
        BinanceMarkPriceDaily(symbol="BTCUSDT"),
        BinanceIndexPriceDaily(symbol="BTCUSDT"),
        BinanceMetricsLocalDaily(csv_path="crypto_metrics_daily.csv", symbol="BTCUSDT"),
        FearGreedDaily(),
        # 想启用 FRED / Glassnode 时，把下面两个加进来即可：
        # FredSeriesDaily(series_id="DGS10"),
        # GlassnodeMetricDaily(path="/v1/metrics/addresses/active_count", asset="BTC"),
    ]

    infos: List[DataSourceInfo] = []

    for ds in data_sources:
        print(f"\n=== 处理数据源: {ds.name} ===")
        try:
            info = ds.build_info(n_rows=300)
            print(f"最早日期: {info.earliest_date}, sample_rows: {info.sample_rows}")
            print(f"缺失值概览: {info.missing_summary}")
        except Exception as e:
            # 如果出现错误，也在 CSV 里记录下来
            print(f"数据源 {ds.name} 处理出错: {e}")
            info = DataSourceInfo(
                name=ds.name,
                endpoint=getattr(ds, "endpoint", ""),
                earliest_date=None,
                sample_rows=0,
                missing_summary=f"ERROR: {e}",
                params=getattr(ds, "params", {}),
                notes=getattr(ds, "notes", ""),
            )
        infos.append(info)

    # 汇总成 DataFrame 输出
    rows = []
    for info in infos:
        row = {
            "data_type": info.name,
            "endpoint": info.endpoint,
            "earliest_date": (
                info.earliest_date.strftime("%Y-%m-%d")
                if info.earliest_date is not None
                else None
            ),
            "sample_rows": info.sample_rows,
            "missing_summary": info.missing_summary,
            "params": json.dumps(info.params, ensure_ascii=False),
            "notes": info.notes,
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    output_file = "data_sources_overviewgpt51pro.csv"
    df_summary.to_csv(output_file, index=False)
    print(f"\n已生成数据源概览 CSV: {output_file}")


if __name__ == "__main__":
    main()
