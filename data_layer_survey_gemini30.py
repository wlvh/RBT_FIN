import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
from abc import ABC, abstractmethod

# ==========================================
# 1. 抽象基类 (Modular Interface)
# ==========================================


class BaseDataSource(ABC):
    def __init__(self, name, category, endpoint, params_note):
        self.name = name
        self.category = category
        self.endpoint = endpoint
        self.params_note = params_note
        self.earliest_date = None
        self.sample_data = None
        self.is_paid_placeholder = False

    @abstractmethod
    def fetch_survey_data(self):
        """
        执行调研：
        1. 获取最早可用日期
        2. 获取从最早日期开始的 300 条数据
        3. 返回 (earliest_date, dataframe)
        """
        pass

# ==========================================
# 2. 具体实现类 (Concrete Implementations)
# ==========================================


class YahooMacroSource(BaseDataSource):
    def __init__(self, name, ticker):
        super().__init__(
            name=name,
            category="Macro (Global)",
            endpoint="https://query2.finance.yahoo.com/v8/finance/chart/",
            params_note=f"Ticker: {ticker}, Interval: 1d"
        )
        self.ticker = ticker

    def fetch_survey_data(self):
        try:
            # yfinance 获取 max 历史是最快的方法
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(period="max")

            if df.empty:
                return None, None

            df.reset_index(inplace=True)
            # 统一时间列名为 'date' 并转为 datetime且去掉时区
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.rename(columns={'Date': 'date', 'Close': 'value'})
            df = df[['date', 'value']].sort_values('date')

            earliest_date = df['date'].iloc[0]

            # 截取前300条
            sample_df = df.head(300).copy()
            return earliest_date, sample_df

        except Exception as e:
            print(f"Error fetching {self.name}: {e}")
            return None, None


class BlockchainComSource(BaseDataSource):
    def __init__(self, name, slug):
        super().__init__(
            name=name,
            category="On-Chain (Fundamental)",
            endpoint=f"https://api.blockchain.info/charts/{slug}",
            params_note="timespan=all, format=json"
        )
        self.slug = slug

    def fetch_survey_data(self):
        try:
            url = f"{self.endpoint}?timespan=all&format=json"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            values = data.get('values', [])
            if not values:
                return None, None

            df = pd.DataFrame(values)
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df = df.rename(columns={'y': 'value'})
            df = df[['date', 'value']].sort_values('date')

            earliest_date = df['date'].iloc[0]
            sample_df = df.head(300).copy()

            return earliest_date, sample_df
        except Exception as e:
            print(f"Error fetching {self.name}: {e}")
            return None, None


class AlternativeMeSource(BaseDataSource):
    def __init__(self):
        super().__init__(
            name="Fear & Greed Index",
            category="Sentiment",
            endpoint="https://api.alternative.me/fng/",
            params_note="limit=0 (All History)"
        )

    def fetch_survey_data(self):
        try:
            url = f"{self.endpoint}?limit=0"
            resp = requests.get(url, timeout=15)
            data = resp.json()

            values = data.get('data', [])
            if not values:
                return None, None

            df = pd.DataFrame(values)
            df['date'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
            df['value'] = pd.to_numeric(df['value'])
            df = df[['date', 'value']].sort_values('date')

            earliest_date = df['date'].iloc[0]
            sample_df = df.head(300).copy()

            return earliest_date, sample_df
        except Exception as e:
            print(f"Error fetching {self.name}: {e}")
            return None, None


class DefiLlamaStablecoinSource(BaseDataSource):
    def __init__(self):
        super().__init__(
            name="Total Stablecoin Market Cap",
            category="Liquidity",
            endpoint="https://stablecoins.llama.fi/stablecoincharts/all",
            params_note="N/A"
        )

    def fetch_survey_data(self):
        try:
            resp = requests.get(self.endpoint, timeout=15)
            data = resp.json()

            processed_data = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue

                raw_date = entry.get('date')
                usd_bucket = entry.get('totalCirculatingUSD') or entry.get('totalCirculating')
                if not (raw_date and isinstance(usd_bucket, dict)):
                    continue

                # Prefer peggedUSD if present; otherwise sum numeric fields
                if 'peggedUSD' in usd_bucket:
                    value = usd_bucket.get('peggedUSD')
                else:
                    numeric_vals = [
                        v for v in usd_bucket.values() if isinstance(v, (int, float))
                    ]
                    value = sum(numeric_vals) if numeric_vals else None

                if value is None:
                    continue

                # Dates come back as second-level timestamps in string form
                date = pd.to_datetime(pd.to_numeric(raw_date, errors='coerce'), unit='s', errors='coerce')
                if pd.isna(date):
                    continue

                processed_data.append({
                    'date': date,
                    'value': value
                })

            if not processed_data:
                return None, None

            df = pd.DataFrame(processed_data)
            df = df.sort_values('date')

            earliest_date = df['date'].iloc[0]
            sample_df = df.head(300).copy()

            return earliest_date, sample_df
        except Exception as e:
            print(f"Error fetching {self.name}: {e}")
            return None, None


class PaidSourcePlaceholder(BaseDataSource):
    def __init__(self, name, category, endpoint, note):
        super().__init__(name, category, endpoint, note)
        self.is_paid_placeholder = True

    def fetch_survey_data(self):
        return None, None

# ==========================================
# 3. 核心逻辑：对齐与统计
# ==========================================


def analyze_missing_values(earliest_date, data_df):
    """
    创建一个从最早日期开始的 300 天连续日历（包含周末），
    将数据映射上去，计算有多少天是空的。
    """
    if earliest_date is None or data_df is None:
        return "N/A", 0

    # 标准化时间，去掉时分秒
    start_date = earliest_date.replace(
        hour=0, minute=0, second=0, microsecond=0)

    # 创建完美的 300 天日历 (7x24)
    perfect_index = pd.date_range(start=start_date, periods=300, freq='D')
    perfect_df = pd.DataFrame(index=perfect_index)

    # 处理样本数据
    data_df['date'] = data_df['date'].dt.normalize()  # 去掉时分秒
    data_df = data_df.set_index('date')

    # 合并 - 只有在 perfect_index 里的日期才算数
    # 任何在源数据里缺失的日期（包括周末对于美股数据）都会变成 NaN
    merged_df = perfect_df.join(data_df, how='left')

    # 计算缺失值
    missing_count = merged_df['value'].isna().sum()
    total_count = len(merged_df)
    missing_ratio = missing_count / total_count

    return start_date.strftime('%Y-%m-%d'), missing_ratio

# ==========================================
# 4. 主执行流程
# ==========================================


def main():
    sources = [
        # --- Macro ---
        YahooMacroSource("Dollar Index (DXY)", "DX-Y.NYB"),
        YahooMacroSource("S&P 500", "^GSPC"),
        YahooMacroSource("US 10Y Treasury", "^TNX"),
        YahooMacroSource("Volatility Index (VIX)", "^VIX"),
        YahooMacroSource("Gold Futures", "GC=F"),

        # --- On-Chain ---
        BlockchainComSource("Hash Rate", "hash-rate"),
        BlockchainComSource("Mining Difficulty", "difficulty"),
        BlockchainComSource("Mempool Size", "mempool-size"),

        # --- Sentiment ---
        AlternativeMeSource(),

        # --- Liquidity ---
        DefiLlamaStablecoinSource(),

        # --- TODO: Paid Sources (Placeholders) ---
        PaidSourcePlaceholder("Exchange Flows (Spot)", "On-Chain (Advanced)",
                              "CryptoQuant/Glassnode API", "TODO: Paid Plan Required"),
        PaidSourcePlaceholder("Whale Wallet Balance", "On-Chain (Advanced)",
                              "Glassnode API", "TODO: Tier 2 API Required"),
        PaidSourcePlaceholder("Option Gamma Exposure", "Derivatives",
                              "Amberdata/Deribit API", "TODO: Paid API Required")
    ]

    results = []
    print(f"Starting Survey for {len(sources)} Data Sources...\n")

    for src in sources:
        row = {
            "Category": src.category,
            "Data Name": src.name,
            "Endpoint": src.endpoint,
            "Params/Notes": src.params_note,
        }

        if src.is_paid_placeholder:
            print(f"[-] Skipping Paid/TODO: {src.name}")
            row.update({
                "Earliest Date": "TODO (Paid)",
                "Sample Size": "0",
                "Missing Ratio (Initial 300d)": "N/A",
                "Status": "TODO"
            })
        else:
            print(f"[+] Surveying: {src.name}...")
            earliest_date, sample_df = src.fetch_survey_data()

            if earliest_date:
                start_str, missing_ratio = analyze_missing_values(
                    earliest_date, sample_df)
                row.update({
                    "Earliest Date": start_str,
                    "Sample Size": "300 Days",
                    "Missing Ratio (Initial 300d)": f"{missing_ratio:.2%}",
                    "Status": "Active"
                })
                # 特殊备注：如果缺失率很高且是宏观数据，提示这是周末导致的
                if missing_ratio > 0.2 and "Macro" in src.category:
                    row["Params/Notes"] += " (High missing ratio due to Weekends/Holidays)"
            else:
                row.update({
                    "Earliest Date": "Fetch Failed",
                    "Sample Size": "0",
                    "Missing Ratio (Initial 300d)": "N/A",
                    "Status": "Error"
                })

        results.append(row)
        time.sleep(0.5)  # 避免触发频率限制

    # 输出 CSV
    output_file = "data_layer_survey_reportg_gemini30.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print(f"Survey Complete. Report saved to: {output_file}")
    print("="*80)
    # 打印预览
    print(df_results[['Category', 'Data Name', 'Earliest Date',
          'Missing Ratio (Initial 300d)', 'Status']].to_string())


if __name__ == "__main__":
    main()
