# -*- coding: utf-8 -*-
"""
数据集注册表。

功能或目的：
    把“来源适配器 + 校验函数 + 身份 + 参数 + 版本”组装成 `Dataset`。
    Issue #5 §3.1 要求“参数记录与真实请求从同一份有效配置生成”——因此这里的 `params`
    既是 manifest 里的记录，也是适配器真正读取的参数，不存在第二份 lambda 硬编码。

    WP1 注册三项：
        - `btcusdt_perp_1m`：首批生产承诺的分钟行情（BTCUSDT USDⓈ-M 期货成交价）。
        - `binance_funding_btcusdt_raw`：首个日频代表（原始结算记录，日均是它的派生视图）。
        - `local_metrics_sample`：本地导入样例；它与上面两项共用同一条公共流程，
          用来证明流程没有被硬编码成行情。
    WP2 在此文件继续追加其余注册项，不新增第二套框架。

调用关系：
    `rbt_fin.cli` 与 trading shim 通过 `build_registry()` / `get_dataset()` 取得数据集定义。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    REVISION_REJECT,
    CoverageReport,
    DataRequest,
    Dataset,
    GridSpec,
)
from .errors import ConfigError
from .sources.binance_funding import RAW_TIME_COLUMN, acquire_funding_records
from .sources.binance_klines import acquire_klines
from .sources.binance_reference_price import acquire_reference_price
from .sources.bybit_funding import RAW_TIME_COLUMN as BYBIT_TIME_COLUMN
from .sources.bybit_funding import acquire_bybit_funding
from .sources.daily_series import (
    CRYPTOCOMPARE_BASE_FIELDS,
    PCT_SUFFIX,
    TIME_COLUMN,
    acquire_cryptocompare_chain_pct,
    acquire_daily_series,
)
from .sources.local_csv import acquire_local_csv
from .validation import (
    validate_irregular_series,
    validate_strict_grid_ohlcv,
    validate_strict_grid_series,
)

OHLCV_COLUMNS: Sequence[str] = ("open", "high", "low", "close", "volume")


def build_klines_dataset(
    *,
    dataset_id: str = "btcusdt_perp_1m",
    market: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 1500,
    refresh_interval: dt.timedelta = dt.timedelta(minutes=1),
) -> Dataset:
    """
    构造分钟行情数据集。

    Args:
        dataset_id: 稳定标识，同时是缓存目录名。
        market: `sources.binance_klines.KLINES_ENDPOINTS` 的键。
        symbol: 交易对。
        interval: 固定长度周期。
        limit: 单页条数。
        refresh_interval: 默认重新检查来源的间隔。分钟行情的尾部会持续增长，
                          因此间隔设为一个 bar；真正决定是否联网的仍是“覆盖是否满足请求”。

    Returns:
        Dataset。
    """
    return Dataset(
        dataset_id=dataset_id,
        identity={
            "exchange": "binance",
            "market": market,
            "symbol": symbol,
            "price_type": "trade",       # 成交价，区别于 index/mark/premium 三类参考价
            "frequency": interval,
            "price_unit": "quote_usdt",
            "volume_unit": "base_asset",
            "value_meaning": "raw_ohlcv",  # 原始值，未做任何转换
        },
        params={"market": market, "symbol": symbol, "interval": interval, "limit": limit},
        acquire=acquire_klines,
        validate=validate_strict_grid_ohlcv,
        time_column="datetime",
        time_meaning="bar_open_time_utc",
        value_columns=OHLCV_COLUMNS,
        grid=GridSpec(interval=interval, origin_note="K 线开盘时间，epoch 对齐"),
        default_refresh_interval=refresh_interval,
        schema_version="ohlcv-1",
        transform_version="raw-1",
        revision_policy=REVISION_REJECT,
        requires_network=True,
        point_in_time_verified=False,
        calendar_verified=True,  # 严格网格：期望时间点完全由 interval 决定，无需外部日历
        remark="BTCUSDT USDⓈ-M 期货成交价 1m；trading 批次输入的唯一行情来源",
    )


def build_spot_mirror_dataset(
    *,
    dataset_id: str = "btcusdt_spot_1m_mirror",
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 1000,
) -> Dataset:
    """
    构造基于 Binance 官方公开只读镜像的现货分钟行情数据集。

    为什么单独注册而不是复用期货数据集的 id：
        数据身份必须区分市场与价格类型。把镜像（spot）的验证结果记到期货数据集名下，
        就等于用一次 spot 验证冒充 USDⓈ-M 期货已验证——这正是 §4.1 明令禁止的混写。
        本数据集只用于在期货端点被地域限制时对同一条代码路径做真实 HTTP 资格验证。

    Args:
        dataset_id: 稳定标识。
        symbol: 交易对。
        interval: 固定长度周期。
        limit: 单页条数（现货端点上限 1000）。

    Returns:
        Dataset。
    """
    dataset = build_klines_dataset(
        dataset_id=dataset_id, market="spot_data_mirror", symbol=symbol,
        interval=interval, limit=limit,
    )
    import dataclasses

    return dataclasses.replace(
        dataset,
        identity={
            **dict(dataset.identity),
            "market": "spot",
            "endpoint_host": "data-api.binance.vision",
        },
        remark="Binance 官方公开只读镜像的现货 1m；仅用于真实 HTTP 资格验证，不是生产行情来源",
    )


def build_funding_dataset(
    *,
    dataset_id: str = "binance_funding_btcusdt_raw",
    symbol: str = "BTCUSDT",
    limit: int = 1000,
    refresh_interval: dt.timedelta = dt.timedelta(hours=1),
) -> Dataset:
    """
    构造 Binance 资金费率原始结算记录数据集（日均视图由 transforms 派生）。

    为什么落原始记录而不是直接落日均：
        §4.4 要求“先按原始事件主键去重，再生成日均等旧视图”。把日均当作存储单位，
        分页重叠就会在聚合里被重复加权，而且一旦口径要改就必须重新下载全部历史。

    Args:
        dataset_id: 稳定标识。
        symbol: 交易对。
        limit: 单页条数。
        refresh_interval: 默认重新检查来源的间隔（结算为 8 小时一次，1 小时足够及时）。

    Returns:
        Dataset。
    """
    return Dataset(
        dataset_id=dataset_id,
        identity={
            "exchange": "binance",
            "market": "usdm_futures",
            "symbol": symbol,
            "price_type": "n/a",
            "frequency": "event_8h",
            "rate_unit": "fraction_per_interval",
            "value_meaning": "raw_settlement_record",
        },
        params={"symbol": symbol, "limit": limit},
        acquire=acquire_funding_records,
        validate=validate_irregular_series,
        time_column=RAW_TIME_COLUMN,
        time_meaning="funding_settlement_time_utc",
        value_columns=("funding_rate",),
        grid=None,  # 事件流：结算间隔历史上变更过，不能套固定网格判定完整性
        default_refresh_interval=refresh_interval,
        schema_version="funding-raw-1",
        transform_version="raw-1",
        revision_policy=REVISION_REJECT,
        requires_network=True,
        point_in_time_verified=False,
        calendar_verified=False,  # 结算日历未验证 -> 覆盖最高只能到 unverified
        remark="Binance USDⓈ-M 资金费率原始结算记录；日均视图见 transforms.daily_mean_by_utc_date",
    )


def build_local_csv_dataset(
    *,
    dataset_id: str,
    time_column: str,
    value_columns: Sequence[str],
    identity: Mapping[str, str],
    refresh_interval: dt.timedelta = dt.timedelta(days=1),
    remark: str = "",
) -> Dataset:
    """
    构造本地 CSV 导入数据集（不联网）。

    Args:
        dataset_id: 稳定标识。
        time_column: 时间列名。
        value_columns: 数值列名。
        identity: 数据身份。
        refresh_interval: 重新检查本地文件的间隔。
        remark: 说明。

    Returns:
        Dataset。
    """
    return Dataset(
        dataset_id=dataset_id,
        identity=dict(identity),
        params={"time_column": time_column, "value_columns": list(value_columns)},
        acquire=acquire_local_csv,
        validate=validate_irregular_series,
        time_column=time_column,
        time_meaning="local_import_observation_time_utc",
        value_columns=tuple(value_columns),
        grid=None,
        default_refresh_interval=refresh_interval,
        schema_version="local-csv-1",
        transform_version="raw-1",
        revision_policy=REVISION_REJECT,
        requires_network=False,
        point_in_time_verified=False,
        calendar_verified=False,
        remark=remark or "本地 CSV 导入；不联网、不修改用户原文件",
    )


def build_reference_price_dataset(
    *, dataset_id: str, market: str, symbol: str, value_column: str, require_positive: bool,
    remark: str, interval: str = "1d", limit: int = 1500,
) -> Dataset:
    """
    构造 Binance 参考价格日频数据集（指数价 / 标记价 / 溢价指数）。

    Args:
        dataset_id: 稳定标识。
        market: `sources.binance_klines.KLINES_ENDPOINTS` 的键。
        symbol: 标的（指数价端点用 `pair`，由适配器按 market 自动选参数名）。
        value_column: 输出列名，按价格类型命名以免三类价格被混用。
        require_positive: 是否要求为正。溢价指数可以为负，必须传 False。
        remark: 说明。
        interval: 周期。
        limit: 单页条数。

    Returns:
        Dataset。
    """
    return Dataset(
        dataset_id=dataset_id,
        identity={
            "exchange": "binance", "market": "usdm_futures", "symbol": symbol,
            "price_type": value_column.replace("_close", ""), "frequency": interval,
            "price_unit": "quote_usdt", "value_meaning": "reference_price_close",
        },
        params={
            "market": market, "symbol": symbol, "interval": interval, "limit": limit,
            "value_column": value_column, "require_positive": require_positive,
        },
        acquire=acquire_reference_price,
        validate=validate_strict_grid_series,
        time_column="datetime",
        time_meaning="bar_open_time_utc",
        value_columns=(value_column,),
        grid=GridSpec(interval=interval, origin_note="参考价格 K 线开盘时间，epoch 对齐"),
        default_refresh_interval=dt.timedelta(hours=6),
        schema_version="reference-price-1",
        transform_version="close-only-1",
        revision_policy=REVISION_REJECT,
        calendar_verified=True,
        remark=remark,
    )


def build_bybit_funding_dataset(
    *, dataset_id: str = "bybit_funding_btcusdt_raw", symbol: str = "BTCUSDT",
    category: str = "linear", limit: int = 200,
) -> Dataset:
    """构造 Bybit 资金费率原始结算记录数据集（日均是它的派生视图）。"""
    return Dataset(
        dataset_id=dataset_id,
        identity={
            "exchange": "bybit", "market": category, "symbol": symbol,
            "price_type": "n/a", "frequency": "event_8h",
            "rate_unit": "fraction_per_interval", "value_meaning": "raw_settlement_record",
        },
        params={"symbol": symbol, "category": category, "limit": limit},
        acquire=acquire_bybit_funding,
        validate=validate_irregular_series,
        time_column=BYBIT_TIME_COLUMN,
        time_meaning="funding_settlement_time_utc",
        value_columns=("funding_rate",),
        grid=None,
        default_refresh_interval=dt.timedelta(hours=1),
        schema_version="funding-raw-1",
        transform_version="raw-1",
        revision_policy=REVISION_REJECT,
        calendar_verified=False,
        remark="Bybit 资金费率原始结算记录；首页不传 startTime、只带固定 endTime 向历史翻页",
    )


def build_daily_series_dataset(
    *, dataset_id: str, url: str, query: Mapping[str, Any], extractor: str,
    value_columns: Sequence[str], identity: Mapping[str, Any], remark: str,
    refresh_interval: dt.timedelta = dt.timedelta(hours=12),
    credential_env: Optional[str] = None, credential_style: Optional[str] = None,
    credential_query_key: Optional[str] = None, optional_dependency: Optional[str] = None,
) -> Dataset:
    """
    构造单次请求型日频序列数据集。

    Args:
        dataset_id: 稳定标识。
        url: 端点 URL。
        query: 非凭据查询参数；它同时是 manifest 记录与真实请求参数，不存在第二份。
        extractor: `sources.daily_series.EXTRACTORS` 的键。
        value_columns: 输出列名。
        identity: 数据身份。
        remark: 说明。
        refresh_interval: 默认重新检查来源的间隔。
        credential_env: 需要的凭据环境变量名（只存名字，值永不进入配置/记录/日志）。
        credential_style: 'query' 或 'header'。
        credential_query_key: `credential_style='query'` 时的参数名。
        optional_dependency: 可选依赖名。

    Returns:
        Dataset。
    """
    return Dataset(
        dataset_id=dataset_id,
        identity=dict(identity),
        params={
            "url": url, "query": dict(query), "extractor": extractor,
            "credential_env": credential_env, "credential_style": credential_style,
            "credential_query_key": credential_query_key,
        },
        acquire=acquire_daily_series,
        validate=validate_irregular_series,
        time_column=TIME_COLUMN,
        time_meaning="observation_date_utc",
        value_columns=tuple(value_columns),
        grid=None,
        default_refresh_interval=refresh_interval,
        schema_version="daily-series-1",
        transform_version="raw-1",
        revision_policy=REVISION_REJECT,
        # 这些来源都有真实的非发布日（周末/假期/停更），来源日历未经验证 ->
        # 覆盖最高只能到 unverified，要求严格证明完整的调用必须被拒绝。
        calendar_verified=False,
        optional_dependency=optional_dependency,
        remark=remark,
    )


# Blockchain.com 的六个链上指标：共用同一个采集函数，按 chart_id 参数化。
BLOCKCHAIN_CHARTS: Tuple[str, ...] = (
    "hash-rate", "difficulty", "mempool-size",
    "n-transactions", "transaction-fees", "n-unique-addresses",
)

# FRED：七个注册名对应五个不同序列。DGS10 与 DTWEXBGS 各有两个注册名。
# Why 用别名而不是两份数据集：同一序列同一参数必须共用基础数据，否则会重复下载
# 并产生两份可以互相矛盾的缓存（Issue #5 §5）。
# 每项为 `(series_id, 基础数据集 id, 说明)`。基础 id 直接沿用基线已有的注册名，
# 不另造新名字——基线的 25 个 dataset_id 必须原样保留其概览用途。
FRED_SERIES: Tuple[Tuple[str, str, str], ...] = (
    ("DTWEXBGS", "macro_dxy", "广义美元指数"),
    ("SP500", "macro_spx", "标普500"),
    ("DGS10", "macro_us10y", "美国10Y国债收益率"),
    ("VIXCLS", "macro_vix", "VIX"),
    ("T10Y2Y", "fred_t10y2y", "美债 10Y-2Y 利差"),
)

# 注册名 -> 基础数据集 id。
# Why 需要别名：基线里 DGS10 与 DTWEXBGS 各有两个注册名，资金费率也各有一个不带 `_raw`
# 后缀的历史名。别名解析到**同一个 Dataset**，因此共用同一个缓存目录——
# 第二个名字的请求必然是零网络的缓存命中，不会重复下载（Issue #5 §9）。
DATASET_ALIASES: Mapping[str, str] = {
    "fred_dtwexbgs": "macro_dxy",
    "fred_dgs10": "macro_us10y",
    "binance_funding_btcusdt": "binance_funding_btcusdt_raw",
    "bybit_funding_btcusdt": "bybit_funding_btcusdt_raw",
}


def build_registry() -> List[Dataset]:
    """
    构造全部**基础**数据集（别名不在此列，见 `DATASET_ALIASES`）。

    Returns:
        Dataset 列表，顺序稳定。
    """
    datasets: List[Dataset] = [
        build_klines_dataset(),
        build_funding_dataset(),
        build_bybit_funding_dataset(),
        build_spot_mirror_dataset(),
    ]

    # Binance 三类参考价格。溢价指数可以为负，因此单独声明 require_positive=False。
    for dataset_id, market, symbol, column, positive, note in (
        ("binance_index_price_1d", "usdm_index_price", "BTCUSDT", "index_close", True, "指数价 close"),
        ("binance_mark_price_1d", "usdm_mark_price", "BTCUSDT", "mark_close", True, "标记价 close"),
        ("binance_premium_index_1d", "usdm_premium_index", "BTCUSDT", "premium_close", False,
         "溢价指数 close（可为负，不按价格必须为正校验）"),
    ):
        datasets.append(build_reference_price_dataset(
            dataset_id=dataset_id, market=market, symbol=symbol, value_column=column,
            require_positive=positive, remark=note,
        ))

    for chart_id in BLOCKCHAIN_CHARTS:
        datasets.append(build_daily_series_dataset(
            dataset_id=f"blockchain_{chart_id.replace('-', '_')}",
            url=f"https://api.blockchain.info/charts/{chart_id}",
            query={"timespan": "all", "format": "json"},
            extractor="blockchain_chart", value_columns=("value",),
            identity={"source": "blockchain.com", "chart_id": chart_id,
                      "frequency": "daily", "value_meaning": "raw_metric"},
            remark=f"Blockchain.com Charts {chart_id}；早期链上数据稀疏",
        ))

    datasets.append(build_daily_series_dataset(
        dataset_id="altme_fng", url="https://api.alternative.me/fng/",
        query={"limit": 0, "format": "json"}, extractor="fear_greed", value_columns=("index",),
        identity={"source": "alternative.me", "frequency": "daily", "value_meaning": "sentiment_index"},
        remark="Fear&Greed 全历史（limit=0 表示全部）",
    ))
    datasets.append(build_daily_series_dataset(
        dataset_id="wikipedia_bitcoin_pageviews",
        url=("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia.org/all-access/all-agents/Bitcoin/daily/20150701/20991231"),
        query={}, extractor="wikipedia_pageviews", value_columns=("views",),
        identity={"source": "wikimedia", "article": "Bitcoin", "project": "en.wikipedia.org",
                  "frequency": "daily", "value_meaning": "pageviews"},
        remark="en.wikipedia Bitcoin 条目日 PV（2015-07-01 起）",
    ))
    datasets.append(build_daily_series_dataset(
        dataset_id="defillama_stablecoins_total",
        url="https://stablecoins.llama.fi/stablecoincharts/all", query={},
        extractor="defillama_stablecoins", value_columns=("total_circulating_usd",),
        identity={"source": "defillama", "frequency": "daily", "unit": "usd",
                  "value_meaning": "total_circulating_pegged_usd"},
        remark="全稳定币流通总量；只取 totalCirculatingUSD.peggedUSD，不跨锚定资产求和",
    ))

    for series_id, base_id, desc in FRED_SERIES:
        datasets.append(build_daily_series_dataset(
            dataset_id=base_id,
            url="https://api.stlouisfed.org/fred/series/observations",
            query={"series_id": series_id, "file_type": "json"},
            extractor="fred_observations", value_columns=("value",),
            identity={"source": "fred", "series_id": series_id, "frequency": "business_daily",
                      "value_meaning": "latest_vintage_observation"},
            remark=f"{desc} {series_id} | 需要 FRED_API_KEY | v1 不建 vintage 系统",
            credential_env="FRED_API_KEY", credential_style="query", credential_query_key="api_key",
        ))

    datasets.append(build_daily_series_dataset(
        dataset_id="macro_gold", url="https://stooq.pl/q/d/l/",
        query={"s": "xauusd", "i": "d"}, extractor="stooq_csv", value_columns=("value",),
        identity={"source": "stooq", "ticker": "xauusd", "frequency": "business_daily",
                  "value_meaning": "close"},
        remark="Stooq XAUUSD 收盘价，无需 key",
    ))
    datasets.append(build_daily_series_dataset(
        dataset_id="glassnode_active_addresses_btc",
        url="https://api.glassnode.com/v1/metrics/addresses/active_count",
        query={"a": "BTC", "i": "24h"}, extractor="glassnode_metric", value_columns=("value",),
        identity={"source": "glassnode", "metric": "addresses/active_count", "asset": "BTC",
                  "frequency": "daily", "value_meaning": "raw_metric"},
        remark="需要 GLASSNODE_API_KEY",
        credential_env="GLASSNODE_API_KEY", credential_style="query", credential_query_key="api_key",
    ))
    # 保留基线注册名与"环比"用途，但按 §4.4 同时保留基础值，并把
    # 首次记录 / 缺失前值 / 零分母如实标记为未定义，而不是统一填成零增长。
    # 口径变化已通过 transform_version 记录，不静默换口径。
    cryptocompare = build_daily_series_dataset(
        dataset_id="cryptocompare_btc_chain_pct",
        url="https://min-api.cryptocompare.com/data/blockchain/histo/day",
        query={"fsym": "BTC", "limit": 2000}, extractor="cryptocompare_chain",
        value_columns=tuple(CRYPTOCOMPARE_BASE_FIELDS)
        + tuple(f"{name}{PCT_SUFFIX}" for name in CRYPTOCOMPARE_BASE_FIELDS),
        identity={"source": "cryptocompare", "asset": "BTC", "frequency": "daily",
                  "value_meaning": "base_metric_and_pct_change"},
        remark=("链上核心字段的基础值 + 环比。相对基线的口径修正："
                "首次记录/缺失前值/零分母不再统一填 0.0，而是标记为未定义（NaN），"
                "并保留基础值列。凭据只来自 CRYPTOCOMPARE_API_KEY，无内置备用 key"),
        credential_env="CRYPTOCOMPARE_API_KEY", credential_style="header",
    )
    datasets.append(dataclasses.replace(
        cryptocompare, acquire=acquire_cryptocompare_chain_pct, transform_version="pct-keep-base-2",
        # 环比列在首期 / 缺失前值 / 零分母处没有定义，NaN 就是正确答案；
        # 基础值列不在此列，它们的空值仍然是必须 Fail Fast 的字段缺失。
        nullable_columns=tuple(f"{name}{PCT_SUFFIX}" for name in CRYPTOCOMPARE_BASE_FIELDS),
    ))

    datasets.append(build_local_csv_dataset(
        dataset_id="binance_metrics_local", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "symbol": "BTCUSDT", "frequency": "daily",
                  "value_meaning": "user_supplied_metric"},
        remark="本地 metrics CSV；不联网、不修改用户原文件",
    ))
    datasets.append(build_local_csv_dataset(
        dataset_id="local_metrics_sample", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "frequency": "daily",
                  "value_meaning": "user_supplied_metric"},
        remark="本地低频样例：证明公共流程不依赖行情或分页",
    ))
    return datasets


def list_registered_ids(*, registry: Optional[Sequence[Dataset]] = None) -> List[str]:
    """
    列出全部对外注册标识（基础数据集 + 别名），顺序稳定。

    Why 别名也要列出：基线的 25 个 `dataset_id` 都必须有明确去向；
    概览要能逐项回答它们的状态，而不是只显示合并后的基础数据集。
    """
    datasets = list(registry) if registry is not None else build_registry()
    return sorted([item.dataset_id for item in datasets] + list(DATASET_ALIASES))


def get_dataset(dataset_id: str, *, registry: Optional[Sequence[Dataset]] = None) -> Dataset:
    """
    按 id 取数据集定义；别名解析到同一个基础数据集。

    Args:
        dataset_id: 数据集标识或别名。
        registry: 可选的注册表；默认使用 `build_registry()`。

    Returns:
        Dataset。别名与基础 id 返回**同一个**定义，因此共用同一个缓存目录，
        第二次请求必然是零网络的缓存命中。

    Raises:
        ConfigError: 标识未注册（不做模糊匹配，避免静默取错数据集）。
    """
    datasets = list(registry) if registry is not None else build_registry()
    table: Dict[str, Dataset] = {item.dataset_id: item for item in datasets}
    resolved = DATASET_ALIASES.get(dataset_id, dataset_id)
    if resolved not in table:
        raise ConfigError(
            f"未注册的 dataset_id={dataset_id!r}；已注册: {list_registered_ids(registry=datasets)}"
        )
    return table[resolved]
