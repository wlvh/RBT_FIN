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

import datetime as dt
from typing import Dict, List, Mapping, Optional, Sequence

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
from .sources.local_csv import acquire_local_csv
from .validation import validate_irregular_series, validate_strict_grid_ohlcv

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


def build_registry() -> List[Dataset]:
    """
    构造 WP1 注册表。

    Returns:
        Dataset 列表，顺序稳定（行情 -> 日频代表 -> 本地样例）。
    """
    return [
        build_klines_dataset(),
        build_funding_dataset(),
        build_spot_mirror_dataset(),
        build_local_csv_dataset(
            dataset_id="local_metrics_sample",
            time_column="date",
            value_columns=("value",),
            identity={
                "source": "local_file",
                "frequency": "daily",
                "value_meaning": "user_supplied_metric",
            },
            remark="本地低频样例：证明公共流程不依赖行情或分页",
        ),
    ]


def get_dataset(dataset_id: str, *, registry: Optional[Sequence[Dataset]] = None) -> Dataset:
    """
    按 id 取数据集定义。

    Args:
        dataset_id: 数据集标识。
        registry: 可选的注册表；默认使用 `build_registry()`。

    Returns:
        Dataset。

    Raises:
        ConfigError: 标识未注册（不做模糊匹配，避免静默取错数据集）。
    """
    datasets = list(registry) if registry is not None else build_registry()
    table: Dict[str, Dataset] = {item.dataset_id: item for item in datasets}
    if dataset_id not in table:
        raise ConfigError(f"未注册的 dataset_id={dataset_id!r}；已注册: {sorted(table)}")
    return table[dataset_id]
