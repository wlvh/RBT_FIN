# -*- coding: utf-8 -*-
"""
少量显式转换函数（不是通用特征框架）。

功能或目的：
    Issue #5 §4.4 要求“保留原始结算时间/基础值后再执行少量显式转换函数”。
    这些函数只做三件事，并且都以原始时间为准、不依赖已经被截成日期后的偶然排列：
        - 按 UTC 日期取日均（资金费率的旧日频视图）。
        - 按 UTC 日期取日内最后一条（各价格类型的日频视图）。
        - 计算环比变化率，且保留基础值、把首次记录/缺失前值/零分母区分开。

调用关系：
    `rbt_fin.registry` 在数据集上声明“基础数据 + 转换函数”；旧输出视图复用基础数据，
    避免为同一份原始数据反复下载。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ConfigError, TimeContractError

# 环比转换中“无法定义”的标记：首次记录、缺失前值、零分母都归入这一类，
# 绝不统一填成 0.0 假装“零增长”（Issue #5 §4.4）。
UNDEFINED_PCT = float("nan")


def _require_sorted_utc_index(frame: pd.DataFrame, *, context: str) -> pd.DatetimeIndex:
    """
    校验并返回 tz-aware UTC 且严格升序的时间索引。

    为什么每个转换都先校验：
        “日内 last” 之类的聚合如果依赖调用方恰好传进来的顺序，就会在换一个上游实现后
        静默换口径。转换函数必须自己保证按原始时间排序。
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TimeContractError(f"{context}: 需要 tz-aware 的 UTC DatetimeIndex。")
    if not index.is_monotonic_increasing:
        raise TimeContractError(f"{context}: 转换要求输入按原始时间升序；请先排序再转换。")
    return index


def daily_mean_by_utc_date(
    frame: pd.DataFrame, *, value_columns, date_column: str = "date", context: str
) -> pd.DataFrame:
    """
    按 UTC 日期取均值（例如 8 小时资金费率 -> 日均）。

    前置条件：
        输入必须已按原始事件主键去重。分页重叠产生的重复记录若参与均值，
        会让日均被同一条结算重复加权（Issue #5 §4.4 明确要求“先去重、后聚合”）。

    Args:
        frame: 索引为原始事件时间（tz-aware UTC）的数据帧。
        value_columns: 参与聚合的列名序列。
        date_column: 输出日期列名。
        context: 报错上下文。

    Returns:
        以 UTC 日期（tz-aware，零点）为索引的数据帧。

    Raises:
        TimeContractError: 输入时间索引非法、未排序，或存在重复主键。
    """
    index = _require_sorted_utc_index(frame, context=context)
    if not index.is_unique:
        raise TimeContractError(
            f"{context}: 输入存在重复事件主键，必须先按原始主键去重再聚合。"
        )
    missing = [col for col in value_columns if col not in frame.columns]
    if missing:
        raise ConfigError(f"{context}: 缺少聚合列 {missing}。")
    dates = index.normalize()
    grouped = frame[list(value_columns)].groupby(dates).mean()
    grouped.index.name = date_column
    return grouped


def daily_last_by_utc_date(
    frame: pd.DataFrame, *, value_columns, date_column: str = "date", context: str
) -> pd.DataFrame:
    """
    按 UTC 日期取日内最后一条记录。

    为什么显式按原始时间排序后再取 last：
        旧实现先把时间截成日期再 `groupby().last()`，结果取决于行在 DataFrame 中的
        偶然顺序；乱序输入会得到“当天任意一条”而不是“当天最后一条”。

    Args:
        frame: 索引为原始事件时间（tz-aware UTC）的数据帧。
        value_columns: 需要保留的列名序列。
        date_column: 输出日期列名。
        context: 报错上下文。

    Returns:
        以 UTC 日期为索引的数据帧。
    """
    index = _require_sorted_utc_index(frame, context=context)
    missing = [col for col in value_columns if col not in frame.columns]
    if missing:
        raise ConfigError(f"{context}: 缺少列 {missing}。")
    dates = index.normalize()
    grouped = frame[list(value_columns)].groupby(dates).last()
    grouped.index.name = date_column
    return grouped


def pct_change_keep_base(
    frame: pd.DataFrame, *, value_columns, context: str, suffix: str = "_pct"
) -> pd.DataFrame:
    """
    计算环比变化率，同时保留基础值，并显式区分“无法定义”的情形。

    无法定义的三类（统一输出 NaN，绝不填 0.0）：
        - 首次记录：没有前值。
        - 前值缺失：上一期该字段为空。
        - 零分母：前值为 0，变化率没有定义。

    Args:
        frame: 索引为时间的数据帧。
        value_columns: 需要计算环比的列。
        context: 报错上下文。
        suffix: 环比列名后缀。

    Returns:
        同时包含基础值列与 `<col><suffix>` 环比列的数据帧。
    """
    _require_sorted_utc_index(frame, context=context)
    missing = [col for col in value_columns if col not in frame.columns]
    if missing:
        raise ConfigError(f"{context}: 缺少列 {missing}。")
    out = frame.copy()
    for column in value_columns:
        values = out[column].to_numpy(dtype="float64")
        previous = np.empty_like(values)
        previous[0] = np.nan
        previous[1:] = values[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (values - previous) / previous
        # 零分母与非有限结果一律标记为未定义，保留“这里没有可解释的增长率”这一事实。
        ratio = np.where(previous == 0.0, UNDEFINED_PCT, ratio)
        ratio = np.where(np.isfinite(ratio), ratio, UNDEFINED_PCT)
        out[f"{column}{suffix}"] = ratio
    return out
