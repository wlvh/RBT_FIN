# -*- coding: utf-8 -*-
"""
UTC 时间、周期边界与分钟网格工具。

功能或目的：
    Issue #5 §4.2 要求：真正时间点使用带时区 UTC；naive 输入在接口处拒绝；
    复用 trading 的“已完成周期边界”；时间戳必须精确落在网格上（秒、微秒、纳秒都检查）。
    本模块是整个数据层唯一的时间语义真相源，禁止各来源适配器自行解析/对齐。

调用关系：
    - `rbt_fin.contracts` 用 `require_utc` 校验请求区间。
    - `rbt_fin.validation` 用 `expected_grid` / `assert_on_grid` 做严格网格校验。
    - `rbt_fin.sources.*` 用 `parse_interval` / `to_epoch_ms` 构造端点参数。
    - 调用方（CLI 或 trading shim）用 `complete_end_exclusive` 固定一次任务截止时刻。
"""

from __future__ import annotations

import datetime as dt
from typing import Union

import pandas as pd

from .errors import ConfigError, TimeContractError

UTC = dt.timezone.utc

# 为什么显式列出：周线/月线的锚点由交易所定义（例如周一 00:00 / 自然月），
# 直接按 epoch 取整会错锚；在没有专门实现之前必须拒绝，而不是返回“看似合理”的时间。
_EPOCH_FLOOR_UNSAFE_SUFFIXES = ("w", "M")

TimeLike = Union[dt.datetime, pd.Timestamp]


def require_utc(value: TimeLike, *, arg_name: str) -> pd.Timestamp:
    """
    校验并规范化一个时间点为 tz-aware UTC 的 `pd.Timestamp`。

    为什么：
        naive datetime 在 `.timestamp()` / `tz_localize` 时会被按本地时区解释，
        造成静默时间漂移；公共接口必须在入口 Fail Fast（Issue #5 §4.2）。

    Args:
        value: 待校验时间；必须是 tz-aware 的 `datetime` 或 `pd.Timestamp`。
        arg_name: 参数名，用于构造可读报错。

    Returns:
        转换到 UTC 的 `pd.Timestamp`（tz-aware）。

    Raises:
        ConfigError: 类型错误或缺少时区信息。
    """
    if not isinstance(value, (dt.datetime, pd.Timestamp)):
        raise ConfigError(f"{arg_name} 必须是 datetime/Timestamp，实际为: {type(value)!r}")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ConfigError(
            f"{arg_name} 必须是 tz-aware（UTC 语义）时间；收到 naive 值 {value!r}。"
            "无时区输入只允许在有明确 legacy 合同的导入边界解释。"
        )
    return ts.tz_convert("UTC")


def parse_interval(interval: str) -> pd.Timedelta:
    """
    将周期字符串解析为固定长度 `pd.Timedelta`，并拒绝无法按 epoch 对齐的周期。

    Args:
        interval: 例如 '1m'、'15m'、'1h'、'1d'；'1w'/'1M' 会被拒绝。

    Returns:
        对应的固定长度 `pd.Timedelta`（> 0）。

    Raises:
        ConfigError: 类型/取值非法，或属于需要交易所锚点的周期。
    """
    if not isinstance(interval, str):
        raise ConfigError(f"interval 必须是 str，实际为: {type(interval)!r}")
    cleaned = interval.strip()
    if not cleaned:
        raise ConfigError("interval 不能为空。")
    # 为什么在解析之前拦截：pandas 能解析 '1W'，但它的锚点语义与交易所周线不同。
    if cleaned.lower().endswith(_EPOCH_FLOOR_UNSAFE_SUFFIXES[0]) or cleaned.endswith(_EPOCH_FLOOR_UNSAFE_SUFFIXES[1]):
        raise ConfigError(
            f"interval={interval!r} 依赖交易所锚点，不支持 epoch floor 对齐；"
            "需要周/月周期时必须单独实现锚点逻辑。"
        )
    try:
        delta = pd.Timedelta(cleaned)
    except ValueError as exc:
        raise ConfigError(f"不支持的 interval={interval!r}（pd.Timedelta 无法解析）。") from exc
    if delta <= pd.Timedelta(0):
        raise ConfigError(f"interval={interval!r} 解析为非正区间：{delta!r}")
    return delta


def floor_to_interval(value: TimeLike, *, interval: str, arg_name: str = "value") -> pd.Timestamp:
    """
    把时间点按 epoch 向下取整到周期起点。

    为什么以 epoch 为基准：'3m'/'15m' 等非 1 分钟整数倍对齐只有以 epoch 取整才正确。

    Args:
        value: tz-aware 时间点。
        interval: 固定长度周期。
        arg_name: 报错用参数名。

    Returns:
        周期起点（tz-aware UTC）。
    """
    ts = require_utc(value, arg_name=arg_name)
    interval_ns = int(parse_interval(interval).value)
    return pd.Timestamp((int(ts.value) // interval_ns) * interval_ns, tz="UTC")


def complete_end_exclusive(*, now_utc: TimeLike, interval: str) -> pd.Timestamp:
    """
    计算“只包含已收盘周期”的右开边界。

    为什么：
        直接用 now 作为截止会把正在形成的 bar 写进正式数据，导致尾部随运行时刻变化、
        回测不可复现。当前未完成 bar 的开盘时间正好是右开边界（本库统一使用
        `[start, end_exclusive)`；端点适配器各自转换为自己的包含/排除语义）。

    Args:
        now_utc: 本次任务固定的当前时间（tz-aware）。
        interval: 固定长度周期。

    Returns:
        右开边界（tz-aware UTC），等于当前未完成 bar 的开盘时间。
    """
    return floor_to_interval(now_utc, interval=interval, arg_name="now_utc")


def expected_grid(*, start: TimeLike, end_exclusive: TimeLike, interval: str) -> pd.DatetimeIndex:
    """
    构造 `[start, end_exclusive)` 上的期望网格。

    Args:
        start: 区间左闭端点，必须精确落在网格上。
        end_exclusive: 区间右开端点，必须精确落在网格上。
        interval: 固定长度周期。

    Returns:
        期望时间戳序列（tz-aware UTC，严格升序，可能为空）。

    Raises:
        ConfigError: 端点未落在网格上，或 start > end_exclusive。
    """
    start_ts = require_utc(start, arg_name="start")
    end_ts = require_utc(end_exclusive, arg_name="end_exclusive")
    if start_ts > end_ts:
        raise ConfigError(f"start 不能晚于 end_exclusive：{start_ts} > {end_ts}")
    delta = parse_interval(interval)
    interval_ns = int(delta.value)
    for label, ts in (("start", start_ts), ("end_exclusive", end_ts)):
        if int(ts.value) % interval_ns != 0:
            raise ConfigError(f"{label}={ts} 未落在 interval={interval} 的网格上。")
    if start_ts == end_ts:
        return pd.DatetimeIndex([], tz="UTC", name="datetime")
    return pd.date_range(start=start_ts, end=end_ts - delta, freq=delta, tz="UTC", name="datetime")


def assert_on_grid(index: pd.DatetimeIndex, *, interval: str, context: str) -> None:
    """
    断言索引上的每个时间戳都精确落在周期网格上。

    为什么检查纳秒而不是只检查秒：
        Issue #5 §4.2 明确要求“秒、微秒、纳秒都检查”。旧实现只检查 `.second == 0`，
        `00:00:00.500000` 这类偏移会被放行（本库反例 A）。以纳秒取模可一次覆盖三级精度。

    Args:
        index: 待检查的时间索引；必须是 tz-aware UTC 的 `DatetimeIndex`。
        interval: 固定长度周期。
        context: 报错上下文（数据集/来源标识）。

    Raises:
        TimeContractError: 存在未对齐时间戳，或索引不是 tz-aware UTC。
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TimeContractError(f"{context}: 期望 DatetimeIndex，实际为 {type(index)!r}")
    if index.tz is None:
        raise TimeContractError(f"{context}: 时间索引必须带 UTC 时区，收到 naive 索引。")
    if len(index) == 0:
        return
    interval_ns = int(parse_interval(interval).value)
    # 为什么必须先 as_unit("ns")：pandas 2.0 起 DatetimeIndex 可以是 s/ms/us 分辨率，
    # 此时 `asi8` 返回的是该分辨率下的整数（例如秒），直接与纳秒 interval 取模会得到
    # “整点时间戳不在网格上”的错误结论。统一到纳秒后，取模一次覆盖秒/微秒/纳秒三级偏移。
    offsets = index.as_unit("ns").asi8 % interval_ns
    bad = offsets != 0
    if bool(bad.any()):
        sample = [str(ts) for ts in index[bad][:5]]
        raise TimeContractError(
            f"{context}: {int(bad.sum())} 个时间戳未落在 interval={interval} 网格上，示例: {sample}"
        )


def assert_strictly_increasing_unique(index: pd.DatetimeIndex, *, context: str) -> None:
    """
    断言时间索引严格升序且无重复。

    Args:
        index: 待检查索引。
        context: 报错上下文。

    Raises:
        TimeContractError: 存在重复或非严格升序。
    """
    if len(index) < 2:
        return
    if not index.is_unique:
        duplicated = index[index.duplicated(keep="first")]
        raise TimeContractError(
            f"{context}: 时间主键存在 {len(duplicated)} 条重复，示例: {[str(t) for t in duplicated[:5]]}"
        )
    if not index.is_monotonic_increasing:
        raise TimeContractError(f"{context}: 时间索引不是严格升序。")


def to_epoch_ms(value: TimeLike, *, arg_name: str) -> int:
    """
    将 tz-aware 时间转换为毫秒时间戳（交易所端点通用参数形式）。

    Args:
        value: tz-aware 时间点。
        arg_name: 报错用参数名。

    Returns:
        整数毫秒时间戳。

    Raises:
        ConfigError: 时间点带有亚毫秒精度（端点无法表达，必须显式处理）。
    """
    ts = require_utc(value, arg_name=arg_name)
    if int(ts.value) % 1_000_000 != 0:
        raise ConfigError(f"{arg_name}={ts} 含亚毫秒精度，无法无损转换为毫秒端点参数。")
    return int(ts.value) // 1_000_000


def from_epoch_ms(value: int, *, arg_name: str) -> pd.Timestamp:
    """
    将毫秒时间戳转换为 tz-aware UTC `pd.Timestamp`。

    Args:
        value: 毫秒时间戳；必须是整数类型（含 numpy 整数），拒绝 float 以免精度丢失。
        arg_name: 报错用参数名。

    Returns:
        tz-aware UTC 时间戳。

    Raises:
        ConfigError: 类型非整数。
    """
    if isinstance(value, bool) or not isinstance(value, (int,)):
        raise ConfigError(f"{arg_name} 必须是整数毫秒时间戳，实际为: {type(value)!r} -> {value!r}")
    return pd.Timestamp(int(value) * 1_000_000, tz="UTC")


def format_utc_naive(index: pd.DatetimeIndex) -> pd.Index:
    """
    将 tz-aware UTC 索引格式化为 legacy CSV 的无时区秒精度文本。

    为什么单独提供：
        trading 的 CSV 合同是“UTC 语义的无时区 `%Y-%m-%d %H:%M:%S` 文本”
        （trading#105 §2.2）。格式化只允许发生在导出边界，库内部一律保持 tz-aware。

    Args:
        index: tz-aware UTC 的 DatetimeIndex。

    Returns:
        字符串索引。

    Raises:
        TimeContractError: 索引不是 tz-aware UTC。
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TimeContractError("format_utc_naive 需要 tz-aware 的 DatetimeIndex。")
    return index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
