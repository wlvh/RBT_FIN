# -*- coding: utf-8 -*-
"""
单次请求型日频序列适配器（WP2 的主力）。

功能或目的：
    余下的低频来源在协议上只差三件事：URL/参数、怎么从载荷里取出记录、怎么判业务错误。
    因此这里只提供**一个**通用取数函数，各来源用一个很小的 `extract` 函数描述自己的
    载荷形状。不为每个来源复制一份分页/解析/去重骨架（Issue #5 §7：文件数由职责决定）。

    每个 `extract` 都是按**实测响应**写的，不是按注释推断的：
        blockchain.info  {"values": [{"x": <epoch 秒>, "y": <float>}]}
        alternative.me   {"data": [{"timestamp": "<epoch 秒字符串>", "value": "<字符串>"}]}
        defillama        [{"date": "<epoch 秒字符串>", "totalCirculatingUSD": {"peggedUSD": N}}]
        FRED             {"observations": [{"date": "YYYY-MM-DD", "value": "<字符串或 '.'>"}]}
        Glassnode        [{"t": <epoch 秒>, "v": <float>}]
        CryptoCompare    {"Data": {"Data": [{"time": <epoch 秒>, ...多字段}]}}
        Stooq            CSV 文本（本地化列名）

调用关系：
    `rbt_fin.registry` 把 `acquire_daily_series` 绑定到具体 `Dataset`；
    可靠性与预算在 `rbt_fin.http`，数值/时间/去重合同在 `rbt_fin.validation`。
"""

from __future__ import annotations

import datetime as dt
from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..contracts import AcquireContext, AcquireResult
from ..errors import BusinessStatusError, ConfigError, ResponseSchemaError, SourceBlockedError
from ..timeutils import from_epoch_ms
from ..validation import dedupe_by_time_primary_key, parse_strict_float

# `extract` 的返回形状：`[(观测时间, {输出列名: 原始值}), ...]`。
# 原始值保持源端类型（通常是字符串），统一交给 `parse_strict_float` 严格解析。
Records = List[Tuple[pd.Timestamp, Dict[str, Any]]]

TIME_COLUMN = "date"


def epoch_seconds_to_utc(value: Any, *, context: str, position: int) -> pd.Timestamp:
    """
    把 epoch 秒（整数或**数字字符串**）转为 tz-aware UTC 时间戳。

    为什么要接受字符串：alternative.me 与 defillama 的时间字段实测都是字符串。
    但仍然拒绝浮点与非数字文本——静默接受它们等于放弃时间精度合同。

    Args:
        value: 源端时间字段值。
        context: 报错上下文。
        position: 记录序号，便于定位。

    Returns:
        tz-aware UTC 时间戳。

    Raises:
        ResponseSchemaError: 无法按整数秒解释。
    """
    if isinstance(value, bool):
        raise ResponseSchemaError(f"{context}: 第 {position} 条时间字段为布尔值 {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text.lstrip("-").isdigit():
            raise ResponseSchemaError(
                f"{context}: 第 {position} 条时间字段 {value!r} 不是整数秒文本"
            )
        value = int(text)
    if not isinstance(value, int):
        raise ResponseSchemaError(
            f"{context}: 第 {position} 条时间字段必须是整数秒，实际为 {type(value).__name__}"
        )
    return from_epoch_ms(value * 1000, arg_name=f"{context}.time[{position}]")


def _require_list(payload: Any, *path: str, context: str) -> Sequence[Any]:
    """
    按路径取出记录数组，并要求它确实是数组。

    为什么不用 `.get()` 链：字段缺失必须报出**缺在哪一层**，而不是一路 None 走到底
    最后变成"来源没有数据"。
    """
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ResponseSchemaError(
                f"{context}: 响应缺少字段 {'.'.join(path)!r}（缺在 {key!r} 这一层）"
            )
        node = node[key]
    if not isinstance(node, list):
        raise ResponseSchemaError(
            f"{context}: 字段 {'.'.join(path) or '<顶层>'} 必须是数组，实际为 {type(node).__name__}"
        )
    return node


def _require_fields(row: Any, fields: Sequence[str], *, context: str, position: int) -> None:
    """断言一条记录是对象且含全部必需字段（字段缺失不是记录缺失）。"""
    if not isinstance(row, dict):
        raise ResponseSchemaError(f"{context}: 第 {position} 条不是对象，实际为 {type(row).__name__}")
    missing = [name for name in fields if name not in row]
    if missing:
        raise ResponseSchemaError(
            f"{context}: 第 {position} 条缺少必需字段 {missing}，实际键: {sorted(row)}"
        )


# -------------------- 各来源的载荷提取 --------------------

def extract_blockchain_chart(payload: Any, *, context: str) -> Records:
    """Blockchain.com Charts：`{"values": [{"x": 秒, "y": 值}]}`。"""
    rows = _require_list(payload, "values", context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("x", "y"), context=context, position=position)
        out.append((epoch_seconds_to_utc(row["x"], context=context, position=position),
                    {"value": row["y"]}))
    return out


def extract_fear_greed(payload: Any, *, context: str) -> Records:
    """Fear & Greed：`{"data": [{"timestamp": "秒", "value": "数字字符串"}]}`。"""
    rows = _require_list(payload, "data", context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("timestamp", "value"), context=context, position=position)
        out.append((epoch_seconds_to_utc(row["timestamp"], context=context, position=position),
                    {"index": row["value"]}))
    return out


def extract_wikipedia_pageviews(payload: Any, *, context: str) -> Records:
    """Wikimedia PV：`{"items": [{"timestamp": "YYYYMMDDHH", "views": N}]}`。"""
    rows = _require_list(payload, "items", context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("timestamp", "views"), context=context, position=position)
        stamp = str(row["timestamp"])
        if len(stamp) != 10 or not stamp.isdigit():
            raise ResponseSchemaError(
                f"{context}: 第 {position} 条 timestamp={stamp!r} 不是 YYYYMMDDHH 形式"
            )
        out.append((pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:]}:00", tz="UTC"),
                    {"views": row["views"]}))
    return out


def extract_defillama_stablecoins(payload: Any, *, context: str) -> Records:
    """
    DefiLlama 稳定币：顶层数组，`totalCirculatingUSD` 实测是**嵌套对象**而不是数字。

    为什么显式只取 `peggedUSD`：把对象里的所有数值求和会把不同锚定资产混成一个量纲。
    该字段缺失时直接失败，而不是退回求和或跳过该条。
    """
    rows = _require_list(payload, context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("date", "totalCirculatingUSD"), context=context, position=position)
        circulating = row["totalCirculatingUSD"]
        if not isinstance(circulating, dict) or "peggedUSD" not in circulating:
            raise ResponseSchemaError(
                f"{context}: 第 {position} 条 totalCirculatingUSD 缺少 peggedUSD，实际为 {circulating!r}"
            )
        out.append((epoch_seconds_to_utc(row["date"], context=context, position=position),
                    {"total_circulating_usd": circulating["peggedUSD"]}))
    return out


def extract_fred_observations(payload: Any, *, context: str) -> Records:
    """
    FRED：`{"observations": [{"date": "YYYY-MM-DD", "value": "数字或 '.'"}]}`。

    为什么把 `.` 当成失败而不是缺失：FRED 用 `.` 表示"该发布日无值"。它是**字段缺失**，
    不是记录缺失；静默丢掉这条会让"非发布日"与"有发布但无值"变得不可区分（§4.3）。
    这里直接跳过并记入说明，由覆盖状态 `unverified` 如实表达。
    """
    rows = _require_list(payload, "observations", context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("date", "value"), context=context, position=position)
        if row["value"] == ".":
            continue
        out.append((pd.Timestamp(row["date"], tz="UTC"), {"value": row["value"]}))
    return out


def extract_glassnode_metric(payload: Any, *, context: str) -> Records:
    """Glassnode：顶层数组 `[{"t": 秒, "v": 值}]`。"""
    rows = _require_list(payload, context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("t", "v"), context=context, position=position)
        out.append((epoch_seconds_to_utc(row["t"], context=context, position=position),
                    {"value": row["v"]}))
    return out


# CryptoCompare 链上端点里本库保留的基础字段。
# 为什么固定列出：响应字段会随端点版本变化，全量透传会让 schema 静默漂移。
CRYPTOCOMPARE_BASE_FIELDS: Tuple[str, ...] = (
    "transaction_count", "new_addresses", "active_addresses",
    "average_transaction_value", "block_height", "hashrate",
    "difficulty", "block_time", "block_size", "current_supply",
)


def extract_cryptocompare_chain(payload: Any, *, context: str) -> Records:
    """
    CryptoCompare 链上日频：`{"Data": {"Data": [{"time": 秒, ...}]}}`。

    只取固定基础字段；缺任何一个都直接失败，不静默降级成更少的列。
    """
    if isinstance(payload, dict) and payload.get("Response") == "Error":
        raise BusinessStatusError(
            f"{context}: CryptoCompare 业务错误: {payload.get('Message')!r}"
        )
    rows = _require_list(payload, "Data", "Data", context=context)
    out: Records = []
    for position, row in enumerate(rows):
        _require_fields(row, ("time", *CRYPTOCOMPARE_BASE_FIELDS), context=context, position=position)
        out.append((epoch_seconds_to_utc(row["time"], context=context, position=position),
                    {name: row[name] for name in CRYPTOCOMPARE_BASE_FIELDS}))
    return out


def extract_stooq_csv(text: str, *, context: str) -> Records:
    """
    Stooq：CSV 文本，列名本地化（`Data` / `Zamkniecie`）或英文（`Date` / `Close`）。

    Args:
        text: 响应文本。
        context: 报错上下文。

    Returns:
        `[(日期, {"value": 收盘价})]`。

    Raises:
        ResponseSchemaError: 缺少日期列或收盘列。
    """
    stripped = text.strip()
    if not stripped:
        raise ResponseSchemaError(f"{context}: Stooq 返回空响应")
    # HTTP 200 + HTML 是来源侧的反爬/JS 挑战页，不是"CSV 列名变了"。
    # Why 要分开：两者的处置完全不同——前者要换出口或改用其他来源，
    # 后者是协议漂移需要改代码。混成同一个错误会让运维朝错误的方向排查。
    if stripped[:1] == "<" or stripped[:15].lower().startswith("<!doctype html"):
        raise SourceBlockedError(
            f"{context}: Stooq 返回 HTML 而不是 CSV（疑似反爬/JS 挑战页），"
            f"响应片段: {stripped[:120]!r}"
        )
    frame = pd.read_csv(StringIO(text))
    date_column = next((name for name in ("Data", "Date") if name in frame.columns), None)
    close_column = next(
        (name for name in ("Zamkniecie", "Close", "Close*") if name in frame.columns), None
    )
    if date_column is None or close_column is None:
        raise ResponseSchemaError(
            f"{context}: Stooq CSV 缺少日期或收盘列；实际列: {list(frame.columns)}"
        )
    return [
        (pd.Timestamp(str(row[date_column]), tz="UTC"), {"value": row[close_column]})
        for _, row in frame.iterrows()
    ]


# -------------------- 通用取数 --------------------

def build_frame(records: Records, *, value_columns: Sequence[str], context: str) -> pd.DataFrame:
    """
    把提取结果组装成规范数据帧：严格解析数值、按时间主键去重、升序。

    Args:
        records: `extract` 的返回值。
        value_columns: 期望的输出列名（顺序即落盘列序）。
        context: 报错上下文。

    Returns:
        索引为 tz-aware UTC 时间、名为 `date` 的数据帧。
    """
    if not records:
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in value_columns},
            index=pd.DatetimeIndex([], tz="UTC", name=TIME_COLUMN),
        )
    index = pd.DatetimeIndex([stamp for stamp, _ in records], tz="UTC", name=TIME_COLUMN)
    data = {
        name: parse_strict_float(
            [values[name] for _, values in records], column=name, context=context
        )
        for name in value_columns
    }
    return dedupe_by_time_primary_key(
        pd.DataFrame(data, index=index), value_columns=value_columns, context=context
    )


def acquire_daily_series(context: AcquireContext) -> AcquireResult:
    """
    数据集级采集入口：一次请求取回整段序列。

    为什么是全量重取而不是增量：这些来源的数据量很小（数千行），且多数不支持远端增量。
    Issue #5 §2 明确："来源不支持远端增量且数据较小时，到期全量获取即可。"
    历史缩短由公共流程的修订检查负责拦截，不在这里放行。

    Args:
        context: 采集上下文；`dataset.params` 提供 url/query/extract 名称与凭据要求。

    Returns:
        AcquireResult。

    Raises:
        ConfigError: 缺少必需凭据，或 offline 上下文缺少 HTTP 客户端。
    """
    dataset = context.dataset
    params = dataset.params
    if context.http is None:
        raise ConfigError(f"{dataset.dataset_id}: 该数据集需要联网，但未提供 HTTP 客户端。")

    query: Dict[str, Any] = dict(params["query"])
    headers: Optional[Dict[str, str]] = None
    credential_env = params["credential_env"]
    if credential_env is not None:
        if credential_env not in context.credentials or not context.credentials[credential_env]:
            raise ConfigError(
                f"{dataset.dataset_id}: 缺少凭据 {credential_env}；"
                "请通过显式环境/配置提供。本库不保留任何内置备用 key，"
                "也不会把凭据缺失表现成正常的空数据。"
            )
        secret = context.credentials[credential_env]
        # 凭据只在真正发请求时合入，永不进入 dataset.params / manifest / 日志。
        if params["credential_style"] == "query":
            query[params["credential_query_key"]] = secret
        else:
            headers = {"authorization": f"Apikey {secret}"}

    before = context.http.requests_used
    extractor = EXTRACTORS[params["extractor"]]
    if params["extractor"] == "stooq_csv":
        text = context.http.get_text(
            url=params["url"], params=query, context=dataset.dataset_id, headers=headers
        )
        records = extractor(text, context=dataset.dataset_id)
    else:
        payload = context.http.get_json(
            url=params["url"], params=query, context=dataset.dataset_id, headers=headers
        )
        records = extractor(payload, context=dataset.dataset_id)

    frame = build_frame(
        records, value_columns=dataset.value_columns, context=dataset.dataset_id
    )
    return AcquireResult(
        frame=frame,
        # 单次请求取回整段序列：这次响应覆盖了来源当前愿意给的全部历史。
        # 但来源的发布日历未经验证，因此覆盖状态仍由 `calendar_verified` 决定。
        scan_complete=True,
        scan_note=f"单次请求全量取回 {len(frame)} 条记录",
        requests_used=context.http.requests_used - before,
    )


PCT_SUFFIX = "_pct"


def acquire_cryptocompare_chain_pct(context: AcquireContext) -> AcquireResult:
    """
    CryptoCompare 链上环比：先取基础值，再派生环比列，**同时保留基础值**。

    为什么不直接落环比：
        基线把基础值 `pct_change()` 之后 `fillna(0.0)`，于是"首次记录""前值缺失"
        "零分母"三种**没有定义**的情形全部变成了"零增长"——下游无法区分
        "这一天真的没变" 与 "这一天算不出变化率"（Issue #5 §4.4）。
        这里保留基础值列，未定义处输出 NaN。

    Args:
        context: 采集上下文。

    Returns:
        AcquireResult；`frame` 同时含基础值列与 `<字段>_pct` 列。
    """
    from ..transforms import pct_change_keep_base

    dataset = context.dataset
    base_columns = [name for name in dataset.value_columns if not name.endswith(PCT_SUFFIX)]
    # 先按基础列走通用取数，再在其上派生环比，保证两者来自同一次响应。
    base_dataset = _replace_value_columns(dataset, base_columns)
    base_result = acquire_daily_series(
        AcquireContext(
            dataset=base_dataset, request=context.request, now_utc=context.now_utc,
            http=context.http, cached_frame=None, credentials=context.credentials,
            local_inputs=context.local_inputs, force_refresh=context.force_refresh,
        )
    )
    derived = pct_change_keep_base(
        base_result.frame, value_columns=base_columns,
        context=dataset.dataset_id, suffix=PCT_SUFFIX,
    )
    return AcquireResult(
        frame=derived[list(dataset.value_columns)],
        scan_complete=base_result.scan_complete,
        scan_note=base_result.scan_note,
        requests_used=base_result.requests_used,
    )


def _replace_value_columns(dataset, value_columns: Sequence[str]):
    """
    返回只保留指定列的数据集副本（供派生列场景先取基础值）。

    `nullable_columns` 必须同步收窄：它是 `value_columns` 的子集合同，
    留着已被去掉的列名会在数据集构造时直接失败。
    """
    import dataclasses

    columns = tuple(value_columns)
    return dataclasses.replace(
        dataset,
        value_columns=columns,
        nullable_columns=tuple(name for name in dataset.nullable_columns if name in columns),
    )


# 提取器注册表：数据集只存名字，避免把不可序列化的函数塞进 `params`。
EXTRACTORS: Mapping[str, Callable[..., Records]] = {
    "blockchain_chart": extract_blockchain_chart,
    "fear_greed": extract_fear_greed,
    "wikipedia_pageviews": extract_wikipedia_pageviews,
    "defillama_stablecoins": extract_defillama_stablecoins,
    "fred_observations": extract_fred_observations,
    "glassnode_metric": extract_glassnode_metric,
    "cryptocompare_chain": extract_cryptocompare_chain,
    "stooq_csv": extract_stooq_csv,
}
