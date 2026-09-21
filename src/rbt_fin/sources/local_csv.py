# -*- coding: utf-8 -*-
"""
本地 CSV 导入适配器（不联网）。

功能或目的：
    Issue #5 §3.2 要求“本地导入不联网，校验内容和范围，记录哈希，不修改用户原文件”。
    本适配器同时承担一个结构性作用：它证明公共流程（缓存资格 -> 采集 -> 校验 -> 发布）
    没有被硬编码成行情——同一条流程可以驱动一个既不分页也没有严格网格的低频本地来源。

调用关系：
    `rbt_fin.registry.build_local_csv_dataset()` 绑定本模块的 `acquire_local_csv`；
    源文件哈希写入采集说明（manifest 的 `extra.scan_note`：`source_file_sha256=…`），作为本地导入的可追溯证据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..contracts import AcquireContext, AcquireResult
from ..errors import ConfigError, ResponseSchemaError
from ..storage import sha256_file
from ..validation import parse_strict_float


def read_local_csv(
    *, path: Path, time_column: str, value_columns: Sequence[str], context: str
) -> pd.DataFrame:
    """
    以只读方式解析本地 CSV，并严格校验列与数值。

    Args:
        path: 用户提供的 CSV 路径；只读，绝不改写。
        time_column: 时间列名；内容必须可解析为 UTC 时间。
        value_columns: 必需的数值列。
        context: 报错上下文。

    Returns:
        索引为 tz-aware UTC 时间、名为 `time_column` 的数据帧。

    Raises:
        ConfigError: 文件不存在。
        ResponseSchemaError: 缺少必需列。
    """
    if not path.exists():
        raise ConfigError(f"{context}: 本地导入文件不存在: {path}")
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = [col for col in (time_column, *value_columns) if col not in frame.columns]
    if missing:
        raise ResponseSchemaError(
            f"{context}: 本地 CSV 缺少必需列 {missing}；实际列: {list(frame.columns)}"
        )
    # 为什么用 utc=True 而不是先 localize：本地导入是“有明确 legacy 合同的导入边界”，
    # 无时区文本按 UTC 解释；带时区文本则统一转换到 UTC。
    times = pd.to_datetime(frame[time_column], utc=True)
    data = {
        column: parse_strict_float(frame[column].tolist(), column=column, context=context)
        for column in value_columns
    }
    out = pd.DataFrame(data, index=pd.DatetimeIndex(times, name=time_column))
    return out.sort_index()


def acquire_local_csv(context: AcquireContext) -> AcquireResult:
    """
    本地导入入口：读取用户文件、记录其哈希，并只保留请求范围内的记录。

    Args:
        context: 采集上下文；`local_inputs['csv_path']` 为必填。

    Returns:
        AcquireResult；`scan_note` 内含源文件哈希，便于归档追溯。

    Raises:
        ConfigError: 未提供 `csv_path`。
    """
    dataset = context.dataset
    local_inputs = context.local_inputs
    if "csv_path" not in local_inputs:
        raise ConfigError(f"{dataset.dataset_id}: 本地导入必须显式提供 local_inputs['csv_path']。")
    path = Path(local_inputs["csv_path"])
    frame = read_local_csv(
        path=path,
        time_column=dataset.time_column,
        value_columns=dataset.value_columns,
        context=dataset.dataset_id,
    )
    # 为什么返回整份文件而不是请求范围切片：
    # 采集函数的返回值就是“本数据集的新完整内容”。若在这里切窗，一次窄窗口请求会把
    # 缓存里更早的历史直接丢掉，违反“不因默认值裁掉已存历史”。覆盖只在校验阶段按请求范围统计。
    source_hash = sha256_file(path)
    return AcquireResult(
        frame=frame,
        scan_complete=True,
        scan_note=f"本地导入（未联网）：source_file={path.name}, source_file_sha256={source_hash}",
        requests_used=0,
    )
