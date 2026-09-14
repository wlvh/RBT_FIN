# -*- coding: utf-8 -*-
"""
trading 兼容 CSV 导出。

功能或目的：
    trading 的实际消费者读取固定名称 CSV，列序为 `datetime,open,high,low,close,volume`，
    时间是 UTC 语义的**无时区** `%Y-%m-%d %H:%M:%S` 文本（trading#105 §2.2）。
    本库内部一律保存 tz-aware ISO 文本，因此导出是一次真实的格式转换，
    转换结果必须**验证导出文件本身**并计算**它自己的哈希**；来源内容哈希单独保留。
    字节不同的数据不能用硬链接代替转换，也不能混用两个哈希。

调用关系：
    - `rbt_fin.cli export-trading-csv` 与 trading 的 `trading_data_pipeline.py` shim 调用
      `export_trading_csv()`。
    - 校验逻辑复用 `rbt_fin.timeutils` 的网格断言，保证与采集端同一套时间合同。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .__about__ import LEGACY_CSV_FORMAT_VERSION
from .errors import ConfigError, TimeContractError, ValueContractError
from .storage import sha256_file
from .timeutils import (
    assert_on_grid,
    assert_strictly_increasing_unique,
    expected_grid,
    require_utc,
)

LEGACY_COLUMNS: Sequence[str] = ("datetime", "open", "high", "low", "close", "volume")
LEGACY_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class ExportRecord:
    """
    一次导出的可归档记录。

    Attributes:
        export_path: 导出文件绝对路径。
        export_sha256: **导出文件自己的**字节哈希。
        export_bytes: 导出文件字节数。
        source_content_sha256: 来源内容文件哈希（与导出哈希是两个不同的东西）。
        rows: 导出行数。
        actual_start / actual_end: 导出数据的实际首尾时间（ISO-8601 UTC 文本）。
        requested_start / requested_end_exclusive: 本次导出请求范围。
        csv_format_version: 导出格式版本。
        interval: 网格周期。
    """

    export_path: str
    export_sha256: str
    export_bytes: int
    source_content_sha256: str
    rows: int
    actual_start: Optional[str]
    actual_end: Optional[str]
    requested_start: str
    requested_end_exclusive: str
    csv_format_version: str
    interval: str

    def to_dict(self) -> dict:
        """序列化为可写入输入记录的字典。"""
        return {
            "export_path": self.export_path,
            "export_sha256": self.export_sha256,
            "export_bytes": int(self.export_bytes),
            "source_content_sha256": self.source_content_sha256,
            "rows": int(self.rows),
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "requested_start": self.requested_start,
            "requested_end_exclusive": self.requested_end_exclusive,
            "csv_format_version": self.csv_format_version,
            "interval": self.interval,
        }


def _assert_legacy_contract(frame: pd.DataFrame, *, interval: str, context: str) -> None:
    """
    断言一份（已回读的）legacy CSV 满足 trading 的消费合同。

    检查项：分钟网格精确对齐、严格升序、主键唯一、列序固定、数值有限。

    Raises:
        TimeContractError / ValueContractError: 任一项不满足。
    """
    if list(frame.columns) != list(LEGACY_COLUMNS[1:]):
        raise TimeContractError(
            f"{context}: 导出列序必须为 {list(LEGACY_COLUMNS)}，实际数据列为 {list(frame.columns)}"
        )
    assert_on_grid(frame.index, interval=interval, context=context)
    assert_strictly_increasing_unique(frame.index, context=context)
    for column in LEGACY_COLUMNS[1:]:
        series = frame[column]
        if series.isna().any():
            raise ValueContractError(f"{context}: 导出数据列 {column!r} 含空值。")


def export_trading_csv(
    *,
    frame: pd.DataFrame,
    target_path: Path,
    interval: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    source_content_sha256: str,
    require_full_coverage: bool = True,
) -> ExportRecord:
    """
    把行情数据导出为 trading 兼容 CSV，并验证导出文件本身。

    流程（任何一步失败都不会留下可被消费的半文件）：
        切窗 -> 校验网格/唯一/升序 -> 写独立命名临时文件 -> fsync ->
        回读并按 legacy 合同再校验 -> 原子替换到目标路径 -> 计算导出文件哈希。

    Args:
        frame: 索引为 tz-aware UTC 时间的 OHLCV 数据（可包含比请求更宽的历史）。
        target_path: 目标文件路径（例如 `<batch_dir>/BTCUSDT_future_1m.csv`）。
        interval: 网格周期。
        start: 导出范围左闭端点。
        end_exclusive: 导出范围右开端点。
        source_content_sha256: 来源内容文件哈希，原样保留在记录中。
        require_full_coverage: 是否要求导出范围内网格无缺口（批次输入必须为 True）。

    Returns:
        ExportRecord。

    Raises:
        ConfigError: 数据缺少必需列。
        TimeContractError: 网格/顺序/唯一性不满足，或要求完整覆盖时存在缺口。
    """
    start_ts = require_utc(start, arg_name="export.start")
    end_ts = require_utc(end_exclusive, arg_name="export.end_exclusive")
    missing = [col for col in LEGACY_COLUMNS[1:] if col not in frame.columns]
    if missing:
        raise ConfigError(f"导出缺少必需列 {missing}；实际列: {list(frame.columns)}")

    window = frame[(frame.index >= start_ts) & (frame.index < end_ts)][list(LEGACY_COLUMNS[1:])]
    _assert_legacy_contract(window, interval=interval, context=f"export:{target_path.name}")
    if require_full_coverage:
        grid = expected_grid(start=start_ts, end_exclusive=end_ts, interval=interval)
        gaps = grid.difference(window.index)
        if len(gaps) > 0:
            raise TimeContractError(
                f"export:{target_path.name}: 导出范围内存在 {len(gaps)} 个缺口，"
                f"示例 {[str(t) for t in gaps[:5]]}；批次输入不允许带缺口发布。"
            )

    out = window.copy()
    # 唯一允许出现无时区文本的地方：legacy 消费合同要求 UTC 语义的 naive 秒精度文本。
    out.index = out.index.tz_convert("UTC").strftime(LEGACY_TIME_FORMAT)
    out.index.name = LEGACY_COLUMNS[0]

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target_path.parent), prefix=f".{target_path.name}.tmp-", suffix=".csv"
    )
    tmp_path = Path(tmp_name)
    try:
        # 直接写入文件句柄，不在内存里先拼出完整字节：
        # 生产规模下这份文本有数百 MB，先拼后写会让峰值内存凭空翻倍。
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            out.to_csv(handle, index=True, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        # 回读验证导出文件本身：证明写下去的字节能按 legacy 合同被读回来。
        readback = pd.read_csv(tmp_path)  # 直接按路径解析，避免额外持有整份字节
        if list(readback.columns) != list(LEGACY_COLUMNS):
            raise TimeContractError(
                f"export:{target_path.name}: 回读列序为 {list(readback.columns)}，"
                f"期望 {list(LEGACY_COLUMNS)}"
            )
        parsed = pd.to_datetime(readback[LEGACY_COLUMNS[0]], format=LEGACY_TIME_FORMAT)
        readback = readback.set_index(pd.DatetimeIndex(parsed).tz_localize("UTC"))
        readback.index.name = "datetime"
        _assert_legacy_contract(
            readback[list(LEGACY_COLUMNS[1:])], interval=interval,
            context=f"export-readback:{target_path.name}",
        )
        if len(readback) != len(window):
            raise TimeContractError(
                f"export:{target_path.name}: 回读行数 {len(readback)} 与导出行数 {len(window)} 不符。"
            )
        os.replace(str(tmp_path), str(target_path))
    except BaseException:
        # 任何失败都不留下可被消费的半文件。
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return ExportRecord(
        export_path=str(target_path.resolve()),
        export_sha256=sha256_file(target_path),
        export_bytes=target_path.stat().st_size,
        source_content_sha256=source_content_sha256,
        rows=int(len(window)),
        actual_start=None if len(window) == 0 else window.index.min().isoformat(),
        actual_end=None if len(window) == 0 else window.index.max().isoformat(),
        requested_start=start_ts.isoformat(),
        requested_end_exclusive=end_ts.isoformat(),
        csv_format_version=LEGACY_CSV_FORMAT_VERSION,
        interval=interval,
    )
