# -*- coding: utf-8 -*-
"""
trading 兼容 CSV 导出：列序、无时区秒精度文本、网格对齐、导出文件自身哈希。

合同来源：trading#105 §2.2。
    - 固定名称与列序 `datetime,open,high,low,close,volume`。
    - UTC 语义的无时区 `%Y-%m-%d %H:%M:%S` 文本，分钟网格精确对齐、升序、主键唯一、
      只包含已完成周期。
    - 与来源内容哈希是两个不同的哈希，不得混用。
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from rbt_fin.errors import TimeContractError
from rbt_fin.exports import LEGACY_COLUMNS, export_trading_csv

SOURCE_HASH = "0" * 64


def _frame(count: int, *, start: str = "2026-01-01 00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="1min", tz="UTC", name="datetime")
    base = [30000.0 + i for i in range(count)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [p + 5 for p in base],
            "low": [p - 5 for p in base],
            "close": [p + 1 for p in base],
            "volume": [1.0 + i for i in range(count)],
        },
        index=index,
    )


def test_export_matches_legacy_contract(tmp_path):
    """导出文件必须满足 legacy 合同，并可被真实读取器按固定格式读回。"""
    target = tmp_path / "BTCUSDT_future_1m.csv"
    record = export_trading_csv(
        frame=_frame(5), target_path=target, interval="1m",
        start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
        source_content_sha256=SOURCE_HASH,
    )
    text = target.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert lines[0] == ",".join(LEGACY_COLUMNS)
    assert lines[1].startswith("2026-01-01 00:00:00,")
    assert "+00:00" not in text, "legacy CSV 必须是无时区文本"
    assert record.rows == 5

    # 真实读取器路径：按固定格式解析，主键唯一且严格升序。
    back = pd.read_csv(target, index_col="datetime", parse_dates=["datetime"])
    assert back.index.is_monotonic_increasing and back.index.is_unique
    assert (back.index.second == 0).all()
    assert list(back.columns) == list(LEGACY_COLUMNS[1:])


def test_export_hash_is_the_export_files_own_hash(tmp_path):
    """
    导出哈希必须是**导出文件自己的**字节哈希；来源内容哈希单独保留，不得混用。
    """
    target = tmp_path / "BTCUSDT_future_1m.csv"
    record = export_trading_csv(
        frame=_frame(3), target_path=target, interval="1m",
        start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-01-01 00:03", tz="UTC"),
        source_content_sha256=SOURCE_HASH,
    )
    assert record.export_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert record.source_content_sha256 == SOURCE_HASH
    assert record.export_sha256 != record.source_content_sha256
    assert record.export_bytes == target.stat().st_size


def test_export_rejects_gap_in_requested_window(tmp_path):
    """批次输入不允许带缺口发布：范围内缺一分钟必须失败。"""
    frame = _frame(5).drop(index=pd.Timestamp("2026-01-01 00:02", tz="UTC"))
    target = tmp_path / "BTCUSDT_future_1m.csv"
    with pytest.raises(TimeContractError, match="存在 1 个缺口"):
        export_trading_csv(
            frame=frame, target_path=target, interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
            source_content_sha256=SOURCE_HASH,
        )
    assert not target.exists(), "失败不得留下可被消费的半文件"


def test_export_rejects_off_grid_timestamp(tmp_path):
    """半秒偏移的数据不得被导出为 legacy CSV（反例 A 在导出边界的再次拦截）。"""
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00:00", "2026-01-01 00:01:00.500000"], tz="UTC", name="datetime"
    )
    frame = _frame(2)
    frame.index = index
    target = tmp_path / "BTCUSDT_future_1m.csv"
    with pytest.raises(TimeContractError, match="未落在 interval=1m 网格上"):
        export_trading_csv(
            frame=frame, target_path=target, interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:02", tz="UTC"),
            source_content_sha256=SOURCE_HASH,
        )
    assert not target.exists()


def test_export_window_excludes_records_outside_range(tmp_path):
    """
    导出只取请求窗口：缓存里更早/更晚的历史不进入批次输入，但也不会导致失败。
    """
    target = tmp_path / "BTCUSDT_future_1m.csv"
    record = export_trading_csv(
        frame=_frame(10), target_path=target, interval="1m",
        start=pd.Timestamp("2026-01-01 00:02", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
        source_content_sha256=SOURCE_HASH,
    )
    assert record.rows == 3
    assert record.actual_start.startswith("2026-01-01T00:02")
    assert record.actual_end.startswith("2026-01-01T00:04")


def test_repeated_export_is_byte_identical(tmp_path):
    """
    同样的数据与窗口必须导出字节完全相同的文件：批次输入可复现的前提。
    """
    frame = _frame(5)
    hashes = []
    for name in ("a.csv", "b.csv"):
        record = export_trading_csv(
            frame=frame, target_path=tmp_path / name, interval="1m",
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
            source_content_sha256=SOURCE_HASH,
        )
        hashes.append(record.export_sha256)
    assert hashes[0] == hashes[1]
