# -*- coding: utf-8 -*-
"""
公共写安全原语与纯数值合同的单元测试（Issue #5 §6 / §4.2）。

功能或目的：
    `exclusive_lock`、`atomic_write_bytes` 与 `validate_ohlcv_values` 是从
    `DatasetStore` / `validate_strict_grid_ohlcv` 里提出来的公共原语，目的就是让
    库外消费者（例如 trading 的实盘滚动窗口）不必再写第二份略有差异的实现。
    既然对外承诺"复用这三个就够安全"，就必须在这里把承诺钉死：

        - 固定临时文件名会让两个写者互踩，独立命名不会；
        - 锁必须覆盖整个"读旧状态 -> 更新 -> 发布"周期，只锁最后一次写不够；
        - 数值合同与时间轴、覆盖率无关，任何持有 OHLCV 的路径都能直接调用。

调用关系：
    - 被测对象 `rbt_fin.storage` / `rbt_fin.validation` 的公共函数。
    - 并发用例使用真实子进程与同步点，不用 sleep 猜时序。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rbt_fin.errors import LockError, PublishError, ValueContractError
from rbt_fin.storage import atomic_write_bytes, exclusive_lock, sha256_bytes


def ohlcv(rows: int = 3) -> pd.DataFrame:
    """构造一段合法 OHLCV，供数值合同用例做局部改坏。"""
    index = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC", name="datetime")
    base = np.arange(rows, dtype="float64") + 100.0
    return pd.DataFrame(
        {"open": base, "high": base + 1.0, "low": base - 1.0, "close": base + 0.5, "volume": base},
        index=index,
    )


# --------------------------------------------------------------------------
# atomic_write_bytes
# --------------------------------------------------------------------------

def test_atomic_write_uses_independent_temp_names(tmp_path):
    """两次写入使用不同的临时文件名；提交后目录里只剩正式文件。"""
    seen = []
    target = tmp_path / "out.bin"
    for payload in (b"first", b"second"):
        atomic_write_bytes(
            target=target, payload=payload, tmp_prefix=".t-", tmp_suffix=".bin",
            context="test", before_replace=lambda: seen.extend(sorted(
                p.name for p in tmp_path.iterdir() if p.name.startswith(".t-")
            )),
        )
    assert target.read_bytes() == b"second"
    assert len(set(seen)) == 2, f"两次写入复用了同一个临时文件名: {seen}"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_atomic_write_leaves_old_file_untouched_on_failure(tmp_path):
    """提交点之前失败：旧正式文件不变，且不留临时文件。"""
    target = tmp_path / "out.bin"
    atomic_write_bytes(target=target, payload=b"old", tmp_prefix=".t-", tmp_suffix=".bin", context="test")

    def boom():
        raise RuntimeError("提交前中断")

    with pytest.raises(RuntimeError, match="提交前中断"):
        atomic_write_bytes(
            target=target, payload=b"new", tmp_prefix=".t-", tmp_suffix=".bin",
            context="test", before_replace=boom,
        )
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_atomic_write_detects_untrusted_disk_write(tmp_path, monkeypatch):
    """回读哈希不符时必须报 PublishError，而不是把不可信的字节发布出去。"""
    monkeypatch.setattr("rbt_fin.storage.sha256_file", lambda path, **kw: "deadbeef")
    with pytest.raises(PublishError, match="哈希与内存字节不符"):
        atomic_write_bytes(
            target=tmp_path / "out.bin", payload=b"x", tmp_prefix=".t-", tmp_suffix=".bin",
            context="test", verify_written_hash=True,
        )
    assert list(tmp_path.iterdir()) == []


def _legacy_writer_a(target: str, a_wrote, b_opened, out: str) -> None:
    """反例写者 A：照搬"固定 .tmp"的语句序，写完内容后等 B 打开同一个临时文件再提交。"""
    temp = Path(target + ".tmp")
    handle = open(temp, "wb")
    handle.write(b"AAAA")
    handle.flush()
    a_wrote.set()
    b_opened.wait(5)
    os.fsync(handle.fileno())
    os.replace(temp, Path(target))
    handle.close()
    Path(out).write_text("A 自认发布成功", encoding="utf-8")


def _legacy_writer_b(target: str, a_wrote, b_opened, out: str) -> None:
    """反例写者 B：在 A 写完之后才打开同一个固定临时文件——"wb" 会直接截断 A 的内容。"""
    temp = Path(target + ".tmp")
    a_wrote.wait(5)
    handle = open(temp, "wb")
    b_opened.set()
    time.sleep(0.4)  # 让 A 先完成 replace
    try:
        handle.write(b"BBBB"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, Path(target)); handle.close()
        Path(out).write_text("ok", encoding="utf-8")
    except OSError as exc:
        Path(out).write_text(type(exc).__name__, encoding="utf-8")


def test_shared_temp_name_corrupts_published_file_but_primitive_does_not(tmp_path):
    """
    确定性对照：固定 `.tmp` 会让"自认发布成功"的正式文件变成空文件；公共原语不会。

    为什么保留这个反例进程：没有它，"独立临时文件"只是一句注释；
    有了它，这条安全属性才是被证明过的。时序由 Event 确定，不靠 sleep 猜。
    """
    target = tmp_path / "shared.bin"
    a_wrote, b_opened = mp.Event(), mp.Event()
    ra, rb = tmp_path / "a.txt", tmp_path / "b.txt"
    pa = mp.Process(target=_legacy_writer_a, args=(str(target), a_wrote, b_opened, str(ra)))
    pb = mp.Process(target=_legacy_writer_b, args=(str(target), a_wrote, b_opened, str(rb)))
    pa.start(); pb.start()
    pa.join(10)
    # 阶段一：A 自认发布成功，但 B 的 "wb" 已经把它的内容截掉了，读者此刻看到空文件。
    assert ra.read_text(encoding="utf-8") == "A 自认发布成功"
    assert target.stat().st_size == 0, (
        f"固定 .tmp 的反例未复现阶段一（正式文件 {target.read_bytes()!r}）；"
        "若本断言失败说明复现条件变了，需要重新设计反例而不是删掉它。"
    )
    pb.join(10)
    # 阶段二：B 的 fd 指向的正是那个已被 rename 成正式文件的 inode，
    # 于是它继续往"已发布的文件"里写，而 A 从未同意过这次改动；B 自己的 replace 则找不到源。
    assert target.read_bytes() == b"BBBB"
    assert rb.read_text(encoding="utf-8") == "FileNotFoundError"

    # 同样的交错换成公共原语：两个写者各写各的临时文件，正式文件必然是某一方的完整内容。
    target2 = tmp_path / "safe.bin"
    for payload in (b"AAAA", b"BBBB"):
        atomic_write_bytes(target=target2, payload=payload, tmp_prefix=".s-", tmp_suffix=".bin", context="t")
    assert target2.read_bytes() == b"BBBB"
    assert sha256_bytes(target2.read_bytes()) == sha256_bytes(b"BBBB")


# --------------------------------------------------------------------------
# exclusive_lock
# --------------------------------------------------------------------------

def test_exclusive_lock_rejects_second_writer_when_non_blocking(tmp_path):
    """非阻塞模式下第二写者必须立即失败，而不是并行进入更新周期。"""
    lock = tmp_path / "w.lock"
    with exclusive_lock(lock_path=lock, context="first"):
        # 同进程内再取同一把 flock 不会自阻塞，因此用子进程验证跨进程互斥。
        proc = mp.Process(target=_try_lock_nonblocking, args=(str(lock), str(tmp_path / "r.txt")))
        proc.start(); proc.join(10)
    assert (tmp_path / "r.txt").read_text() == "LockError"


def _try_lock_nonblocking(lock_path: str, out: str) -> None:
    """子进程：非阻塞抢锁，把结果写文件。"""
    try:
        with exclusive_lock(lock_path=Path(lock_path), context="second", blocking=False):
            Path(out).write_text("acquired", encoding="utf-8")
    except LockError:
        Path(out).write_text("LockError", encoding="utf-8")


def _locked_increment(lock_path: str, counter_path: str) -> None:
    """子进程：在锁内完成"读 -> 加一 -> 写"整个周期，中间故意让出 CPU。"""
    with exclusive_lock(lock_path=Path(lock_path), context="worker"):
        current = int(Path(counter_path).read_text(encoding="utf-8"))
        time.sleep(0.2)  # 放大"读后写前"的窗口；没有锁时必然丢更新
        atomic_write_bytes(
            target=Path(counter_path), payload=str(current + 1).encode(),
            tmp_prefix=".c-", tmp_suffix=".txt", context="worker",
        )


def test_exclusive_lock_covers_whole_read_modify_write_cycle(tmp_path):
    """
    锁必须覆盖整个周期：两个真实写者各自"读-改-写"后，计数必须是 2 而不是 1。

    只给最后一次写加锁的实现会在这里得到 1——后写者拿着更旧的输入覆盖了先写者。
    """
    lock, counter = tmp_path / "w.lock", tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    procs = [mp.Process(target=_locked_increment, args=(str(lock), str(counter))) for _ in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(20)
    assert counter.read_text(encoding="utf-8") == "2"


# --------------------------------------------------------------------------
# validate_ohlcv_values
# --------------------------------------------------------------------------

def test_validate_ohlcv_values_accepts_valid_frame():
    """合法数据不应被拒绝。"""
    from rbt_fin import validate_ohlcv_values
    validate_ohlcv_values(ohlcv(), context="t")


@pytest.mark.parametrize(
    "column,row,value,match",
    [
        ("high", 0, 0.5, "high/low 未正确包住"),   # high < low
        ("volume", 1, -1.0, "成交量为负"),
        ("close", 2, float("nan"), "非有限值"),
        ("open", 0, -1.0, "非有限值|非正价格"),
        ("low", 1, 0.0, "非正价格"),
    ],
)
def test_validate_ohlcv_values_rejects_each_violation(column, row, value, match):
    """五类数值违约逐条拒绝，并给出可定位的索引示例。"""
    from rbt_fin import validate_ohlcv_values
    frame = ohlcv()
    frame.iloc[row, frame.columns.get_loc(column)] = value
    with pytest.raises(ValueContractError, match=match):
        validate_ohlcv_values(frame, context="t")


def test_validate_ohlcv_values_rejects_missing_column():
    """缺列必须在数值合同入口就失败，而不是等到后面某个 KeyError。"""
    from rbt_fin import validate_ohlcv_values
    with pytest.raises(ValueContractError, match="缺少必需 OHLCV 列"):
        validate_ohlcv_values(ohlcv().drop(columns=["volume"]), context="t")
