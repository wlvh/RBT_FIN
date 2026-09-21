# -*- coding: utf-8 -*-
"""
两个关键中断 + 一个真实双写者（Issue #5 §8）。

证据边界：本文件的结论只适用于**本地文件系统**上的进程终止（SIGKILL）。
它不冒充“所有断电情形都已验证”——断电还涉及磁盘缓存与固件行为，不在 v1 承诺内。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from rbt_fin.registry import build_klines_dataset
from rbt_fin.storage import (
    PHASE_AFTER_MANIFEST_REPLACE,
    PHASE_BEFORE_MANIFEST_REPLACE,
    DatasetStore,
    sha256_file,
)

CHILD = Path(__file__).parent / "child_worker.py"
DATASET = build_klines_dataset()
SYNC_TIMEOUT = 30.0


def _spawn(args: list) -> subprocess.Popen:
    """启动子进程工作器；PYTHONPATH 指向仓库根，保证子进程能导入已安装的包。"""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, str(CHILD), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )


def _wait_file(path: Path) -> None:
    """等待同步点文件出现。"""
    deadline = time.monotonic() + SYNC_TIMEOUT
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"等待同步点文件超时: {path}")
        time.sleep(0.01)


def _seed(cache_root: Path, tmp_path: Path) -> str:
    """先建立一份正式状态（5 分钟），返回其内容哈希。"""
    proc = _spawn([
        "writer", "--cache-root", str(cache_root),
        "--start", "2026-01-01 00:00", "--end-exclusive", "2026-01-01 00:05",
        "--source-base", "2026-01-01 00:00", "--source-count", "5",
    ])
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    return json.loads(out.strip().splitlines()[-1])["sha256"]


def test_kill_before_commit_leaves_old_state_readable(cache_root, tmp_path):
    """
    提交前被真实 SIGKILL：旧正式状态必须原样可读。

    杀死点选在“内容已写、manifest 尚未替换”——这是唯一可能留下孤儿文件的窗口。
    """
    old_sha = _seed(cache_root, tmp_path)
    signal_file = tmp_path / "killed_before_commit"

    proc = _spawn([
        "publish-kill", "--cache-root", str(cache_root),
        "--start", "2026-01-01 00:00", "--end-exclusive", "2026-01-01 00:10",
        "--source-base", "2026-01-01 00:00", "--source-count", "10",
        "--kill-phase", PHASE_BEFORE_MANIFEST_REPLACE,
        "--signal-file", str(signal_file),
    ])
    _wait_file(signal_file)
    proc.wait(timeout=60)
    assert proc.returncode == -9, f"子进程应被 SIGKILL 终止，实际 returncode={proc.returncode}"

    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.content_sha256 == old_sha, "提交前被杀 -> 仍指向旧内容"
    assert manifest.rows == 5
    # 旧内容可完整读回，且哈希与 manifest 一致（缓存未被污染）。
    frame = store.read_frame(manifest, verify_content=True)
    assert len(frame) == 5


def test_kill_after_commit_makes_new_state_identifiable(cache_root, tmp_path):
    """
    提交后被真实 SIGKILL：新状态可能已可见；恢复必须能通过 manifest + 内容哈希识别它。

    这里如实断言“新状态已发布”，而不是声称“必然未发布”。
    """
    old_sha = _seed(cache_root, tmp_path)
    signal_file = tmp_path / "killed_after_commit"

    proc = _spawn([
        "publish-kill", "--cache-root", str(cache_root),
        "--start", "2026-01-01 00:00", "--end-exclusive", "2026-01-01 00:10",
        "--source-base", "2026-01-01 00:00", "--source-count", "10",
        "--kill-phase", PHASE_AFTER_MANIFEST_REPLACE,
        "--signal-file", str(signal_file),
    ])
    _wait_file(signal_file)
    proc.wait(timeout=60)
    assert proc.returncode == -9

    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    manifest = store.read_manifest()
    assert manifest.content_sha256 != old_sha, "提交后被杀 -> 新状态已可见"
    assert manifest.rows == 10
    # 恢复识别：manifest 指向的内容文件存在，且其实际字节哈希与 manifest 声明一致。
    content_path = store.content_path(manifest.content_file)
    assert sha256_file(content_path) == manifest.content_sha256
    frame = store.read_frame(manifest, verify_content=True)
    assert len(frame) == 10


def test_two_real_writers_do_not_lose_updates(cache_root, tmp_path):
    """
    真实双写者：第二写者必须从**新状态**更新，不得覆盖第一写者的数据。

    顺序由同步点控制，不靠 sleep：
        1. 写者 A 进入锁内、提交前停下，创建 `a_in_lock`。
        2. 写者 B 先用非阻塞锁证明“锁确实被别人持有”，再创建 `b_blocked`。
        3. A 看到 `b_blocked` 后继续提交并释放锁。
        4. B 的阻塞式获取返回，此时它必须读到 A 已提交的新状态。

    两个写者写的是**不相交**的区间：若 B 用取锁前的旧快照更新，A 的 5 条就会丢失。
    """
    a_in_lock = tmp_path / "a_in_lock"
    b_blocked = tmp_path / "b_blocked"

    writer_a = _spawn([
        "writer", "--cache-root", str(cache_root),
        "--start", "2026-01-01 00:00", "--end-exclusive", "2026-01-01 00:05",
        "--source-base", "2026-01-01 00:00", "--source-count", "10",
        "--signal-file", str(a_in_lock), "--wait-file", str(b_blocked),
    ])
    _wait_file(a_in_lock)

    writer_b = _spawn([
        "lock-prober", "--cache-root", str(cache_root),
        "--start", "2026-01-01 00:05", "--end-exclusive", "2026-01-01 00:10",
        "--source-base", "2026-01-01 00:00", "--source-count", "10",
        "--signal-file", str(b_blocked),
    ])

    out_a, err_a = writer_a.communicate(timeout=120)
    out_b, err_b = writer_b.communicate(timeout=120)
    assert writer_a.returncode == 0, err_a
    assert writer_b.returncode == 0, err_b

    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    manifest = store.read_manifest()
    frame = store.read_frame(manifest, verify_content=True)
    assert len(frame) == 10, "第二写者必须在第一写者的结果之上更新，不能丢失其数据"
    expected = pd.date_range("2026-01-01 00:00", periods=10, freq="1min", tz="UTC")
    assert list(frame.index) == list(expected)
