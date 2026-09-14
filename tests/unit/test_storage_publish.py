# -*- coding: utf-8 -*-
"""
发布失败与缓存完整性：写内容 / 写 manifest / 替换失败必须传播，旧正式状态必须不变。

Issue #5 §6：
    - 缓存内容文件永不原地修改；替换 `manifest.json` 是唯一提交点。
    - 提交前失败：旧正式状态不变；首次初始化不留下冒充正式版本的文件。
    - 临时文件名独立，不共用固定 `.tmp`。
    - 已存在同哈希文件先核验，不原地覆盖；缓存损坏必须拒绝。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from rbt_fin.errors import CacheCorruptionError, PublishError
from rbt_fin.registry import build_klines_dataset
from rbt_fin.storage import (
    MANIFEST_FILENAME,
    PHASE_AFTER_CONTENT_WRITE,
    PHASE_BEFORE_CONTENT_WRITE,
    PHASE_BEFORE_MANIFEST_REPLACE,
    DatasetStore,
    sha256_file,
)

DATASET = build_klines_dataset()


def _frame(count: int, *, close_bump: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=count, freq="1min", tz="UTC", name="datetime")
    base = [30000.0 + i for i in range(count)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [p + 5 for p in base],
            "low": [p - 5 for p in base],
            "close": [p + 1 + close_bump for p in base],
            "volume": [1.0] * count,
        },
        index=index,
    )


def _fields(**overrides) -> dict:
    fields = {
        "identity": dict(DATASET.identity),
        "params": dict(DATASET.params),
        "time_meaning": DATASET.time_meaning,
        "schema_version": DATASET.schema_version,
        "transform_version": DATASET.transform_version,
        "actual_start": "2026-01-01T00:00:00+00:00",
        "actual_end": "2026-01-01T00:04:00+00:00",
        "coverage": {
            "status": "complete",
            "checked_start": "2026-01-01T00:00:00+00:00",
            "checked_end_exclusive": "2026-01-01T00:05:00+00:00",
            "scan_complete": True,
        },
        "fetched_at": "2026-01-01T01:00:00+00:00",
        "last_verified_at": "2026-01-01T01:00:00+00:00",
        "code_version": "test",
        "point_in_time_verified": False,
        "note": "",
        "extra": {},
    }
    fields.update(overrides)
    return fields


def _publish(store: DatasetStore, frame: pd.DataFrame, *, previous=None, phase_hook=None):
    return store.publish(
        frame=frame, columns=DATASET.columns, time_column=DATASET.time_column,
        manifest_fields=_fields(), previous=previous, phase_hook=phase_hook,
    )


def test_publish_then_read_roundtrip(cache_root):
    """发布 -> 读取回路：manifest 指向的内容可按 schema 正确读回。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        manifest = _publish(store, _frame(5))
    assert manifest.rows == 5
    readback = store.read_frame(manifest, verify_content=True)
    assert len(readback) == 5
    assert list(readback.columns) == list(DATASET.value_columns)


def test_first_publish_failure_leaves_no_official_data(cache_root):
    """
    首次初始化失败：不得留下任何冒充正式版本的文件。

    这里在“内容已写、manifest 尚未替换”的同步点抛错——最容易留下孤儿文件的位置。
    """
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)

    def boom(phase: str) -> None:
        if phase == PHASE_BEFORE_MANIFEST_REPLACE:
            raise RuntimeError("模拟替换前失败")

    with store.lock():
        with pytest.raises(RuntimeError, match="模拟替换前失败"):
            _publish(store, _frame(5), phase_hook=boom)

    assert store.read_manifest() is None, "没有 manifest 就没有正式版本"
    tmp_leftovers = list(store.dataset_dir.glob(".manifest-tmp-*"))
    assert tmp_leftovers == [], "临时 manifest 必须被清理"
    # 内容文件可能已落盘，但它不是正式版本——没有 manifest 指向它。
    assert not (store.dataset_dir / MANIFEST_FILENAME).exists()


def test_content_write_failure_keeps_previous_official_state(cache_root):
    """写内容阶段失败：旧 manifest 与旧内容都必须不变。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        first = _publish(store, _frame(5))
    old_manifest_bytes = store.manifest_path.read_bytes()
    old_content_hash = sha256_file(store.content_path(first.content_file))

    def boom(phase: str) -> None:
        if phase == PHASE_BEFORE_CONTENT_WRITE:
            raise OSError("模拟磁盘写入失败")

    with store.lock():
        with pytest.raises(OSError, match="模拟磁盘写入失败"):
            _publish(store, _frame(6), previous=first, phase_hook=boom)

    assert store.manifest_path.read_bytes() == old_manifest_bytes
    assert sha256_file(store.content_path(first.content_file)) == old_content_hash


def test_manifest_write_failure_keeps_previous_official_state(cache_root):
    """写 manifest 阶段失败：旧正式状态不变，且不留临时文件。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        first = _publish(store, _frame(5))
    old_manifest_bytes = store.manifest_path.read_bytes()

    def boom(phase: str) -> None:
        if phase == PHASE_AFTER_CONTENT_WRITE:
            raise RuntimeError("模拟 manifest 写入失败")

    with store.lock():
        with pytest.raises(RuntimeError, match="模拟 manifest 写入失败"):
            _publish(store, _frame(6), previous=first, phase_hook=boom)

    assert store.manifest_path.read_bytes() == old_manifest_bytes
    assert list(store.dataset_dir.glob(".manifest-tmp-*")) == []


def test_temp_files_are_uniquely_named(cache_root):
    """
    临时文件名必须独立，不共用固定 `.tmp`。

    断言方式：在同步点把当时存在的临时文件名收集下来，两次发布的名字不能相同。
    """
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    seen = []

    def collect(phase: str) -> None:
        if phase == PHASE_BEFORE_MANIFEST_REPLACE:
            seen.extend(p.name for p in store.dataset_dir.glob(".manifest-tmp-*"))

    with store.lock():
        first = _publish(store, _frame(5), phase_hook=collect)
        _publish(store, _frame(6), previous=first, phase_hook=collect)
    assert len(seen) == 2
    assert seen[0] != seen[1], "两次发布不得共用同一个临时文件名"


def test_corrupted_content_file_is_rejected_on_read(cache_root):
    """
    缓存损坏必须拒绝，不能继续按文件名信任。

    这里保持文件名（内含哈希）不变、篡改字节——正是“只看文件名”会漏掉的情形。
    """
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        manifest = _publish(store, _frame(5))
    path = store.content_path(manifest.content_file)
    payload = path.read_bytes().replace(b"30000.0", b"39000.0", 1)
    path.write_bytes(payload)

    with pytest.raises(CacheCorruptionError, match="内容文件(大小|哈希)与 manifest 不符"):
        store.read_frame(manifest, verify_content=True)


def test_existing_same_name_content_with_different_bytes_is_rejected(cache_root):
    """
    已存在同哈希文件名但字节不符：核验失败必须拒绝，绝不原地覆盖。
    """
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    frame = _frame(5)
    with store.lock():
        manifest = _publish(store, frame)
    # 篡改成等长但不同内容的字节，保证不是“大小不符”而是真正的哈希冲突检测。
    path = store.content_path(manifest.content_file)
    tampered = path.read_bytes().replace(b"30000.0", b"39000.0", 1)
    assert len(tampered) == path.stat().st_size
    path.write_bytes(tampered)

    with store.lock():
        with pytest.raises(CacheCorruptionError, match="缓存已损坏"):
            _publish(store, frame, previous=manifest)


def test_missing_content_file_is_rejected(cache_root):
    """manifest 指向的内容文件缺失：读取必须拒绝，不能返回空数据。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        manifest = _publish(store, _frame(5))
    store.content_path(manifest.content_file).unlink()
    with pytest.raises(CacheCorruptionError, match="内容文件缺失"):
        store.read_frame(manifest)


def test_manifest_keeps_previous_content_file_reference(cache_root):
    """保留必要的上一份：manifest 记录 `previous_content_file`，但不承诺永久历史。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    with store.lock():
        first = _publish(store, _frame(5))
        second = _publish(store, _frame(6), previous=first)
    assert second.previous_content_file == first.content_file
    assert second.content_file != first.content_file
    assert store.content_path(first.content_file).exists()


def test_unchanged_data_reuses_content_file(cache_root):
    """数据不变可复用内容文件；只更新核对元信息，不产生第二份相同内容。"""
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    frame = _frame(5)
    with store.lock():
        first = _publish(store, frame)
        second = store.publish(
            frame=frame, columns=DATASET.columns, time_column=DATASET.time_column,
            manifest_fields=_fields(last_verified_at="2026-01-01T02:00:00+00:00"),
            previous=first,
        )
    assert second.content_sha256 == first.content_sha256
    assert len(list(store.dataset_dir.glob("data-*.csv"))) == 1
    assert second.last_verified_at != first.last_verified_at
