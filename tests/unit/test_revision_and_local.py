# -*- coding: utf-8 -*-
"""
修订与转换、本地导入与低频来源。

覆盖 Issue #5 §4.3 / §4.4 与“公共流程未硬编码成行情”的证明：
    - 同主键相同 / 不同内容；历史缩短必须保留旧正式数据并报错。
    - 本地导入不联网、不修改用户原文件、记录哈希。
    - 非规则序列不被强加分钟新鲜度；来源日历未验证时严格请求必须拒绝。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from rbt_fin.contracts import (
    COVERAGE_COMPLETE,
    COVERAGE_UNVERIFIED,
    MODE_FORCE,
    MODE_OFFLINE,
    REVISION_ACCEPT_LATEST,
    DataRequest,
)
from rbt_fin.errors import CoverageError, RevisionConflictError
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_klines_dataset, build_local_csv_dataset
from rbt_fin.storage import DatasetStore
from rbt_fin.transforms import pct_change_keep_base

from tests.conftest import FakeResponse, make_client, minute_klines

NOW = pd.Timestamp("2026-01-01 02:00", tz="UTC")


def _request(start: str, end: str, **kwargs) -> DataRequest:
    return DataRequest(
        start=pd.Timestamp(start, tz="UTC"), end_exclusive=pd.Timestamp(end, tz="UTC"), **kwargs
    )


def _kline_handler(rows):
    by_start = {row[0]: row for row in rows}

    def handler(url, params):
        wanted = [row for ms, row in sorted(by_start.items()) if ms >= params["startTime"]]
        return FakeResponse(status_code=200, payload=wanted[: params["limit"]])

    return handler


# ----------------------------- 修订 -----------------------------

def test_same_key_same_content_is_a_noop(cache_root):
    """同主键相同内容：重复取到也只是去重，不构成修订。"""
    dataset = build_klines_dataset()
    rows = minute_klines(start="2026-01-01 00:00", count=5)
    client, _ = make_client(_kline_handler(rows), max_requests=20)
    first = acquire_dataset(
        dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
        store_root=cache_root, now_utc=NOW, http=client,
    )
    client2, _ = make_client(_kline_handler(rows), max_requests=20)
    second = acquire_dataset(
        dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05", mode=MODE_FORCE),
        store_root=cache_root, now_utc=NOW + dt.timedelta(minutes=1), http=client2,
    )
    assert second.content_sha256 == first.content_sha256
    assert second.degradations == ()


def test_same_key_different_content_is_rejected_by_default(cache_root):
    """
    同主键不同内容：默认拒绝，旧正式数据保持不变。

    这是本次相对旧实现最重要的行为变更之一：未经解释的行情冲突不得覆盖正式数据。
    """
    dataset = build_klines_dataset()
    client, _ = make_client(_kline_handler(minute_klines(start="2026-01-01 00:00", count=5)))
    first = acquire_dataset(
        dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
        store_root=cache_root, now_utc=NOW, http=client,
    )

    revised = minute_klines(start="2026-01-01 00:00", count=5, base_price=31000.0)
    client2, _ = make_client(_kline_handler(revised), max_requests=20)
    with pytest.raises(RevisionConflictError, match="声明为不可修订"):
        acquire_dataset(
            dataset=dataset,
            request=_request("2026-01-01 00:00", "2026-01-01 00:05", mode=MODE_FORCE),
            store_root=cache_root, now_utc=NOW + dt.timedelta(minutes=1), http=client2,
        )
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    assert store.read_manifest().content_sha256 == first.content_sha256


def test_revisable_dataset_records_change_count_and_keeps_previous(cache_root):
    """
    显式声明可修订的来源：允许发布新内容，记录变更数量/范围，并保留上一份内容文件。
    """
    import dataclasses

    dataset = dataclasses.replace(build_klines_dataset(), revision_policy=REVISION_ACCEPT_LATEST)
    client, _ = make_client(_kline_handler(minute_klines(start="2026-01-01 00:00", count=5)))
    first = acquire_dataset(
        dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
        store_root=cache_root, now_utc=NOW, http=client,
    )
    revised = minute_klines(start="2026-01-01 00:00", count=5, base_price=31000.0)
    client2, _ = make_client(_kline_handler(revised), max_requests=20)
    second = acquire_dataset(
        dataset=dataset, request=_request("2026-01-01 00:00", "2026-01-01 00:05", mode=MODE_FORCE),
        store_root=cache_root, now_utc=NOW + dt.timedelta(minutes=1), http=client2,
    )
    assert second.content_sha256 != first.content_sha256
    assert any("修订：5 条记录内容变化" in note for note in second.degradations)
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    # 保留必要的上一份：previous_content_file 指向修订前的内容且文件仍在。
    assert manifest.previous_content_file is not None
    assert first.content_sha256 in manifest.previous_content_file
    assert store.content_path(manifest.previous_content_file).exists()


def test_history_shortening_keeps_old_data_and_raises(cache_root, tmp_path):
    """
    原覆盖范围内记录消失：保留旧正式数据并报错，不能靠行数变少就静默清库。

    用本地 CSV 来源演示：小数据全量重取时最容易出现这种“来源悄悄变短”。
    """
    csv_path = tmp_path / "metric.csv"
    csv_path.write_text(
        "date,value\n2026-01-01T00:00:00Z,1.0\n2026-01-02T00:00:00Z,2.0\n2026-01-03T00:00:00Z,3.0\n",
        encoding="utf-8",
    )
    dataset = build_local_csv_dataset(
        dataset_id="local_metric", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "frequency": "daily", "value_meaning": "metric"},
    )
    request = _request("2026-01-01", "2026-01-04", require_full_coverage=False)
    first = acquire_dataset(
        dataset=dataset, request=request, store_root=cache_root, now_utc=NOW,
        local_inputs={"csv_path": str(csv_path)},
    )
    assert first.coverage.rows == 3

    # 来源缩短：中间一天消失。
    csv_path.write_text(
        "date,value\n2026-01-01T00:00:00Z,1.0\n2026-01-03T00:00:00Z,3.0\n", encoding="utf-8"
    )
    with pytest.raises(RevisionConflictError, match="既有记录在新结果中消失"):
        acquire_dataset(
            dataset=dataset, request=dataclasses_replace_mode(request), store_root=cache_root,
            now_utc=NOW + dt.timedelta(days=1), local_inputs={"csv_path": str(csv_path)},
        )
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    assert store.read_manifest().rows == 3, "旧正式数据必须保持不变"


def dataclasses_replace_mode(request: DataRequest) -> DataRequest:
    """返回同一请求的 force 版本（本地来源需要强制重读文件）。"""
    import dataclasses

    return dataclasses.replace(request, mode=MODE_FORCE)


# ----------------------------- 转换 -----------------------------

def test_pct_change_does_not_fill_zero_for_undefined_cases():
    """
    链上变化率保留基础值；首次记录、零分母不能统一填成零增长。

    手算标准答案：[NaN(首次), NaN(零分母), 1.0]。
    """
    index = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"], tz="UTC", name="date")
    frame = pd.DataFrame({"tx": [5.0, 0.0, 0.0]}, index=index)
    out = pct_change_keep_base(frame, value_columns=["tx"], context="pct")
    assert list(out.columns) == ["tx", "tx_pct"]
    assert out["tx"].tolist() == [5.0, 0.0, 0.0], "必须保留基础值"
    assert pd.isna(out["tx_pct"].iloc[0]), "首次记录没有前值"
    assert out["tx_pct"].iloc[1] == pytest.approx(-1.0), "5 -> 0 是 -100%，有定义"
    assert pd.isna(out["tx_pct"].iloc[2]), "零分母必须标记为未定义，不能填 0"


# ----------------------------- 本地导入与低频 -----------------------------

def test_local_import_does_not_touch_network_or_modify_source(cache_root, tmp_path):
    """
    本地导入：不联网（http=None 仍可完成）、不修改用户原文件、记录源文件哈希。
    """
    from rbt_fin.storage import sha256_file

    csv_path = tmp_path / "metric.csv"
    csv_path.write_text(
        "date,value\n2026-01-01T00:00:00Z,1.0\n2026-01-02T00:00:00Z,2.0\n", encoding="utf-8"
    )
    before_hash = sha256_file(csv_path)
    before_mtime = csv_path.stat().st_mtime_ns

    dataset = build_local_csv_dataset(
        dataset_id="local_metric", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "frequency": "daily", "value_meaning": "metric"},
    )
    result = acquire_dataset(
        dataset=dataset,
        request=_request("2026-01-01", "2026-01-03", require_full_coverage=False),
        store_root=cache_root, now_utc=NOW, http=None,
        local_inputs={"csv_path": str(csv_path)},
    )
    assert result.coverage.rows == 2
    assert sha256_file(csv_path) == before_hash
    assert csv_path.stat().st_mtime_ns == before_mtime
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    assert before_hash in store.read_manifest().extra["scan_note"]


def test_low_frequency_source_is_not_forced_onto_a_minute_grid(cache_root, tmp_path):
    """
    低频来源不被强加分钟新鲜度：正常非发布日（周末）不会被报成缺失。

    但来源日历未验证 -> 覆盖状态为 unverified，严格请求必须被拒绝。
    """
    csv_path = tmp_path / "macro.csv"
    # 2026-01-03/04 是周六周日，来源正常不发布。
    csv_path.write_text(
        "date,value\n"
        "2026-01-01T00:00:00Z,1.0\n2026-01-02T00:00:00Z,2.0\n2026-01-05T00:00:00Z,3.0\n",
        encoding="utf-8",
    )
    dataset = build_local_csv_dataset(
        dataset_id="macro_series", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "frequency": "business_daily", "value_meaning": "macro"},
    )
    loose = acquire_dataset(
        dataset=dataset,
        request=_request("2026-01-01", "2026-01-06", require_full_coverage=False),
        store_root=cache_root, now_utc=NOW, local_inputs={"csv_path": str(csv_path)},
    )
    assert loose.coverage.status == COVERAGE_UNVERIFIED
    assert loose.coverage.rows == 3
    assert "日历未验证" in loose.coverage.note

    with pytest.raises(CoverageError, match="严格请求要求完整覆盖"):
        acquire_dataset(
            dataset=dataset,
            request=_request("2026-01-01", "2026-01-06", mode=MODE_FORCE, require_full_coverage=True),
            store_root=cache_root, now_utc=NOW, local_inputs={"csv_path": str(csv_path)},
        )


def test_local_dataset_offline_mode_reads_cache_without_source_file(cache_root, tmp_path):
    """
    缓存复用为零网络/零文件读取：offline 模式下即使源文件被删除也能读回正式数据。
    """
    csv_path = tmp_path / "metric.csv"
    csv_path.write_text("date,value\n2026-01-01T00:00:00Z,1.0\n", encoding="utf-8")
    dataset = build_local_csv_dataset(
        dataset_id="local_metric", time_column="date", value_columns=("value",),
        identity={"source": "local_file", "frequency": "daily", "value_meaning": "metric"},
    )
    request = _request("2026-01-01", "2026-01-02", require_full_coverage=False)
    acquire_dataset(
        dataset=dataset, request=request, store_root=cache_root, now_utc=NOW,
        local_inputs={"csv_path": str(csv_path)},
    )
    csv_path.unlink()
    import dataclasses

    offline = acquire_dataset(
        dataset=dataset, request=dataclasses.replace(request, mode=MODE_OFFLINE),
        store_root=cache_root, now_utc=NOW, local_inputs={},
    )
    assert offline.used_cache is True
    assert offline.coverage.rows == 1
