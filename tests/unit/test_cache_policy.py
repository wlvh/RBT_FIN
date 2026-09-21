# -*- coding: utf-8 -*-
"""
低频与缓存：反例 C、零网络请求、offline、刷新失败降级、核对时间不被失败推进。

反例 C（Issue #5 §8）：
    刷新失败时，允许陈旧且覆盖够用的研究请求可以明确降级；
    **同一缓存**不满足严格请求时必须拒绝。
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from rbt_fin.contracts import (
    COVERAGE_COMPLETE,
    MODE_AUTO,
    MODE_FORCE,
    MODE_OFFLINE,
    DataRequest,
)
from rbt_fin.errors import (
    HttpStatusError,
    OfflineUnavailableError,
    StaleCacheError,
)
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_klines_dataset
from rbt_fin.storage import DatasetStore

from tests.conftest import FakeResponse, make_client, minute_klines, to_ms

DATASET = build_klines_dataset()
T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def _ok_handler(count: int = 5, start: str = "2026-01-01 00:00"):
    """返回覆盖 `[start, start+count)` 的完整分钟数据的端点。"""
    rows = minute_klines(start=start, count=count)
    by_start = {row[0]: row for row in rows}

    def handler(url, params):
        wanted = [row for ms, row in sorted(by_start.items()) if ms >= params["startTime"]]
        return FakeResponse(status_code=200, payload=wanted[: params["limit"]])

    return handler


def _request(start: str, end: str, **kwargs) -> DataRequest:
    return DataRequest(
        start=pd.Timestamp(start, tz="UTC"), end_exclusive=pd.Timestamp(end, tz="UTC"), **kwargs
    )


def _seed_cache(cache_root, *, now: pd.Timestamp, count: int = 5) -> None:
    """预置一份覆盖 `[00:00, 00:0count)` 的完整缓存。"""
    client, _ = make_client(_ok_handler(count=count), max_requests=20)
    result = acquire_dataset(
        dataset=DATASET,
        request=_request("2026-01-01 00:00", f"2026-01-01 00:0{count}"),
        store_root=cache_root, now_utc=now, http=client,
    )
    assert result.coverage.status == COVERAGE_COMPLETE
    assert result.used_cache is False


def test_cache_hit_within_refresh_window_makes_zero_requests(cache_root):
    """未到重新检查来源的时机且覆盖满足请求 -> 零网络请求。"""
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    client, session = make_client(_ok_handler(), max_requests=20)
    result = acquire_dataset(
        dataset=DATASET, request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
        store_root=cache_root, now_utc=seeded_at + dt.timedelta(seconds=10), http=client,
    )
    assert result.used_cache is True
    assert session.calls == [], "缓存命中时不得发出任何请求"
    assert result.coverage.status == COVERAGE_COMPLETE


def test_closed_historical_range_needs_no_network_even_after_refresh_interval(cache_root):
    """
    固定历史区间可离线使用，不要求追到现在。

    该数据集声明为不可修订，且请求范围已完整 -> 即使早已超过刷新间隔也无需联网。
    """
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    client, session = make_client(_ok_handler(), max_requests=20)
    result = acquire_dataset(
        dataset=DATASET, request=_request("2026-01-01 00:00", "2026-01-01 00:05"),
        store_root=cache_root, now_utc=seeded_at + dt.timedelta(days=30), http=client,
    )
    assert result.used_cache is True
    assert session.calls == []


def test_offline_mode_makes_zero_requests_and_refuses_when_insufficient(cache_root):
    """
    显式 offline：零网络请求；缓存不足以满足请求时必须拒绝，而不是返回半成品。
    """
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    result = acquire_dataset(
        dataset=DATASET, request=_request("2026-01-01 00:00", "2026-01-01 00:05", mode=MODE_OFFLINE),
        store_root=cache_root, now_utc=seeded_at, http=None,
    )
    assert result.used_cache is True
    assert "offline" in result.degradations[0]

    # 请求范围超出缓存：offline 无法补齐 -> 拒绝。
    with pytest.raises(OfflineUnavailableError, match="offline 模式下缓存不满足请求"):
        acquire_dataset(
            dataset=DATASET,
            request=_request("2026-01-01 00:00", "2026-01-01 00:09", mode=MODE_OFFLINE),
            store_root=cache_root, now_utc=seeded_at, http=None,
        )


def test_counterexample_c_same_cache_degrades_for_research_but_rejects_strict(cache_root):
    """
    反例 C：同一份缓存，在刷新失败时的两种命运。

    - 研究请求（允许陈旧、接受当前覆盖）：明确降级，返回旧数据并标注 refresh_failed。
    - 严格请求（不接受降级）：必须失败。

    另外断言：降级路径**不推进** `last_verified_at`——失败重试不能伪装成一次成功核对。
    """
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    verified_before = store.read_manifest().last_verified_at

    def failing(url, params):
        return FakeResponse(status_code=503, payload={"code": -1, "msg": "unavailable"})

    later = seeded_at + dt.timedelta(hours=1)
    # 请求范围比缓存更宽，保证一定会尝试刷新。
    research = _request(
        "2026-01-01 00:00", "2026-01-01 00:07",
        mode=MODE_AUTO, accept_stale_on_refresh_failure=True,
        max_staleness=dt.timedelta(hours=6), require_full_coverage=False,
    )
    client, _ = make_client(failing, max_requests=20)
    degraded = acquire_dataset(
        dataset=DATASET, request=research, store_root=cache_root, now_utc=later, http=client,
    )
    assert degraded.used_cache is True
    assert degraded.refresh_failed is True
    assert degraded.errors and "HttpStatusError" in degraded.errors[0]
    assert "刷新失败" in degraded.degradations[0]
    assert store.read_manifest().last_verified_at == verified_before, (
        "刷新失败不得推进 last_verified_at"
    )

    # 同一份缓存 + 同样的刷新失败，但这次是严格请求 -> 必须拒绝。
    strict = _request("2026-01-01 00:00", "2026-01-01 00:07", mode=MODE_AUTO)
    client2, _ = make_client(failing, max_requests=20)
    with pytest.raises(HttpStatusError):
        acquire_dataset(
            dataset=DATASET, request=strict, store_root=cache_root, now_utc=later, http=client2,
        )


def test_stale_beyond_limit_is_rejected_even_when_degradation_allowed(cache_root):
    """
    允许降级 != 永远可用旧数据：超过陈旧上限必须拒绝。
    """
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    def failing(url, params):
        return FakeResponse(status_code=503, payload={"code": -1, "msg": "unavailable"})

    client, _ = make_client(failing, max_requests=20)
    with pytest.raises(StaleCacheError, match="超过允许上限"):
        acquire_dataset(
            dataset=DATASET,
            request=_request(
                "2026-01-01 00:00", "2026-01-01 00:07",
                accept_stale_on_refresh_failure=True,
                max_staleness=dt.timedelta(minutes=5), require_full_coverage=False,
            ),
            store_root=cache_root, now_utc=seeded_at + dt.timedelta(hours=3), http=client,
        )


def test_fresh_cache_file_alone_does_not_pass_a_wider_strict_request(cache_root):
    """
    严格任务需要的尾部/范围未覆盖：拒绝，不能凭“缓存文件刚创建”就放行。

    这里让刷新也拿不到尾部数据，断言失败信息指向尾部缺失而不是“缓存很新所以可用”。
    """
    from rbt_fin.errors import CoverageError

    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    def only_old(url, params):
        return FakeResponse(status_code=200, payload=[])

    client, _ = make_client(only_old, max_requests=20)
    with pytest.raises(CoverageError, match="tail="):
        acquire_dataset(
            dataset=DATASET, request=_request("2026-01-01 00:00", "2026-01-01 00:08"),
            store_root=cache_root, now_utc=seeded_at + dt.timedelta(seconds=1), http=client,
        )


def test_previous_check_range_is_not_reused_for_a_wider_request(cache_root):
    """
    后续读取不能把旧检查范围当作新请求已通过。

    缓存的覆盖检查只覆盖 `[00:00, 00:05)`；当请求扩大到 `[00:00, 00:08)` 时，
    多出来的部分**从未被扫描过**，必须重新采集而不是继承旧的 scan_complete。
    """
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    client, session = make_client(_ok_handler(count=8), max_requests=20)
    result = acquire_dataset(
        dataset=DATASET, request=_request("2026-01-01 00:00", "2026-01-01 00:08"),
        store_root=cache_root, now_utc=seeded_at + dt.timedelta(seconds=1), http=client,
    )
    assert session.calls, "扩大范围后必须真的去采集未扫描过的部分"
    assert result.used_cache is False
    assert result.coverage.status == COVERAGE_COMPLETE
    assert result.coverage.rows == 8
    # 只补了缺失段：请求的 startTime 应当从 00:05 开始，而不是从头重下。
    assert session.calls[0][1]["startTime"] == to_ms("2026-01-01 00:05")


def test_force_mode_refreshes_even_when_cache_is_sufficient(cache_root):
    """`force` 模式必须真的重新查来源，哪怕缓存完全够用。"""
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    _seed_cache(cache_root, now=seeded_at)

    client, session = make_client(_ok_handler(), max_requests=20)
    result = acquire_dataset(
        dataset=DATASET,
        request=_request("2026-01-01 00:00", "2026-01-01 00:05", mode=MODE_FORCE),
        store_root=cache_root, now_utc=seeded_at + dt.timedelta(seconds=1), http=client,
    )
    assert result.used_cache is False
    assert session.calls, "force 模式必须真的重新向来源取数，而不是退化成缓存命中"
    assert session.calls[0][1]["startTime"] == to_ms("2026-01-01 00:00")

    # 数据没变 -> 复用同一个内容文件，只更新核对元信息。
    store = DatasetStore(root=cache_root, dataset_id=DATASET.dataset_id)
    manifest = store.read_manifest()
    content_files = sorted(p.name for p in (cache_root / DATASET.dataset_id).glob("data-*.csv"))
    assert len(content_files) == 1, "内容未变时必须复用同一个内容文件"
    assert manifest.last_verified_at != seeded_at.isoformat(), "成功核对后必须推进核对时间"
