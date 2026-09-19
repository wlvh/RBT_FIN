# -*- coding: utf-8 -*-
"""
执行者自审（RBT_FIN#5 §0.2 / trading#105 §9，2026-09-19）发现的包内缺陷：先红后绿的反例 + 正向对照 + 覆盖缺口补测。

功能或目的：
    R-1 存储往返：写入用 repr 最短文本、读回必须逐位还原（全精度浮点来源第二次刷新不得被误判为修订，内容哈希不漂移）；
    R-2 凭据脱敏：连接类异常文本含带 api_key 的 URL，不得进入 HttpStatusError / 任务失败清单；
    R-3 修订比对：合同允许的 NaN（首行/零分母 _pct）两边同为 NaN 不是内容变化；
    R-4 daily_series 并入已缓存历史：滑动窗口来源不因窗口前移而"历史消失"，窄窗口请求不裁掉更早历史；
    R-5 提交后失败：manifest 已替换后的失败是 PublishedUnverifiedError，不得按"刷新失败、沿用旧缓存"报告；
    R-6/R-9 兼容入口：本地项可经 local_inputs 采集；--limit 抽样恢复基线的自然日历语义（缺失日计 NA）；
    R-12 去重：同主键 NaN 与数值是冲突；R-14 FRED "." 计入采集说明；
    覆盖缺口：PIT 拒绝、CryptoCompare 业务错误、真实重叠分页去重、资金费率两处"游标不推进"、日内 last 乱序拒绝。
    全部离线：HTTP 会话为受控替身，不发起任何 socket 连接。

调用关系：
    pytest tests/unit → 本文件；依赖 tests/conftest.py 的 FakeResponse / make_client / minute_klines / to_ms。
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import urllib3

from rbt_fin import acquire_dataset, run_task
from rbt_fin.contracts import MODE_AUTO, DataRequest
from rbt_fin.errors import (
    BusinessStatusError, ContractError, CoverageError, HttpStatusError, PaginationError, RevisionConflictError, TimeContractError,
)
from rbt_fin.http import HttpClient, RequestBudget, build_session
from rbt_fin.registry import build_klines_dataset, get_dataset
from rbt_fin.sources.binance_funding import fetch_funding_records
from rbt_fin.sources.bybit_funding import fetch_bybit_funding
from rbt_fin.sources.daily_series import CRYPTOCOMPARE_BASE_FIELDS, extract_fred_observations
from rbt_fin.storage import PHASE_AFTER_MANIFEST_REPLACE, DatasetStore, csv_bytes_to_frame, frame_to_csv_bytes, sha256_bytes
from rbt_fin.transforms import daily_last_by_utc_date
from rbt_fin.validation import dedupe_by_time_primary_key
from tests.conftest import FakeResponse, funding_row, make_client, minute_klines, to_ms

REPO_ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-08", tz="UTC")
NOW = pd.Timestamp("2026-01-10", tz="UTC")
DAYS = [START + pd.Timedelta(days=i) for i in range(6)]
SECRET = "SECRET-FRED-KEY-0123456789"
FULL_PRECISION_VALUES = [128342553678.36615, 128401234567.12347, 128377891234.56789, 128455123456.78901,
                         128499876543.21098, 128512345678.90123]


@pytest.fixture
def cache_root(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    return root


def _compat_module():
    spec = importlib.util.spec_from_file_location("compat_entry_under_test", REPO_ROOT / "data_layer_unified_opt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cc_payload(days, series):
    return {"Data": {"Data": [{"time": int(d.value // 10**9), **{name: series[i] for name in CRYPTOCOMPARE_BASE_FIELDS}}
                              for i, d in enumerate(days)]}}


def _defillama_payload(days, values):
    return [{"date": str(int(d.value // 10**9)), "totalCirculatingUSD": {"peggedUSD": v}} for d, v in zip(days, values)]


# ---------- R-1 存储往返 ----------

def test_storage_roundtrip_is_exact_and_content_hash_is_stable():
    """写入 → 读回 → 再写入：值逐位相等、字节（内容哈希）不变（此前默认解析器让 1/12 等值往返后不等）。"""
    index = pd.DatetimeIndex(DAYS, tz="UTC", name="date")
    frame = pd.DataFrame({"v": FULL_PRECISION_VALUES + []}, index=index)
    frame.loc[DAYS[0], "v"] = 1.0 / 12.0
    payload = frame_to_csv_bytes(frame, columns=("date", "v"), time_column="date")
    back = csv_bytes_to_frame(payload, columns=("date", "v"), time_column="date")
    assert (back["v"].to_numpy() == frame["v"].to_numpy()).all()
    assert sha256_bytes(frame_to_csv_bytes(back, columns=("date", "v"), time_column="date")) == sha256_bytes(payload)


def test_unchanged_full_precision_source_refreshes_without_revision_conflict(cache_root):
    """DefiLlama 风格 17 位有效数字：到期刷新且来源载荷完全相同 → 无修订冲突、内容哈希不变。"""
    dataset = get_dataset("defillama_stablecoins_total")
    payload = _defillama_payload(DAYS, FULL_PRECISION_VALUES)
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=payload), max_requests=5)
    first = acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW, http=client)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=payload), max_requests=5)
    second = acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)
    assert second.content_sha256 == first.content_sha256 and second.refresh_failed is False


# ---------- R-2 凭据脱敏 ----------

class _FailingAdapter(requests.adapters.BaseAdapter):
    """按 requests.HTTPAdapter 的真实异常链构造：MaxRetryError(url 含查询串) → ConnectionError。"""

    def send(self, request, **kwargs):
        err = urllib3.exceptions.MaxRetryError(None, request.path_url, urllib3.exceptions.NewConnectionError(None, "simulated refused"))
        raise requests.exceptions.ConnectionError(err, request=request)

    def close(self):
        return None


def test_redact_credentials_masks_query_values_only():
    from rbt_fin.http import redact_credentials   # 新增符号：延迟导入，让本文件在旧 wheel 上也能逐条报红
    text = "https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=abc123&file_type=json token=zz"
    assert redact_credentials(text) == "https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=<redacted>&file_type=json token=<redacted>"


def test_query_credential_never_enters_error_text_or_task_failures(cache_root):
    session = build_session()
    session.mount("https://", _FailingAdapter())
    client = HttpClient(session=session, budget=RequestBudget(max_requests=5, max_retries=0), sleep=lambda s: None)
    dataset = get_dataset("macro_us10y")
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)
    with pytest.raises(HttpStatusError) as info:
        acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW, http=client, credentials={"FRED_API_KEY": SECRET})
    assert SECRET not in str(info.value) and "api_key=<redacted>" in str(info.value)
    outcome = run_task(items=[(dataset, request)], store_root=cache_root, now_utc=NOW, http=client, credentials={"FRED_API_KEY": SECRET})
    assert SECRET not in outcome.failures[dataset.dataset_id]


# ---------- R-3 / R-4 修订比对与并入缓存 ----------

def test_identical_cryptocompare_payload_refreshes_without_conflict(cache_root):
    """首行 _pct=NaN 是合同定义；到期刷新载荷相同 → 不是修订冲突。"""
    dataset = get_dataset("cryptocompare_btc_chain_pct")
    payload = _cc_payload(DAYS[:3], [5, 0, 0])
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)
    creds = {"CRYPTOCOMPARE_API_KEY": "fake"}
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=payload), max_requests=5)
    first = acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW, http=client, credentials=creds)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=payload), max_requests=5)
    second = acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2, credentials=creds)
    assert second.content_sha256 == first.content_sha256 and second.coverage.rows == 3


def test_sliding_window_source_keeps_earlier_history_and_stays_refreshable(cache_root):
    """
    来源窗口每天前移（limit=2000 滑动窗口）：第二次取回 D1..D5，正式内容必须是 D0..D5（并入缓存），
    且 D1 的 _pct 仍由 D0 算出（不会因失去前值变成 NaN 而"冲突"）。
    """
    dataset = get_dataset("cryptocompare_btc_chain_pct")
    creds = {"CRYPTOCOMPARE_API_KEY": "fake"}
    values = [10, 20, 40, 80, 160, 320]
    request = DataRequest(start=DAYS[1], end_exclusive=END, require_full_coverage=False)
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_cc_payload(DAYS[0:5], values[0:5])), max_requests=5)
    acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW, http=client, credentials=creds)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_cc_payload(DAYS[1:6], values[1:6])), max_requests=5)
    result = acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2, credentials=creds)
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    assert manifest.rows == 6 and pd.Timestamp(manifest.actual_start) == DAYS[0] and pd.Timestamp(manifest.actual_end) == DAYS[5]
    frame = store.read_frame(manifest, verify_content=True)
    pct_columns = [c for c in frame.columns if c.endswith("_pct")]
    assert frame.loc[DAYS[1], pct_columns[0]] == pytest.approx(1.0) and np.isnan(frame.loc[DAYS[0], pct_columns[0]])
    assert result.refresh_failed is False and result.errors == ()


def test_narrow_request_does_not_drop_earlier_cached_history(cache_root):
    dataset = get_dataset("cryptocompare_btc_chain_pct")
    creds = {"CRYPTOCOMPARE_API_KEY": "fake"}
    values = [10, 20, 40, 80, 160, 320]
    wide = DataRequest(start=DAYS[0], end_exclusive=END, require_full_coverage=False)
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_cc_payload(DAYS[0:5], values[0:5])), max_requests=5)
    acquire_dataset(dataset=dataset, request=wide, store_root=cache_root, now_utc=NOW, http=client, credentials=creds)
    narrow = DataRequest(start=DAYS[2], end_exclusive=END, require_full_coverage=False)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_cc_payload(DAYS[1:6], values[1:6])), max_requests=5)
    acquire_dataset(dataset=dataset, request=narrow, store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2, credentials=creds)
    manifest = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest()
    assert manifest.rows == 6 and pd.Timestamp(manifest.actual_start) == DAYS[0]


def test_real_revision_in_window_is_still_rejected_after_merge(cache_root):
    """正向对照：并入缓存不放行真正的修订——请求窗口内同主键数值变化仍是 RevisionConflictError。"""
    dataset = get_dataset("defillama_stablecoins_total")
    request = DataRequest(start=START, end_exclusive=END, require_full_coverage=False)
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_defillama_payload(DAYS, [1, 2, 3, 4, 5, 6])), max_requests=5)
    acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW, http=client)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_defillama_payload(DAYS, [1, 2, 99, 4, 5, 6])), max_requests=5)
    with pytest.raises(RevisionConflictError):
        acquire_dataset(dataset=dataset, request=request, store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)


# ---------- R-5 提交后失败 ----------

def _kline_handler(count):
    rows = minute_klines(start="2026-01-01 00:00", count=count)

    def handler(url, params):
        lo, hi = params["startTime"], params["endTime"]
        return FakeResponse(status_code=200, payload=[r for r in rows if lo <= r[0] <= hi][: params["limit"]])
    return handler


def test_post_commit_failure_is_reported_as_published_unverified(cache_root):
    """manifest 已原子替换后目录 fsync 失败：新状态已可见——必须抛 PublishedUnverifiedError，不能退回"旧缓存仍有效"。"""
    dataset = build_klines_dataset()
    seeded_at = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    client, _ = make_client(_kline_handler(5), max_requests=20)
    old = acquire_dataset(dataset=dataset, request=DataRequest(start=pd.Timestamp("2026-01-01 00:00", tz="UTC"), end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC")),
                          store_root=cache_root, now_utc=seeded_at, http=client)

    def post_commit_failure(phase):
        if phase == PHASE_AFTER_MANIFEST_REPLACE:
            raise OSError("模拟提交后目录 fsync 失败")

    from rbt_fin.errors import PublishedUnverifiedError   # 新增符号：延迟导入
    client2, _ = make_client(_kline_handler(8), max_requests=20)
    with pytest.raises(PublishedUnverifiedError, match="持久性未确认"):
        acquire_dataset(dataset=dataset,
                        request=DataRequest(start=pd.Timestamp("2026-01-01 00:00", tz="UTC"), end_exclusive=pd.Timestamp("2026-01-01 00:08", tz="UTC"),
                                            accept_stale_on_refresh_failure=True, max_staleness=dt.timedelta(hours=6), require_full_coverage=False),
                        store_root=cache_root, now_utc=seeded_at + dt.timedelta(hours=1), http=client2, phase_hook=post_commit_failure)
    on_disk = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest()
    assert on_disk.rows == 8 and on_disk.content_sha256 != old.content_sha256


# ---------- R-6 / R-9 兼容入口 ----------

def test_compat_entry_acquires_local_dataset_when_csv_provided(tmp_path):
    module = _compat_module()
    csv_path = tmp_path / "crypto_metrics_daily.csv"
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in DAYS[:3]], "value": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)
    frame = module.run_overview(cache_root=tmp_path / "cache", start=START, end_exclusive=END, limit=300, mode=MODE_AUTO,
                                http_factory=lambda: None, credentials={}, now_utc=NOW, registered_ids=["binance_metrics_local"],
                                local_inputs={"binance_metrics_local": {"csv_path": csv_path}})
    row = frame.iloc[0]
    assert row["na_ratio"] is not None and not pd.isna(row["na_ratio"]) and "error=" not in row["remark"]


def test_limit_sampling_follows_baseline_calendar_semantics(tmp_path):
    """基线 sample_first_n_days(n=10) + na_cells_info：自然日历 10 行左连接 → sample_rows=10, na_cells=5, na_ratio=0.5。"""
    module = _compat_module()
    dataset = get_dataset("blockchain_hash_rate")
    index = pd.DatetimeIndex([START + pd.Timedelta(days=2 * i) for i in range(5)], tz="UTC", name="date")
    frame = pd.DataFrame({"value": [1.0] * 5}, index=index)
    row = module.summarize_dataset(registered_id="blockchain_hash_rate", dataset=dataset, frame=frame, limit=10, error=None, end_exclusive=END)
    assert (row["sample_rows"], row["na_cells"], row["na_ratio"]) == (10, 5, 0.5)
    assert row["sample_start"] == START.date().isoformat() and row["sample_end"] == (START + pd.Timedelta(days=9)).date().isoformat()


def test_limit_sampling_folds_intraday_records_to_daily_mean(tmp_path):
    """日内多条（资金费率 8h）折成一行：与基线 to_daily 的 mean 口径一致。"""
    module = _compat_module()
    dataset = get_dataset("binance_funding_btcusdt_raw")
    index = pd.DatetimeIndex([START + pd.Timedelta(hours=8 * i) for i in range(6)], tz="UTC", name="funding_time")
    frame = pd.DataFrame({"funding_rate": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]}, index=index)
    row = module.summarize_dataset(registered_id="binance_funding_btcusdt", dataset=dataset, frame=frame, limit=2, error=None, end_exclusive=END)
    assert (row["sample_rows"], row["na_cells"], row["na_ratio"]) == (2, 0, 0.0)


# ---------- R-12 / R-14 ----------

def test_dedupe_treats_nan_vs_value_on_same_key_as_conflict():
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({"value": [np.nan, 5.0]}, index=pd.DatetimeIndex([ts, ts], tz="UTC", name="date"))
    with pytest.raises(ContractError, match="内容不同"):
        dedupe_by_time_primary_key(frame, value_columns=("value",), context="t")


def test_fred_dot_values_are_counted_in_scan_note_not_silently_dropped(cache_root):
    dataset = get_dataset("macro_us10y")
    payload = {"observations": [{"date": d.strftime("%Y-%m-%d"), "value": "." if i == 1 else f"{4.0 + i}"} for i, d in enumerate(DAYS[:3])]}
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=payload), max_requests=5)
    result = acquire_dataset(dataset=dataset, request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False),
                             store_root=cache_root, now_utc=NOW, http=client, credentials={"FRED_API_KEY": "x"})
    assert result.coverage.rows == 2
    note = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest().extra["scan_note"]
    assert "字段缺失 1 条" in note and DAYS[1].date().isoformat() in note
    assert [values["value"] for _, values in extract_fred_observations(payload, context="t")] == ["4.0", None, "6.0"]


# ---------- 覆盖缺口补测 ----------

def test_point_in_time_request_is_rejected_for_unverified_dataset(cache_root):
    with pytest.raises(CoverageError, match="point-in-time"):
        acquire_dataset(dataset=get_dataset("macro_us10y"),
                        request=DataRequest(start=START, end_exclusive=END, require_full_coverage=False, require_point_in_time=True),
                        store_root=cache_root, now_utc=NOW, http=None, credentials={"FRED_API_KEY": "x"})


def test_cryptocompare_business_error_is_detected():
    from rbt_fin.sources.daily_series import extract_cryptocompare_chain
    with pytest.raises(BusinessStatusError, match="业务错误"):
        extract_cryptocompare_chain({"Response": "Error", "Message": "api key invalid"}, context="t")


def test_truly_overlapping_kline_pages_are_deduped_without_loss():
    """两页真的共享 00:02（同内容）：去重路径被触发，5 根不重不漏（原有用例的两页并不重叠）。"""
    from rbt_fin.sources.binance_klines import fetch_klines_range
    series = minute_klines(start="2026-01-01 00:00", count=5)
    pages = {to_ms("2026-01-01 00:00"): series[:3], to_ms("2026-01-01 00:03"): series[2:]}
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=pages.get(p["startTime"], [])), max_requests=10)
    frame, complete, _ = fetch_klines_range(http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
                                            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"), end_exclusive=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
                                            limit=3, context="ctl")
    assert len(frame) == 5 and frame.index.is_unique and complete


def test_binance_funding_cursor_not_advancing_with_new_records_is_detected():
    """第二页有未见记录但最晚时间不晚于上一页：不是重复页，仍必须按游标未推进终止。"""
    t = [to_ms("2026-01-01 00:00"), to_ms("2026-01-01 08:00"), to_ms("2026-01-01 16:00")]
    pages = [[funding_row(funding_time_ms=t[1], rate=0.0001), funding_row(funding_time_ms=t[2], rate=0.0002)],
             [funding_row(funding_time_ms=t[0], rate=0.0003)]]
    calls = {"n": 0}

    def handler(url, params):
        page = pages[min(calls["n"], 1)]
        calls["n"] += 1
        return FakeResponse(status_code=200, payload=page)
    client, _ = make_client(handler, max_requests=10)
    with pytest.raises(PaginationError, match="分页游标未推进"):
        fetch_funding_records(http=client, symbol="BTCUSDT", start=START, end_exclusive=END, limit=2, context="ctl")


def test_bybit_cursor_not_advancing_with_new_records_is_detected():
    """向历史翻页：第二页含未见记录但最早时间不早于上一页 → 命中"未向历史推进"分支（不是重复页分支）。"""
    def row(ts_ms, rate):
        return {"symbol": "BTCUSDT", "fundingRate": f"{rate:.8f}", "fundingRateTimestamp": str(ts_ms)}
    t = [to_ms("2026-01-02 00:00"), to_ms("2026-01-03 00:00"), to_ms("2026-01-05 00:00")]
    pages = [[row(t[1], 0.0002), row(t[0], 0.0001)], [row(t[2], 0.0003)]]
    calls = {"n": 0}

    def handler(url, params):
        page = pages[min(calls["n"], 1)]
        calls["n"] += 1
        return FakeResponse(status_code=200, payload={"retCode": 0, "result": {"list": page}})
    client, _ = make_client(handler, max_requests=10)
    with pytest.raises(PaginationError, match="未向历史推进"):
        fetch_bybit_funding(http=client, symbol="BTCUSDT", category="linear", start=START - pd.Timedelta(days=30), end_exclusive=END, limit=2, context="ctl")


def test_daily_last_by_utc_date_takes_last_and_rejects_unsorted_input():
    index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 16:00", "2026-01-02 08:00"], tz="UTC")
    frame = pd.DataFrame({"funding_rate": [0.1, 0.2, 0.3]}, index=index)
    out = daily_last_by_utc_date(frame, value_columns=("funding_rate",), context="t")
    assert list(out["funding_rate"]) == [0.2, 0.3]
    with pytest.raises(TimeContractError, match="升序"):
        daily_last_by_utc_date(frame.iloc[[1, 0, 2]], value_columns=("funding_rate",), context="t")


# ---------- 外部抽查 R12（2026-09-19）：修订检查范围必须覆盖实际发布范围 ----------

def _blockchain_payload(days, values):
    return {"values": [{"x": int(d.value // 10**9), "y": float(v)} for d, v in zip(days, values)]}


def _cache_three_days(cache_root):
    """缓存 1 月 1/2/3 日 = 100/200/300（blockchain_hash_rate：全量响应、非滑动窗口、REVISION_REJECT）。"""
    dataset = get_dataset("blockchain_hash_rate")
    client, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[:3], [100, 200, 300])), max_requests=5)
    first = acquire_dataset(dataset=dataset, request=DataRequest(start=DAYS[0], end_exclusive=DAYS[3], require_full_coverage=False),
                            store_root=cache_root, now_utc=NOW, http=client)
    assert first.coverage.rows == 3
    return dataset, first


def test_out_of_window_revision_is_rejected_before_publish(cache_root):
    """只请求 1 月 3 日，来源全量响应把 1 月 1 日从 100 改成 999：REVISION_REJECT 必须拒绝（此前报告 revision_changed_rows=0 后发布整段）。"""
    dataset, first = _cache_three_days(cache_root)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[:3], [999, 200, 300])), max_requests=5)
    with pytest.raises(RevisionConflictError, match="内容不同"):
        acquire_dataset(dataset=dataset, request=DataRequest(start=DAYS[2], end_exclusive=DAYS[3], require_full_coverage=False),
                        store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    assert manifest.content_sha256 == first.content_sha256
    assert store.read_frame(manifest, verify_content=True).loc[DAYS[0], "value"] == 100.0


def test_out_of_window_disappearance_is_rejected_before_publish(cache_root):
    """只请求 1 月 3 日，来源全量响应少了 1 月 1 日：历史删除不由修订策略放行，旧正式数据保持不变。"""
    dataset, first = _cache_three_days(cache_root)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[1:3], [200, 300])), max_requests=5)
    with pytest.raises(RevisionConflictError, match="消失"):
        acquire_dataset(dataset=dataset, request=DataRequest(start=DAYS[2], end_exclusive=DAYS[3], require_full_coverage=False),
                        store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)
    assert DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest().rows == 3


def test_unchanged_overlap_with_narrow_request_publishes_normally(cache_root):
    """正向对照：区间外内容不变、只新增 1 月 4 日 → 正常发布，revision_changed_rows=0。"""
    dataset, first = _cache_three_days(cache_root)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[:4], [100, 200, 300, 400])), max_requests=5)
    result = acquire_dataset(dataset=dataset, request=DataRequest(start=DAYS[3], end_exclusive=DAYS[4], require_full_coverage=False),
                             store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)
    manifest = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id).read_manifest()
    assert result.refresh_failed is False and manifest.rows == 4 and manifest.extra["revision_changed_rows"] == 0


def test_accept_latest_records_every_actual_revision_and_keeps_previous_content(cache_root):
    """ACCEPT_LATEST：区间外的实际修订也必须计入（1 条、范围 1 月 1 日），并保留上一份内容文件；历史删除仍拒绝。"""
    import dataclasses
    from rbt_fin.contracts import REVISION_ACCEPT_LATEST
    dataset, first = _cache_three_days(cache_root)
    first_content_file = Path(first.content_path).name
    accepting = dataclasses.replace(dataset, revision_policy=REVISION_ACCEPT_LATEST)
    client2, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[:3], [999, 200, 300])), max_requests=5)
    result = acquire_dataset(dataset=accepting, request=DataRequest(start=DAYS[2], end_exclusive=DAYS[3], require_full_coverage=False),
                             store_root=cache_root, now_utc=NOW + dt.timedelta(days=1), http=client2)
    store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    assert manifest.extra["revision_changed_rows"] == 1 and any("修订：1 条" in d and DAYS[0].strftime("%Y-%m-%d") in d for d in result.degradations)
    assert store.read_frame(manifest, verify_content=True).loc[DAYS[0], "value"] == 999.0
    assert manifest.previous_content_file == first_content_file and (store.dataset_dir / manifest.previous_content_file).exists()
    client3, _ = make_client(lambda u, p: FakeResponse(status_code=200, payload=_blockchain_payload(DAYS[1:3], [200, 300])), max_requests=5)
    with pytest.raises(RevisionConflictError, match="消失"):
        acquire_dataset(dataset=accepting, request=DataRequest(start=DAYS[2], end_exclusive=DAYS[3], require_full_coverage=False),
                        store_root=cache_root, now_utc=NOW + dt.timedelta(days=2), http=client3)
