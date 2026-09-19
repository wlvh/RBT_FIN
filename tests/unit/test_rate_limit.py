# -*- coding: utf-8 -*-
"""
主动限速（rbt_fin.ratelimit + HttpClient 接线）的虚拟时钟测试。

功能或目的（trading#105 E2E 修订指令 §一）：
    用虚拟时钟与假响应验证——连续成功请求按端点权重计费、窗口额度耗尽时等待到下一窗口、
    多个客户端共享同一份进程内额度、响应头只向上校正账本、429 遵守有界 Retry-After、
    缺失 Retry-After 时保守等待、418 立即按封禁停止、重试同样计费、总等待预算耗尽明确失败、
    非 Binance host 不受权重限速、exchangeInfo 发现刷新限额、分页结果不因限速改变。
    测试不真的等待，也不触发真实来源限流。

调用关系：
    pytest tests/unit → 本文件；依赖 tests/conftest.py 的 FakeResponse / ScriptedSession / kline_row。
"""

from __future__ import annotations

import pandas as pd
import pytest

from rbt_fin.errors import BudgetExhaustedError, ConfigError, HttpStatusError, SourceBannedError
from rbt_fin.http import HttpClient, RequestBudget
from rbt_fin.ratelimit import (
    FUTURES_FUNDING_BUCKET,
    FUTURES_REQUEST_WEIGHT_BUCKET,
    RateLimitConfig,
    RateLimitRegistry,
    classify_request,
    discover_binance_futures_limit,
    futures_klines_weight,
)
from rbt_fin.registry import build_klines_dataset
from rbt_fin.sources.binance_klines import fetch_klines_range
from tests.conftest import FakeResponse, ScriptedSession, kline_row

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OTHER_URL = "https://api.blockchain.info/charts/hash-rate"
WINDOW_START = 1_700_000_020.0  # 任意 epoch 秒；1_700_000_000 % 60 == 20，因此此刻位于 60s 窗口内第 40 秒


class VirtualClock:
    """
    虚拟时钟：墙钟（epoch 秒，只定位窗口）与单调钟（只计经过时间）分开控制；
    sleep 默认同步推进两者并记录，不真的等待。测试可单独拨动墙钟（`jump_wall`）或只推进单调钟（`advance_mono`）。
    """

    def __init__(self, start: float = WINDOW_START) -> None:
        self.wall = start
        self.mono = 1000.0
        self.sleeps: list = []

    def clock(self) -> float:  # 兼容旧用例：同一虚拟时钟
        return self.wall

    def wall_clock(self) -> float:
        return self.wall

    def mono_clock(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.wall += seconds
        self.mono += seconds

    def jump_wall(self, seconds: float) -> None:
        """只校正墙钟（前拨为正、回拨为负），单调时间不变。"""
        self.wall += seconds

    def advance_mono(self, seconds: float) -> None:
        """只推进单调时间（墙钟被冻结/回拨的情形）。"""
        self.mono += seconds


def _registry(clock: VirtualClock, **config) -> RateLimitRegistry:
    return RateLimitRegistry(config=RateLimitConfig(**config), wall_clock=clock.wall_clock, mono_clock=clock.mono_clock, sleep=clock.sleep)


def _client(handler, registry, *, max_requests=10_000, max_retries=0, max_total_wait=3600.0):
    session = ScriptedSession(handler)
    client = HttpClient(
        session=session,
        budget=RequestBudget(max_requests=max_requests, max_retries=max_retries, backoff_seconds=1.5,
                             max_total_wait_seconds=max_total_wait),
        sleep=registry._sleep,  # 冷却/退避等待也走同一虚拟时钟
        rate_limit=registry,
    )
    return client, session


def _ok(payload=None, headers=None):
    return FakeResponse(status_code=200, payload=[] if payload is None else payload, headers=headers)


# ---------------------------------------------------------------------------
# 权重表与分类
# ---------------------------------------------------------------------------


def test_futures_klines_weight_tiers_follow_official_table():
    """官方 Kline 文档分档：[1,100) 1；[100,500) 2；[500,1000] 5；>1000 10。"""
    assert [futures_klines_weight(x) for x in (1, 99, 100, 499, 500, 1000, 1001, 1500)] == [1, 1, 2, 2, 5, 5, 10, 10]
    with pytest.raises(ConfigError):
        futures_klines_weight(0)
    with pytest.raises(ConfigError):
        futures_klines_weight(1501)


def test_classify_request_by_host_endpoint_and_limit():
    """按 host/端点/LIMIT 计费；未登记的 Binance 端点 Fail Fast；非 Binance host 不限速。"""
    cost = classify_request(KLINES_URL, {"limit": 1000})
    assert (cost.bucket, cost.weight) == (FUTURES_REQUEST_WEIGHT_BUCKET.name, 5)
    assert classify_request(KLINES_URL, {"limit": 1500}).weight == 10
    assert classify_request("https://fapi.binance.com/fapi/v1/indexPriceKlines", {"limit": 499}).weight == 2
    assert classify_request(KLINES_URL, {}).weight == 5  # 未传 limit 按端点默认 500 计费
    funding = classify_request(FUNDING_URL, {"limit": 1000})
    assert (funding.bucket, funding.weight) == (FUTURES_FUNDING_BUCKET.name, 1)
    assert classify_request("https://fapi.binance.com/fapi/v1/exchangeInfo", {}).weight == 1
    assert classify_request(OTHER_URL, {}) is None
    with pytest.raises(ConfigError, match="未在 rbt_fin.ratelimit 登记"):
        classify_request("https://fapi.binance.com/fapi/v1/depth", {})


def test_page_size_default_is_1000_and_weight_per_bar_comparison():
    """默认页 1000：每权重 K 线数 499→249.5、1000→200、1500→150；1500 少发请求但总权重更高。"""
    assert build_klines_dataset().params["limit"] == 1000
    bars_per_weight = {limit: limit / futures_klines_weight(limit) for limit in (499, 1000, 1500)}
    assert bars_per_weight == {499: 249.5, 1000: 200.0, 1500: 150.0}
    rows = 2_784_961  # E2E 冷启动的实际行数
    weight_1000 = -(-rows // 1000) * 5
    weight_1500 = -(-rows // 1500) * 10
    assert weight_1500 > weight_1000  # 请求数少 1/3，总权重反而多 1/3


# ---------------------------------------------------------------------------
# 排程
# ---------------------------------------------------------------------------


def test_consecutive_successful_requests_are_metered_and_wait_at_window_boundary():
    """连续成功请求同样计费：每窗口只用 limit×max_share；额度耗尽等待到下一窗口起点。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, session = _client(lambda u, p: _ok(), registry)
    # 1200 权重 / 5 = 240 次 limit=1000 请求不等待
    for _ in range(240):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == []
    assert client.stats["throttle_wait_seconds"] == 0.0
    # 第 241 次必须等到下一窗口：窗口内第 40 秒 → 等 20s + 0.25s 边距
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)]
    assert client.stats["throttle_wait_seconds"] == pytest.approx(20.25)
    bucket = client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]
    assert bucket["requests"] == 241 and bucket["weight_used"] == 241 * 5 and bucket["throttle_waits"] == 1
    assert bucket["budget_per_window"] == 1200
    assert len(session.calls) == 241


def test_multiple_clients_share_one_process_budget():
    """同一进程内两个客户端共用同一份额度：合计 240 次后第 241 次等待，而不是各拿 240 次。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    a, _ = _client(lambda u, p: _ok(), registry)
    b, _ = _client(lambda u, p: _ok(), registry)
    for _ in range(120):
        a.get_json(url=KLINES_URL, params={"limit": 1000}, context="a")
        b.get_json(url=KLINES_URL, params={"limit": 1000}, context="b")
    assert clock.sleeps == []
    b.get_json(url=KLINES_URL, params={"limit": 1000}, context="b")
    assert len(clock.sleeps) == 1


def test_used_weight_header_only_raises_local_accounting():
    """响应头报告的已用量高于本地时上调账本（同一出口其他程序的用量）；低于本地时不下调。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    responses = iter([_ok(headers={"X-MBX-USED-WEIGHT-1M": "1195"}), _ok(headers={"x-mbx-used-weight-1m": "3"}), _ok()])
    client, _ = _client(lambda u, p: next(responses), registry)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")  # 本地 5，头 1195 → 账本 1195
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).used_in_window == 1195
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")  # 1195+5=1200 仍可发；头 3 不下调
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).used_in_window == 1200
    assert clock.sleeps == []
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")  # 超出 → 等下一窗口
    assert len(clock.sleeps) == 1
    stats = client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]
    assert stats["header_corrections"] == 1 and stats["max_header_used"] == 1195


def test_missing_header_keeps_local_accounting_unchanged():
    """没有已用量响应头：只按本地计数，不据此放松。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    client, _ = _client(lambda u, p: _ok(), registry)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).used_in_window == 5
    assert client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]["header_corrections"] == 0


def test_funding_endpoint_uses_its_own_five_minute_bucket():
    """fundingRate 走独立的 500/5min 桶（与 fundingInfo 共享），不消耗 REQUEST_WEIGHT 分钟额度。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    client, _ = _client(lambda u, p: _ok(), registry)
    for _ in range(250):
        client.get_json(url=FUNDING_URL, params={"limit": 1000}, context="f")
    assert clock.sleeps == []
    client.get_json(url=FUNDING_URL, params={"limit": 1000}, context="f")  # 第 251 次等到下一个 5 分钟窗口
    assert len(clock.sleeps) == 1 and 0 < clock.sleeps[0] <= 300.25
    assert FUTURES_REQUEST_WEIGHT_BUCKET.name not in client.stats["buckets"]


def test_non_binance_hosts_are_not_weight_limited():
    """其他来源不做权重限速（各自协议不同，不统一加 0.3s）。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.01)
    client, _ = _client(lambda u, p: _ok(payload={"values": []}), registry)
    for _ in range(50):
        client.get_json(url=OTHER_URL, params={}, context="o")
    assert clock.sleeps == [] and client.stats["buckets"] == {}


# ---------------------------------------------------------------------------
# 429 / 418 / 重试 / 预算
# ---------------------------------------------------------------------------


def test_429_honors_retry_after_then_retries_and_counts_cooldown():
    """429 带 Retry-After：冷却指定秒数后重试；等待计入 cooldown 而不是 throttle。"""
    clock = VirtualClock()
    registry = _registry(clock)
    responses = iter([FakeResponse(status_code=429, payload={"code": -1003}, headers={"Retry-After": "7"}), _ok()])
    client, session = _client(lambda u, p: next(responses), registry, max_retries=2)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [7.0] and len(session.calls) == 2
    assert client.stats["cooldown_wait_seconds"] == 7.0 and client.stats["rate_limited_responses"] == 1
    assert client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]["cooldowns"] == 1


def test_429_without_retry_after_waits_until_next_window_conservatively():
    """429 缺 Retry-After：保守等到下一窗口起点（有界 ≤ 60s + 边距），不按 1.5s 退避猜。"""
    clock = VirtualClock()
    registry = _registry(clock, window_margin_seconds=0.25)
    responses = iter([FakeResponse(status_code=429, payload={"code": -1003}), _ok()])
    client, _ = _client(lambda u, p: next(responses), registry, max_retries=1)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)]


def test_429_retry_after_beyond_cap_fails_explicitly_without_waiting():
    """服务端要求的冷却超过上限：明确失败，不等待、不返回部分数据。"""
    clock = VirtualClock()
    registry = _registry(clock, cooldown_max_seconds=180.0)
    client, _ = _client(lambda u, p: FakeResponse(status_code=429, payload={}, headers={"Retry-After": "600"}), registry, max_retries=3)
    with pytest.raises(HttpStatusError, match="超过上限"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == []


def test_429_retries_exhausted_fails():
    """持续 429：重试耗尽后失败，不以无限重试绕过。"""
    clock = VirtualClock()
    registry = _registry(clock)
    client, session = _client(lambda u, p: FakeResponse(status_code=429, payload={}, headers={"Retry-After": "1"}), registry, max_retries=2)
    with pytest.raises(HttpStatusError, match="HTTP 429 重试耗尽"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert len(session.calls) == 3 and clock.sleeps == [1.0, 1.0]


def test_418_is_treated_as_ban_and_stops_immediately():
    """418 = 封禁：立即 SourceBannedError，只发一次请求，不冷却重试、不换出口。"""
    clock = VirtualClock()
    registry = _registry(clock)
    client, session = _client(lambda u, p: FakeResponse(status_code=418, payload={}, headers={"Retry-After": "900"}), registry, max_retries=3)
    with pytest.raises(SourceBannedError, match="418"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert len(session.calls) == 1 and clock.sleeps == []
    assert isinstance(SourceBannedError.__mro__[1], type) and issubclass(SourceBannedError, HttpStatusError)


def test_retries_after_server_error_consume_weight_too():
    """503 后重试：两次发送都计费（重试不是免费的），退避等待计入 backoff。"""
    clock = VirtualClock()
    registry = _registry(clock)
    responses = iter([FakeResponse(status_code=503, payload={}), _ok()])
    client, _ = _client(lambda u, p: next(responses), registry, max_retries=1)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    bucket = client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]
    assert bucket["requests"] == 2 and bucket["weight_used"] == 10
    assert client.stats["backoff_wait_seconds"] == 1.5 and clock.sleeps == [1.5]


def test_total_wait_budget_exhaustion_fails_instead_of_waiting_forever():
    """总等待预算耗尽 → BudgetExhaustedError（异常而不是部分数据）。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    client, _ = _client(lambda u, p: _ok(), registry, max_total_wait=10.0)
    for _ in range(240):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    with pytest.raises(BudgetExhaustedError, match="超过剩余等待预算"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == []  # 预算不足：不安排等待


def test_single_request_weight_above_window_budget_is_rejected():
    """单次权重永远排不进窗口（max_share 太小）：ConfigError 而不是死循环。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.001)  # 2400 × 0.001 → 2
    client, _ = _client(lambda u, p: _ok(), registry)
    with pytest.raises(ConfigError, match="超过桶"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")


# ---------------------------------------------------------------------------
# 发现实际限额 / 分页结果不变
# ---------------------------------------------------------------------------


def test_discover_binance_futures_limit_updates_bucket_from_exchange_info():
    """exchangeInfo.rateLimits 的 REQUEST_WEIGHT/MINUTE/1 覆盖文档示例值，并记录来源。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5)
    payload = {"rateLimits": [
        {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
        {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1800},
    ]}

    class RawSession:
        """不经 ScriptedSession 的 exchangeInfo 拦截：本用例要验证的正是发现逻辑本身。"""

        headers: dict = {}

        def __init__(self):
            self.calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append(url)
            return _ok(payload=payload)

    session = RawSession()
    client = HttpClient(session=session, budget=RequestBudget(max_requests=10, max_retries=0), sleep=registry._sleep, rate_limit=registry)
    info = discover_binance_futures_limit(client, registry=registry)
    assert info["limit"] == 1800 and info["source"].startswith("exchangeInfo@")
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).budget_per_window == 900
    assert session.calls == ["https://fapi.binance.com/fapi/v1/exchangeInfo"]
    assert client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]["weight_used"] == 1  # 发现本身也计费
    # 同一 registry 只发现一次：ensure 不再发请求
    from rbt_fin.ratelimit import ensure_binance_futures_limit_discovered
    assert ensure_binance_futures_limit_discovered(client)["limit"] == 1800
    assert session.calls == ["https://fapi.binance.com/fapi/v1/exchangeInfo"]


def test_pagination_result_is_identical_with_and_without_throttling():
    """限速只改变发送时机：同一固定截止内的 K 线内容与不限速时逐值相同。"""
    start = pd.Timestamp("2026-04-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-04-01 00:06", tz="UTC")  # 6 根，每页 2 根 → 3 页

    def handler(url, params):
        cursor = int(params["startTime"])
        rows = [kline_row(open_time_ms=cursor + i * 60_000, open_=1 + i, high=2 + i, low=0.5, close=1.5 + i, volume=10)
                for i in range(2) if cursor + i * 60_000 < int(end.timestamp() * 1000)]
        return FakeResponse(status_code=200, payload=rows)

    def run(max_share):
        clock = VirtualClock()
        registry = _registry(clock, max_share=max_share)
        client, _ = _client(handler, registry)
        frame, complete, _ = fetch_klines_range(http=client, market="usdm_futures", symbol="BTCUSDT", interval="1m",
                                                start=start, end_exclusive=end, limit=2, context="p")
        return frame, complete, clock.sleeps

    loose_frame, loose_complete, loose_sleeps = run(1.0)
    tight_frame, tight_complete, tight_sleeps = run(0.001)  # 2400×0.001=2 → 每窗口只放 2 个权重 1 的请求
    assert loose_sleeps == [] and len(tight_sleeps) == 1
    assert loose_complete and tight_complete
    pd.testing.assert_frame_equal(loose_frame, tight_frame)


# ---------------------------------------------------------------------------
# 修订 2：等待预算先于排程；墙钟窗口定位与单调等待分离
# ---------------------------------------------------------------------------


def _exhaust_window(client):
    """用完本窗口 1200 权重（240 次 limit=1000），不产生等待。"""
    for _ in range(240):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="fill")


def test_wait_budget_is_checked_before_scheduling_the_wait():
    """剩余预算 10s 而进入下一窗口需 20.25s：立即失败，不等待、不发送、不计请求数。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, session = _client(lambda u, p: _ok(), registry, max_total_wait=10.0)
    _exhaust_window(client)
    with pytest.raises(BudgetExhaustedError, match="不安排超预算的等待"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [] and len(session.calls) == 240 and client.requests_used == 240
    bucket = client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]
    assert bucket["requests"] == 240 and bucket["budget_refusals"] == 1 and bucket["throttle_waits"] == 0


def test_wait_budget_sufficient_waits_and_sends():
    """剩余预算 25s：正常等待 20.25s 后发送（对照）。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, session = _client(lambda u, p: _ok(), registry, max_total_wait=25.0)
    _exhaust_window(client)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)] and len(session.calls) == 241
    assert client.stats["throttle_wait_seconds"] == pytest.approx(20.25)


def test_budget_covers_throttle_cooldown_and_backoff_and_stays_exhausted():
    """同一份预算覆盖排程/冷却/退避；耗尽后后续调用不能绕过。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    # 冷却：预算 5s，429 要求 7s → 不等待、失败；后续调用同样失败
    responses = iter([FakeResponse(status_code=429, payload={}, headers={"Retry-After": "7"}), _ok(), _ok()])
    client, session = _client(lambda u, p: next(responses), registry, max_retries=2, max_total_wait=5.0)
    with pytest.raises(BudgetExhaustedError, match="429 冷却"):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [] and len(session.calls) == 1
    assert client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]["cooldowns"] == 0  # 未发生的冷却不记账
    # 退避：预算 1s，503 后退避 1.5s → 不等待、失败
    clock2 = VirtualClock()
    registry2 = _registry(clock2)
    responses2 = iter([FakeResponse(status_code=503, payload={}), _ok()])
    client2, session2 = _client(lambda u, p: next(responses2), registry2, max_retries=1, max_total_wait=1.0)
    with pytest.raises(BudgetExhaustedError, match="故障退避"):
        client2.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock2.sleeps == [] and len(session2.calls) == 1
    # 累计耗尽：预算 25s，等待 20.25s 后再次需要等待 → 立即失败，且再次调用仍失败
    clock3 = VirtualClock()
    registry3 = _registry(clock3, max_share=0.5, window_margin_seconds=0.25)
    client3, session3 = _client(lambda u, p: _ok(), registry3, max_total_wait=25.0)
    _exhaust_window(client3)
    client3.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")  # 等 20.25s，新窗口已用 5
    for _ in range(239):  # 新窗口再用满 1200（不等待）
        client3.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    for _ in range(2):  # 剩余预算 4.75s < 60.25s：两次都立即失败，不能绕过耗尽状态
        with pytest.raises(BudgetExhaustedError):
            client3.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock3.sleeps == [pytest.approx(20.25)] and len(session3.calls) == 480


def test_normal_window_rollover_requires_monotonic_elapse():
    """正常跨窗口：sleep 同步推进墙钟与单调钟，到期后额度重置。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, _ = _client(lambda u, p: _ok(), registry)
    _exhaust_window(client)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)]
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).used_in_window == 5


@pytest.mark.parametrize("jump_seconds", [60.0, -60.0, 3600.0, -3600.0])
def test_wall_clock_jump_without_monotonic_elapse_does_not_refresh_budget(jump_seconds):
    """墙钟前拨/回拨但单调时间未经过：额度不重置，下一次请求仍按单调到期时刻等待。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, session = _client(lambda u, p: _ok(), registry)
    _exhaust_window(client)
    clock.jump_wall(jump_seconds)
    bucket = registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name)
    assert bucket.used_in_window == 1200  # 跳变没有刷新额度
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)]  # 仍按单调到期时刻等待
    assert len(session.calls) == 241
    assert client.stats["buckets"][FUTURES_REQUEST_WEIGHT_BUCKET.name]["clock_jumps"] >= 1


def test_monotonic_elapse_with_frozen_wall_clock_still_rolls_window():
    """单调时间已过期而墙钟未变（墙钟被冻结/回拨到同一窗口）：按经过时间重置额度，不额外等待。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)
    client, session = _client(lambda u, p: _ok(), registry)
    _exhaust_window(client)
    clock.advance_mono(60.0)  # 只推进单调时间
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [] and len(session.calls) == 241
    assert registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).used_in_window == 5


def test_clock_correction_during_wait_does_not_shorten_or_duplicate_budget():
    """等待期间墙钟被校正：等待仍按单调到期时刻结束；校正后按新墙钟重新锚定，额度只重置一次。"""
    clock = VirtualClock()
    registry = _registry(clock, max_share=0.5, window_margin_seconds=0.25)

    def sleep_with_correction(seconds):
        clock.sleeps.append(seconds)
        clock.mono += seconds
        clock.wall += seconds - 45.0  # 等待期间墙钟被回拨 45s

    registry._sleep = sleep_with_correction
    registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name)._sleep = sleep_with_correction
    client, session = _client(lambda u, p: _ok(), registry)
    _exhaust_window(client)
    client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert clock.sleeps == [pytest.approx(20.25)] and len(session.calls) == 241
    bucket = registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name)
    assert bucket.used_in_window == 5
    # 校正后的墙钟落在更早的窗口编号；额度已按经过时间重置一次，不会因编号变化再次重置：
    # 再发 239 次（1195）正好用满 1200 且无需等待
    for _ in range(239):
        client.get_json(url=KLINES_URL, params={"limit": 1000}, context="t")
    assert bucket.used_in_window == 1200 and len(clock.sleeps) == 1
