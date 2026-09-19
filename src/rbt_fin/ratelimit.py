# -*- coding: utf-8 -*-
"""
主动请求限速：按 host / 端点 / LIMIT 计算权重，按官方限额窗口排程，用响应头校正。

功能或目的（trading#105 E2E 修订指令，RBT_FIN#5）：
    迁移前 trading 的 `binance_data_handler.get_binance_bars` 在每次成功请求后固定
    `sleep(0.3)`，承担主动节流。rbt_fin 0.3.2 之前的 `HttpClient` 只有请求总预算、超时
    与失败退避，没有成功请求之间的节流、端点权重预算与响应头用量反馈。本模块补回这项职责：

    - 权重表来自官方文档（见 `BINANCE_FUTURES_*` / `BINANCE_SPOT_*` 常量的 `source` 字段），
      按 host、端点路径与 `limit` 参数计算；未声明权重的 Binance 端点直接 Fail Fast，不猜。
    - 每个限额桶在进程内共享账本（`shared_registry()`）：同一进程内多个 `HttpClient` /
      数据集共用同一份额度，不会每换一个数据集就重新拿到一整份额度。
    - 发送前排程：本窗口剩余可用额度（`limit * max_share`）不足时，等待到下一窗口起点；
      成功请求同样受约束，重试也同样计费。
    - 响应头（`X-MBX-USED-WEIGHT-1M`）只向上校正本地账本：服务端看到的用量可能包含
      同一出口其他程序的请求，本地不能凭自己的计数放松。
    - 429：遵守响应 `Retry-After`（有上限，超过上限即失败）；缺失时保守等到下一窗口起点。
      418：视为封禁状态，立即失败，不再每 1.5s 重试；不换 IP、不换 key、不绕封禁。
    - 总等待时间由 `RequestBudget.max_total_wait_seconds` 限制，耗尽即失败。

    不能保证的事（如实记录）：Binance 限额按 IP 统计，同一出口的其他进程/程序用量本模块
    看不到，只能通过 `max_share` 余量与响应头反馈降低触发限流的概率；不能把本模块当作
    多进程/多机的分布式限流器。

调用关系：
    - `rbt_fin.http.HttpClient._request()` 在每次发送前调用 `RateLimitRegistry.before_send()`，
      收到响应后调用 `after_response()`，收到 429/418 时调用 `penalty()`。
    - `discover_binance_futures_limit()` 用真实 `exchangeInfo` 刷新桶限额（一次请求，权重 1）。
    - 时钟与 sleep 可注入：测试用虚拟时钟验证排程，不真实等待。
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlsplit

from .errors import ConfigError, HttpStatusError, ResponseSchemaError, SourceBannedError

# ---------------------------------------------------------------------------
# 官方限额与权重表（来源写在 source 字段；数值随官方协议变化时必须改这里并重新核对）
# ---------------------------------------------------------------------------

FUTURES_HOST = "fapi.binance.com"
SPOT_HOSTS = ("api.binance.com", "data-api.binance.vision")
USED_WEIGHT_HEADER_1M = "X-MBX-USED-WEIGHT-1M"

FUTURES_GENERAL_INFO_URL = (
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info"
)
FUTURES_KLINES_DOC_URL = (
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data"
)
FUTURES_FUNDING_DOC_URL = (
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History"
)
SPOT_GENERAL_INFO_URL = "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-api-information"


@dataclass(frozen=True)
class BucketSpec:
    """
    一个限额桶：官方对某类请求在固定窗口内允许的最大权重。

    Attributes:
        name: 桶名（进程内唯一），例如 `fapi.binance.com:REQUEST_WEIGHT:60s`。
        limit: 官方限额（窗口内允许的最大权重）。
        interval_seconds: 窗口长度（秒）；Binance 的窗口按墙钟对齐（每分钟/每 5 分钟重置）。
        used_header: 响应中报告"本窗口已用权重"的头名；None 表示该桶没有响应头反馈。
        source: 数值依据（文档 URL 或 `exchangeInfo@<time>`）。
    """

    name: str
    limit: int
    interval_seconds: int
    used_header: Optional[str]
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0:
            raise ConfigError(f"限额桶 {self.name!r} 的 limit 必须为正整数，实际为 {self.limit!r}")
        if not isinstance(self.interval_seconds, int) or self.interval_seconds <= 0:
            raise ConfigError(f"限额桶 {self.name!r} 的 interval_seconds 必须为正整数")


@dataclass(frozen=True)
class RequestCost:
    """一次请求落在哪个桶、消耗多少权重。"""

    bucket: str
    weight: int


FUTURES_REQUEST_WEIGHT_BUCKET = BucketSpec(
    name=f"{FUTURES_HOST}:REQUEST_WEIGHT:60s",
    # 文档 exchangeInfo 示例值；运行时应以 `discover_binance_futures_limit()` 读到的实际值为准。
    limit=2400,
    interval_seconds=60,
    used_header=USED_WEIGHT_HEADER_1M,
    source=f"{FUTURES_GENERAL_INFO_URL}（REQUEST_WEIGHT MINUTE 1；exchangeInfo 示例 2400）",
)
FUTURES_FUNDING_BUCKET = BucketSpec(
    name=f"{FUTURES_HOST}:fundingRate:300s",
    limit=500,
    interval_seconds=300,
    used_header=None,
    source=f"{FUTURES_FUNDING_DOC_URL}（share 500/5min/IP rate limit with fundingInfo）",
)


def _spot_bucket(host: str) -> BucketSpec:
    """现货/公开镜像的 REQUEST_WEIGHT 桶（文档值 6000/min；本次未联网核对该数值）。"""
    return BucketSpec(
        name=f"{host}:REQUEST_WEIGHT:60s",
        limit=6000,
        interval_seconds=60,
        used_header=USED_WEIGHT_HEADER_1M,
        source=f"{SPOT_GENERAL_INFO_URL}（REQUEST_WEIGHT MINUTE 1 = 6000，文档值）",
    )


# 期货 K 线族端点：权重随 LIMIT 分档（官方 Kline 文档：[1,100) 1；[100,500) 2；[500,1000] 5；>1000 10）。
FUTURES_KLINES_PATHS = frozenset(
    {"/fapi/v1/klines", "/fapi/v1/indexPriceKlines", "/fapi/v1/markPriceKlines", "/fapi/v1/premiumIndexKlines"}
)
# 未传 limit 时端点默认 500（文档默认值），按 500 档计费。
FUTURES_KLINES_DEFAULT_LIMIT = 500
# 期货固定权重端点（文档值）。
FUTURES_FIXED_WEIGHTS: Mapping[str, int] = {
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/ping": 1,
    "/fapi/v1/time": 1,
    "/fapi/v1/premiumIndex": 1,
}
# 现货固定权重端点（文档值）：klines 权重 2（与 LIMIT 无关），exchangeInfo 20。
SPOT_FIXED_WEIGHTS: Mapping[str, int] = {
    "/api/v3/klines": 2,
    "/api/v3/exchangeInfo": 20,
    "/api/v3/ping": 1,
    "/api/v3/time": 1,
}


def futures_klines_weight(limit: int) -> int:
    """
    期货 K 线族端点按 LIMIT 分档的请求权重（官方 Kline 文档）。

    Args:
        limit: 请求的 `limit` 参数（1..1500）。

    Returns:
        权重：[1,100) → 1；[100,500) → 2；[500,1000] → 5；(1000,1500] → 10。
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 1500):
        raise ConfigError(f"K 线 limit 必须是 1..1500 的整数，实际为 {limit!r}")
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


def classify_request(url: str, params: Mapping[str, Any]) -> Optional[RequestCost]:
    """
    判定一次 GET 属于哪个限额桶、消耗多少权重。

    Args:
        url: 完整 URL。
        params: 查询参数（K 线族按其中的 `limit` 计费）。

    Returns:
        RequestCost；非 Binance host 返回 None（本模块不对其他来源做权重限速）。

    Raises:
        ConfigError: Binance host 上未声明权重的端点（不猜，必须先在本模块登记）。
    """
    parts = urlsplit(url)
    host = parts.netloc.lower()
    path = parts.path
    if host == FUTURES_HOST:
        if path in FUTURES_KLINES_PATHS:
            raw_limit = params.get("limit", FUTURES_KLINES_DEFAULT_LIMIT)
            return RequestCost(bucket=FUTURES_REQUEST_WEIGHT_BUCKET.name, weight=futures_klines_weight(int(raw_limit)))
        if path == "/fapi/v1/fundingRate":
            return RequestCost(bucket=FUTURES_FUNDING_BUCKET.name, weight=1)
        if path in FUTURES_FIXED_WEIGHTS:
            return RequestCost(bucket=FUTURES_REQUEST_WEIGHT_BUCKET.name, weight=FUTURES_FIXED_WEIGHTS[path])
        raise ConfigError(
            f"Binance 期货端点 {path!r} 未在 rbt_fin.ratelimit 登记请求权重；"
            "请按官方文档声明后再使用，不按猜测的权重发送。"
        )
    if host in SPOT_HOSTS:
        if path in SPOT_FIXED_WEIGHTS:
            return RequestCost(bucket=_spot_bucket(host).name, weight=SPOT_FIXED_WEIGHTS[path])
        raise ConfigError(
            f"Binance 现货端点 {path!r} 未在 rbt_fin.ratelimit 登记请求权重；请先按官方文档声明。"
        )
    return None


def bucket_spec_for(name: str) -> BucketSpec:
    """按桶名取默认规格（文档值）。"""
    if name == FUTURES_REQUEST_WEIGHT_BUCKET.name:
        return FUTURES_REQUEST_WEIGHT_BUCKET
    if name == FUTURES_FUNDING_BUCKET.name:
        return FUTURES_FUNDING_BUCKET
    for host in SPOT_HOSTS:
        spec = _spot_bucket(host)
        if spec.name == name:
            return spec
    raise ConfigError(f"未知限额桶 {name!r}")


# ---------------------------------------------------------------------------
# 排程
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitConfig:
    """
    排程参数。

    Attributes:
        max_share: 本进程允许使用的限额比例（0 < x <= 1）。默认 0.5：与旧实现每页 sleep 0.3s
                   的节奏（≈200 请求/分 × 权重 5 ≈ 1000/分，约为 2400 的 42%）量级相当，
                   同时给同一出口的其他请求留余量。
        window_margin_seconds: 等到下一窗口起点后再多等的安全边距，吸收本地与服务端时钟偏差。
        cooldown_max_seconds: 429 `Retry-After` 允许的最大冷却秒数；超过即失败（有界处理）。
    """

    max_share: float = 0.5
    window_margin_seconds: float = 0.25
    cooldown_max_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not (0.0 < self.max_share <= 1.0):
            raise ConfigError(f"max_share 必须在 (0, 1]，实际为 {self.max_share!r}")
        if self.window_margin_seconds < 0 or self.cooldown_max_seconds < 0:
            raise ConfigError("window_margin_seconds / cooldown_max_seconds 不能为负")


@dataclass
class BucketStats:
    """一个桶的证据统计（全部为累计值）。"""

    requests: int = 0
    weight_used: int = 0
    throttle_waits: int = 0
    throttle_wait_seconds: float = 0.0
    header_corrections: int = 0
    max_header_used: int = 0
    cooldowns: int = 0
    cooldown_wait_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """转成可写入 manifest/证据的字典。"""
        return {
            "requests": self.requests,
            "weight_used": self.weight_used,
            "throttle_waits": self.throttle_waits,
            "throttle_wait_seconds": self.throttle_wait_seconds,
            "header_corrections": self.header_corrections,
            "max_header_used": self.max_header_used,
            "cooldowns": self.cooldowns,
            "cooldown_wait_seconds": self.cooldown_wait_seconds,
        }


class WeightBucket:
    """
    单个限额桶的进程内账本与排程。

    为什么窗口按墙钟对齐：Binance 的计数器按固定的分钟/5 分钟窗口重置（响应头报告的是
    当前窗口已用量），本地账本必须用同一种窗口才能与响应头对得上；等待时长用同一时钟
    计算并有界（最多一个窗口 + 边距），墙钟跳变不会造成超长等待。
    """

    def __init__(
        self,
        *,
        spec: BucketSpec,
        config: RateLimitConfig,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.spec = spec
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._window_id: Optional[int] = None
        self._used = 0
        self.stats = BucketStats()

    @property
    def budget_per_window(self) -> int:
        """本进程在一个窗口内允许消耗的权重（limit × max_share，向下取整，至少 1）。"""
        return max(1, int(math.floor(self.spec.limit * self._config.max_share)))

    @property
    def used_in_window(self) -> int:
        """当前窗口已用权重（本地账本，含响应头上调）。"""
        self._roll(self._clock())
        return self._used

    def _roll(self, now: float) -> None:
        window_id = int(now // self.spec.interval_seconds)
        if window_id != self._window_id:
            self._window_id = window_id
            self._used = 0

    def seconds_until_next_window(self, now: float) -> float:
        """到下一窗口起点（含安全边距）的等待秒数，有界于 [margin, interval + margin]。"""
        interval = self.spec.interval_seconds
        wait = (int(now // interval) + 1) * interval - now + self._config.window_margin_seconds
        return min(max(wait, self._config.window_margin_seconds), interval + self._config.window_margin_seconds)

    def reserve(self, weight: int, *, context: str) -> float:
        """
        在发送前预留权重；本窗口额度不足则等待到下一窗口。

        Args:
            weight: 本次请求权重。
            context: 报错上下文。

        Returns:
            本次为排程实际等待的秒数（0 表示无需等待）。

        Raises:
            ConfigError: 单次权重超过本进程一个窗口的可用额度（永远无法排进去）。
        """
        if weight > self.budget_per_window:
            raise ConfigError(
                f"{context}: 单次请求权重 {weight} 超过桶 {self.spec.name} 的每窗口可用额度 "
                f"{self.budget_per_window}（limit={self.spec.limit} × max_share={self._config.max_share}）。"
            )
        waited = 0.0
        while True:
            now = self._clock()
            self._roll(now)
            if self._used + weight <= self.budget_per_window:
                self._used += weight
                self.stats.requests += 1
                self.stats.weight_used += weight
                return waited
            wait = self.seconds_until_next_window(now)
            self._sleep(wait)
            waited += wait
            self.stats.throttle_waits += 1
            self.stats.throttle_wait_seconds += wait

    def observe_used(self, used: int) -> None:
        """用响应头报告的本窗口已用权重向上校正本地账本（只上调，不下调）。"""
        self._roll(self._clock())
        if used > self.stats.max_header_used:
            self.stats.max_header_used = used
        if used > self._used:
            self._used = used
            self.stats.header_corrections += 1

    def record_cooldown(self, seconds: float) -> None:
        """记录一次冷却等待（由 registry 在 429 后调用）。"""
        self.stats.cooldowns += 1
        self.stats.cooldown_wait_seconds += seconds


def _parse_retry_after(headers: Mapping[str, Any]) -> Optional[float]:
    """解析 `Retry-After`（秒）；缺失返回 None；非法值视为缺失（保守路径）。"""
    for key, value in headers.items():
        if str(key).lower() == "retry-after":
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return None
            return seconds if seconds >= 0 else None
    return None


def _parse_used_weight(headers: Mapping[str, Any], header_name: str) -> Optional[int]:
    """读取已用权重响应头；缺失或非法返回 None（不据此放松账本）。"""
    wanted = header_name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


class RateLimitRegistry:
    """
    进程内共享的限额桶集合。

    为什么是共享对象而不是每个 HttpClient 自带：同一进程里 registry 会为多个数据集
    分别构造/复用 HttpClient，若每个客户端各有一份账本，就等于每换一个数据集重新拿到
    一整份额度，Binance 看到的却是同一个 IP。
    """

    def __init__(
        self,
        *,
        config: Optional[RateLimitConfig] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Args:
            config: 排程参数；默认 `RateLimitConfig()`。
            clock: 返回 epoch 秒的时钟（窗口按其对齐）；测试注入虚拟时钟。
            sleep: 等待实现；测试注入推进虚拟时钟的实现。
        """
        self._config = config if config is not None else RateLimitConfig()
        self._clock = clock
        self._sleep = sleep
        self._buckets: Dict[str, WeightBucket] = {}
        self._lock = threading.Lock()
        # 已用运行时来源（exchangeInfo）刷新过限额的桶：同一进程只发现一次，不为每个数据集重复请求。
        self._discovered: Dict[str, str] = {}

    @property
    def config(self) -> RateLimitConfig:
        """当前排程参数。"""
        return self._config

    def bucket(self, name: str) -> WeightBucket:
        """取（或按默认规格创建）一个桶。"""
        with self._lock:
            if name not in self._buckets:
                self._buckets[name] = WeightBucket(
                    spec=bucket_spec_for(name), config=self._config, clock=self._clock, sleep=self._sleep
                )
            return self._buckets[name]

    def update_limit(self, *, name: str, limit: int, source: str) -> BucketSpec:
        """
        用运行时发现的实际限额替换桶规格（账本与统计保留）。

        Args:
            name: 桶名。
            limit: 实际限额。
            source: 依据（例如 `exchangeInfo@2026-09-19T10:00:00Z`）。

        Returns:
            新的 BucketSpec。
        """
        bucket = self.bucket(name)
        old = bucket.spec
        new_spec = BucketSpec(
            name=old.name, limit=limit, interval_seconds=old.interval_seconds, used_header=old.used_header, source=source
        )
        bucket.spec = new_spec
        self._discovered[name] = source
        return new_spec

    def discovered_source(self, name: str) -> Optional[str]:
        """该桶是否已用运行时来源刷新过限额；返回来源说明或 None。"""
        return self._discovered.get(name)

    def before_send(self, *, url: str, params: Mapping[str, Any], context: str) -> float:
        """发送前排程；返回为排程等待的秒数。非 Binance host 返回 0。"""
        cost = classify_request(url, params)
        if cost is None:
            return 0.0
        return self.bucket(cost.bucket).reserve(cost.weight, context=context)

    def after_response(self, *, url: str, headers: Mapping[str, Any]) -> None:
        """用响应头校正账本；没有对应头或桶无响应头反馈时不做任何事。"""
        cost = classify_request(url, {})  # 只为定位桶；权重在这里无关
        if cost is None:
            return
        bucket = self.bucket(cost.bucket)
        if bucket.spec.used_header is None:
            return
        used = _parse_used_weight(headers, bucket.spec.used_header)
        if used is not None:
            bucket.observe_used(used)

    def penalty(self, *, url: str, status: int, headers: Mapping[str, Any], context: str) -> float:
        """
        处理限流/封禁状态并返回应等待的冷却秒数。

        - 429：`Retry-After` 存在则遵守（超过 `cooldown_max_seconds` 即失败）；缺失则等到
          下一窗口起点（保守、有界）。
        - 418：封禁状态，立即抛 `SourceBannedError`，由调用方终止（不再重试）。

        Returns:
            冷却秒数（调用方负责 sleep）。

        Raises:
            SourceBannedError: 418。
            HttpStatusError: 429 且服务端要求的冷却超过上限。
        """
        retry_after = _parse_retry_after(headers)
        if status == 418:
            raise SourceBannedError(
                f"{context}: HTTP 418 —— 来源已封禁本出口（Retry-After={retry_after}）；"
                "按封禁处理，立即停止，不重试、不换出口。"
            )
        if status != 429:
            raise ConfigError(f"penalty 只处理 429/418，收到 {status}")
        cost = classify_request(url, {})
        bucket = self.bucket(cost.bucket) if cost is not None else None
        if retry_after is not None:
            if retry_after > self._config.cooldown_max_seconds:
                raise HttpStatusError(
                    f"{context}: HTTP 429 要求冷却 {retry_after:.0f}s，超过上限 "
                    f"{self._config.cooldown_max_seconds:.0f}s；按明确失败处理，不返回部分数据。"
                )
            cooldown = max(retry_after, self._config.window_margin_seconds)
        else:
            interval = bucket.spec.interval_seconds if bucket is not None else 60
            now = self._clock()
            cooldown = (int(now // interval) + 1) * interval - now + self._config.window_margin_seconds
            cooldown = min(max(cooldown, self._config.window_margin_seconds), interval + self._config.window_margin_seconds)
        if bucket is not None:
            bucket.record_cooldown(cooldown)
        return cooldown

    def snapshot(self) -> Dict[str, Any]:
        """全部桶的规格与统计（证据归档用）。"""
        with self._lock:
            return {
                name: {"limit": b.spec.limit, "interval_seconds": b.spec.interval_seconds, "source": b.spec.source,
                       "max_share": self._config.max_share, "budget_per_window": b.budget_per_window, **b.stats.as_dict()}
                for name, b in self._buckets.items()
            }


_SHARED: Optional[RateLimitRegistry] = None
_SHARED_LOCK = threading.Lock()


def shared_registry() -> RateLimitRegistry:
    """进程内共享的 registry（惰性创建；`import rbt_fin` 不会创建它，也不联网）。"""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = RateLimitRegistry()
        return _SHARED


def discover_binance_futures_limit(http, *, registry: Optional[RateLimitRegistry] = None) -> Dict[str, Any]:
    """
    读取真实 `GET /fapi/v1/exchangeInfo` 的 `rateLimits`，刷新期货 REQUEST_WEIGHT 桶限额。

    为什么不把文档示例当永久配置：官方要求以运行时 `exchangeInfo` 为准；本函数只消耗 1 权重。

    Args:
        http: `rbt_fin.http.HttpClient`（本身也受限速）。
        registry: 目标 registry；默认进程共享 registry。

    Returns:
        `{"limit": int, "interval_seconds": 60, "source": str}`。

    Raises:
        ResponseSchemaError: 响应中找不到 REQUEST_WEIGHT/MINUTE/1 限额。
    """
    target = registry if registry is not None else shared_registry()
    payload = http.get_json(
        url=f"https://{FUTURES_HOST}/fapi/v1/exchangeInfo", params={}, context="binance_futures_exchange_info"
    )
    limits = payload.get("rateLimits") if isinstance(payload, dict) else None
    if not isinstance(limits, list):
        raise ResponseSchemaError("exchangeInfo 响应缺少 rateLimits 数组，无法确认实际限额。")
    for item in limits:
        if (
            isinstance(item, dict)
            and item.get("rateLimitType") == "REQUEST_WEIGHT"
            and item.get("interval") == "MINUTE"
            and item.get("intervalNum") == 1
        ):
            limit = item.get("limit")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ResponseSchemaError(f"exchangeInfo REQUEST_WEIGHT limit 非法: {limit!r}")
            source = f"exchangeInfo@{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(target._clock()))}"
            spec = target.update_limit(name=FUTURES_REQUEST_WEIGHT_BUCKET.name, limit=limit, source=source)
            return {"limit": spec.limit, "interval_seconds": spec.interval_seconds, "source": spec.source}
    raise ResponseSchemaError("exchangeInfo.rateLimits 中没有 REQUEST_WEIGHT/MINUTE/1 条目。")


def ensure_binance_futures_limit_discovered(http) -> Dict[str, Any]:
    """
    在第一次真正向 fapi 取数前保证限额已按运行时 exchangeInfo 刷新（每个 registry 只做一次）。

    为什么放在适配器取数路径而不是入口：只有确实要联网时才值得花这 1 权重；
    缓存已满足请求时不应发出任何请求（C 类"未变更"路径零请求）。

    Args:
        http: `rbt_fin.http.HttpClient`。

    Returns:
        `{"limit", "interval_seconds", "source"}`；已发现过时直接返回当前规格。
    """
    registry = http.rate_limit
    source = registry.discovered_source(FUTURES_REQUEST_WEIGHT_BUCKET.name)
    if source is not None:
        spec = registry.bucket(FUTURES_REQUEST_WEIGHT_BUCKET.name).spec
        return {"limit": spec.limit, "interval_seconds": spec.interval_seconds, "source": spec.source}
    return discover_binance_futures_limit(http, registry=registry)
