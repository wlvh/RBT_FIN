# -*- coding: utf-8 -*-
"""
公共 HTTP 下载护栏。

功能或目的：
    Issue #5 §4.1 要求：网络超时、有限重试和请求/页数预算明确；重试耗尽、分页无进展、
    业务错误或结构错误均不能伪装成正常空数据；HTTP 状态与 JSON 业务状态分别检查。

    本模块只负责“单次请求的可靠性与预算”，分页边界由各来源适配器负责
    （`rbt_fin.sources.*`），因为不同端点的游标语义无法互相推导。

    主动限速（trading#105 E2E 修订）：每次发送前经 `rbt_fin.ratelimit.RateLimitRegistry`
    按 host/端点/LIMIT 权重排程（成功请求与重试同样计费），收到响应后用已用权重响应头
    校正账本；429 遵守有界 `Retry-After`，418 立即按封禁失败。默认使用进程共享 registry，
    同一进程内多个客户端/数据集共用同一份额度。

调用关系：
    - 调用方用 `build_session()` 显式构造 `requests.Session`（import 本模块不会建连接）。
    - `HttpClient.get_json()` 被 `rbt_fin.sources.*` 使用。
    - 预算由 `RequestBudget` 表达，`HttpClient` 在整个生命周期内累计计数。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import re

import requests

from .errors import (
    BudgetExhaustedError,
    BusinessStatusError,
    ConfigError,
    HttpStatusError,
    ResponseSchemaError,
)
from .ratelimit import RateLimitRegistry, shared_registry

# 可重试状态：网关/服务端临时故障。限流 429 与封禁 418 不在这里：它们由
# `RateLimitRegistry.penalty()` 分别按 Retry-After 冷却 / 立即失败处理，不能按普通短暂故障等 1.5s 重试。
# 为什么不含 4xx 其余状态：参数错误、鉴权失败重试只会掩盖问题并触发更严格的封禁。
RETRYABLE_STATUS = frozenset({500, 502, 503, 504, 520, 521, 522, 524})
RATE_LIMITED_STATUS = frozenset({418, 429})

DEFAULT_USER_AGENT = "rbt-fin/0.1 (+research)"


@dataclass(frozen=True)
class RequestBudget:
    """
    一次任务（或一个数据集一次采集）的请求预算。

    Attributes:
        max_requests: 允许发出的 HTTP 请求总数上限（含重试）。耗尽必须终止并报告未完成。
        max_retries: 单个 URL 的最大重试次数（不含首次尝试）。
        connect_timeout: 连接超时秒数。
        read_timeout: 读取超时秒数。
        backoff_seconds: 重试基础退避秒数；第 i 次重试等待 backoff_seconds * (i + 1)。
        max_total_wait_seconds: 本客户端生命周期内允许的总等待秒数（限速排程 + 冷却 + 退避）；
                                耗尽即 `BudgetExhaustedError`，不返回部分数据冒充完整。
                                默认 3600s：按期货 K 线 limit=1000（权重 5）与 max_share=0.5
                                （每分钟 1200 权重）估算，一次约 280 万根 1m K 线的冷启动需要
                                ≈ 2790 页 × 5 / 1200 ≈ 11.6 分钟的排程等待，留有余量。
    """

    max_requests: int = 400
    max_retries: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    backoff_seconds: float = 1.5
    max_total_wait_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ConfigError("max_requests 必须为正整数。")
        if self.max_retries < 0:
            raise ConfigError("max_retries 不能为负。")
        for name in ("connect_timeout", "read_timeout", "backoff_seconds", "max_total_wait_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"{name} 必须是非负数，实际为 {value!r}")


def build_session(*, user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    """
    显式构造一个 `requests.Session`。

    为什么不在模块导入时建全局 session：
        Issue #5 §7 要求 `import rbt_fin` 不联网、不修改进程环境。模块级全局对象会让
        “什么时候开始持有连接池”变得不可控，也让测试难以注入受控 transport。

    Args:
        user_agent: 请求 UA。

    Returns:
        配置好默认头的新 `requests.Session`。
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json, text/plain;q=0.8"})
    return session


_CREDENTIAL_QUERY_KEYS = ("api_key", "apikey", "api-key", "key", "token", "secret", "access_token")
_CREDENTIAL_PATTERN = re.compile(r"(?i)\b(" + "|".join(re.escape(k) for k in _CREDENTIAL_QUERY_KEYS) + r")=([^&\s'\"]+)")


def redact_credentials(text: str) -> str:
    """
    把异常/日志文本里的查询串凭据值改写为 `<redacted>`。

    为什么需要：`requests`/`urllib3` 的连接类异常文本固定包含完整 URL（含 query），FRED/Glassnode 的
    key 以查询参数发送，原样进入异常消息就会随 `DataResult.errors`、任务失败清单、概览 CSV 落盘（§2/§4 日志脱敏）。
    """
    return _CREDENTIAL_PATTERN.sub(lambda m: f"{m.group(1)}=<redacted>", text)


class HttpClient:
    """
    带超时、有限重试与请求预算的 JSON 取数客户端。

    为什么把业务状态检查做成回调：
        HTTP 200 + `retCode != 0`（Bybit）这类失败必须立刻形成明确失败，而不同端点的
        业务状态字段各不相同；由适配器提供 `business_check` 能避免在公共层堆 if/else。
    """

    def __init__(
        self,
        *,
        session: requests.Session,
        budget: RequestBudget,
        sleep: Callable[[float], None] = time.sleep,
        rate_limit: Optional[RateLimitRegistry] = None,
    ) -> None:
        """
        Args:
            session: 由调用方显式构造的 `requests.Session`。
            budget: 请求预算与超时配置。
            sleep: 退避/冷却等待实现；测试可注入无副作用实现以避免真实等待。
            rate_limit: 限速 registry；默认进程共享 registry（同一进程内所有客户端共用额度）。
                        测试可注入带虚拟时钟的 registry。
        """
        if not isinstance(budget, RequestBudget):
            raise ConfigError(f"budget 必须是 RequestBudget，实际为 {type(budget)!r}")
        if rate_limit is not None and not isinstance(rate_limit, RateLimitRegistry):
            raise ConfigError(f"rate_limit 必须是 RateLimitRegistry，实际为 {type(rate_limit)!r}")
        self._session = session
        self._budget = budget
        self._sleep = sleep
        self._rate_limit = rate_limit if rate_limit is not None else shared_registry()
        self._requests_used = 0
        self._throttle_wait_seconds = 0.0
        self._cooldown_wait_seconds = 0.0
        self._backoff_wait_seconds = 0.0
        self._rate_limited_responses = 0

    @property
    def requests_used(self) -> int:
        """已消耗的请求数（含重试），用于证据归档与预算诊断。"""
        return self._requests_used

    @property
    def requests_remaining(self) -> int:
        """剩余可用请求数。"""
        return self._budget.max_requests - self._requests_used

    @property
    def budget(self) -> RequestBudget:
        """本客户端的请求/等待预算（只读）。"""
        return self._budget

    @property
    def rate_limit(self) -> RateLimitRegistry:
        """本客户端使用的限速 registry（用于发现实际限额与证据快照）。"""
        return self._rate_limit

    @property
    def stats(self) -> Dict[str, Any]:
        """
        证据统计：请求数、三类等待秒数（排程/冷却/退避）、限流响应次数与各桶账本快照。

        为什么分开三类等待：报告必须区分“主动限速等待”“服务端要求的冷却”与“故障退避”，
        不能把它们混成一个墙钟数字。
        """
        return {
            "requests_used": self._requests_used,
            "throttle_wait_seconds": self._throttle_wait_seconds,
            "cooldown_wait_seconds": self._cooldown_wait_seconds,
            "backoff_wait_seconds": self._backoff_wait_seconds,
            "rate_limited_responses": self._rate_limited_responses,
            "buckets": self._rate_limit.snapshot(),
        }

    def _total_wait_seconds(self) -> float:
        return self._throttle_wait_seconds + self._cooldown_wait_seconds + self._backoff_wait_seconds

    def _check_wait_budget(self, *, context: str) -> None:
        """总等待预算耗尽必须明确失败，而不是继续无限等待或返回部分数据。"""
        if self._total_wait_seconds() > self._budget.max_total_wait_seconds:
            raise BudgetExhaustedError(
                f"{context}: 总等待时间 {self._total_wait_seconds():.1f}s 超过预算 "
                f"max_total_wait_seconds={self._budget.max_total_wait_seconds}；"
                "本次扫描未完成，不得按完整数据对待。"
            )

    def get_text(
        self,
        *,
        url: str,
        params: Mapping[str, Any],
        context: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        发起一次 GET 并返回响应文本。

        为什么需要它：不是所有来源都返回 JSON（例如 Stooq 返回 CSV）。
        可靠性、预算与状态判定与 `get_json` 完全共用，只是不做 JSON 解析。

        Args:
            url: 完整 URL。
            params: 查询参数。
            context: 报错上下文。
            headers: 本次请求附加头（例如 API key）；不会写入异常信息。

        Returns:
            响应文本。
        """
        return self._request(url=url, params=params, context=context, headers=headers).text

    def get_json(
        self,
        *,
        url: str,
        params: Mapping[str, Any],
        context: str,
        business_check: Optional[Callable[[Any], None]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        发起一次 GET 并返回解析后的 JSON 载荷。

        失败语义（全部抛异常，绝不返回“空数据”冒充成功）：
            - 预算耗尽 -> BudgetExhaustedError
            - 可重试状态重试耗尽 / 连接超时重试耗尽 -> HttpStatusError
            - 不可重试的非 2xx -> HttpStatusError（立即失败，不重试）
            - JSON 解析失败 -> ResponseSchemaError
            - business_check 判定失败 -> BusinessStatusError（由回调抛出）

        Args:
            url: 完整 URL。
            params: 查询参数；调用方负责保证其中不含凭据明文（凭据走 headers）。
            context: 报错上下文（数据集/端点标识）。
            business_check: 业务状态校验回调；收到已解析载荷，失败时自行抛 BusinessStatusError。
            headers: 本次请求附加头（例如 API key）；不会写入异常信息。

        Returns:
            解析后的 JSON 对象。
        """
        response = self._request(url=url, params=params, context=context, headers=headers)
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResponseSchemaError(
                f"{context}: 响应不是合法 JSON，响应片段: {response.text[:200]!r}"
            ) from exc
        if business_check is not None:
            # 业务状态与 HTTP 状态是两件事：200 不代表业务成功。
            business_check(payload)
        return payload

    def _backoff(self, attempt: int, *, context: str) -> None:
        """故障退避：第 i 次重试等待 backoff_seconds * (i + 1)，计入总等待预算。"""
        wait = self._budget.backoff_seconds * (attempt + 1)
        self._backoff_wait_seconds += wait
        self._check_wait_budget(context=context)
        self._sleep(wait)

    def _request(
        self,
        *,
        url: str,
        params: Mapping[str, Any],
        context: str,
        headers: Optional[Dict[str, str]],
    ):
        """
        执行一次带预算与有限重试的 GET，并把所有失败都变成异常。

        失败语义（绝不返回"空数据"冒充成功）：
            - 预算耗尽 -> BudgetExhaustedError
            - 可重试状态 / 连接超时重试耗尽 -> HttpStatusError
            - 不可重试的非 2xx -> HttpStatusError（立即失败，不重试）

        Args:
            url: 完整 URL。
            params: 查询参数；调用方负责保证其中不含需要保密的凭据。
            context: 报错上下文。
            headers: 附加头；不会写入异常信息。

        Returns:
            `requests.Response`（已确认为 2xx）。
        """
        last_error: Optional[BaseException] = None
        # attempt 0 为首次尝试，其余为重试；总尝试次数 = max_retries + 1。
        for attempt in range(self._budget.max_retries + 1):
            if self._requests_used >= self._budget.max_requests:
                raise BudgetExhaustedError(
                    f"{context}: 请求预算已耗尽（max_requests={self._budget.max_requests}）；"
                    "本次扫描未完成，不得按完整数据对待。"
                )
            # 发送前排程：成功请求与重试同样计费；等待也计入总等待预算。
            self._throttle_wait_seconds += self._rate_limit.before_send(url=url, params=params, context=context)
            self._check_wait_budget(context=context)
            self._requests_used += 1
            try:
                response = self._session.get(
                    url,
                    params=dict(params),
                    headers=headers,
                    timeout=(self._budget.connect_timeout, self._budget.read_timeout),
                )
            except requests.RequestException as exc:
                # 连接/读取类异常可重试；重试耗尽后必须抛错，不能跳过时间段继续。
                last_error = exc
                if attempt < self._budget.max_retries:
                    self._backoff(attempt, context=context)
                    continue
                raise HttpStatusError(
                    f"{context}: 网络请求失败且重试耗尽（attempts={attempt + 1}）: {redact_credentials(str(exc))}"
                ) from exc

            status = response.status_code
            response_headers = response.headers
            # 响应头校正必须先于状态判定：429 响应同样带已用权重。
            self._rate_limit.after_response(url=url, headers=response_headers)
            if status in RATE_LIMITED_STATUS:
                self._rate_limited_responses += 1
                # 418 在 penalty 内直接抛 SourceBannedError；429 返回应冷却的秒数。
                cooldown = self._rate_limit.penalty(url=url, status=status, headers=response_headers, context=context)
                last_error = HttpStatusError(f"{context}: HTTP 429 限流，冷却 {cooldown:.1f}s")
                if attempt < self._budget.max_retries:
                    self._cooldown_wait_seconds += cooldown
                    self._check_wait_budget(context=context)
                    self._sleep(cooldown)
                    continue
                raise HttpStatusError(
                    f"{context}: HTTP 429 重试耗尽（attempts={attempt + 1}），"
                    "按限流处理，不以无限重试绕过。"
                )
            if status in RETRYABLE_STATUS:
                last_error = HttpStatusError(f"{context}: 可重试状态 {status}")
                if attempt < self._budget.max_retries:
                    self._backoff(attempt, context=context)
                    continue
                raise HttpStatusError(
                    f"{context}: HTTP {status} 重试耗尽（attempts={attempt + 1}），"
                    "按服务端故障处理，不以无限重试绕过。"
                )
            if not (200 <= status < 300):
                # 不可重试的失败（参数错误、鉴权失败、404 等）必须立即暴露。
                raise HttpStatusError(
                    f"{context}: HTTP {status} 非成功状态，响应片段: {response.text[:200]!r}"
                )
            return response

        # 理论不可达：循环内所有分支要么 return 要么 raise。
        raise HttpStatusError(f"{context}: 请求未能完成，最后错误: {last_error!r}")
