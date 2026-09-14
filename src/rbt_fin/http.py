# -*- coding: utf-8 -*-
"""
公共 HTTP 下载护栏。

功能或目的：
    Issue #5 §4.1 要求：网络超时、有限重试和请求/页数预算明确；重试耗尽、分页无进展、
    业务错误或结构错误均不能伪装成正常空数据；HTTP 状态与 JSON 业务状态分别检查。

    本模块只负责“单次请求的可靠性与预算”，分页边界由各来源适配器负责
    （`rbt_fin.sources.*`），因为不同端点的游标语义无法互相推导。

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

import requests

from .errors import (
    BudgetExhaustedError,
    BusinessStatusError,
    ConfigError,
    HttpStatusError,
    ResponseSchemaError,
)

# 可重试状态：限流（429/418）与网关/服务端临时故障。
# 为什么不含 4xx 其余状态：参数错误、鉴权失败重试只会掩盖问题并触发更严格的封禁。
RETRYABLE_STATUS = frozenset({418, 429, 500, 502, 503, 504, 520, 521, 522, 524})

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
    """

    max_requests: int = 400
    max_retries: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    backoff_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ConfigError("max_requests 必须为正整数。")
        if self.max_retries < 0:
            raise ConfigError("max_retries 不能为负。")
        for name in ("connect_timeout", "read_timeout", "backoff_seconds"):
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
    ) -> None:
        """
        Args:
            session: 由调用方显式构造的 `requests.Session`。
            budget: 请求预算与超时配置。
            sleep: 退避实现；测试可注入无副作用实现以避免真实等待。
        """
        if not isinstance(budget, RequestBudget):
            raise ConfigError(f"budget 必须是 RequestBudget，实际为 {type(budget)!r}")
        self._session = session
        self._budget = budget
        self._sleep = sleep
        self._requests_used = 0

    @property
    def requests_used(self) -> int:
        """已消耗的请求数（含重试），用于证据归档与预算诊断。"""
        return self._requests_used

    @property
    def requests_remaining(self) -> int:
        """剩余可用请求数。"""
        return self._budget.max_requests - self._requests_used

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
        last_error: Optional[BaseException] = None
        # attempt 0 为首次尝试，其余为重试；总尝试次数 = max_retries + 1。
        for attempt in range(self._budget.max_retries + 1):
            if self._requests_used >= self._budget.max_requests:
                raise BudgetExhaustedError(
                    f"{context}: 请求预算已耗尽（max_requests={self._budget.max_requests}）；"
                    "本次扫描未完成，不得按完整数据对待。"
                )
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
                    self._sleep(self._budget.backoff_seconds * (attempt + 1))
                    continue
                raise HttpStatusError(
                    f"{context}: 网络请求失败且重试耗尽（attempts={attempt + 1}）: {exc}"
                ) from exc

            status = response.status_code
            if status in RETRYABLE_STATUS:
                last_error = HttpStatusError(f"{context}: 可重试状态 {status}")
                if attempt < self._budget.max_retries:
                    self._sleep(self._budget.backoff_seconds * (attempt + 1))
                    continue
                raise HttpStatusError(
                    f"{context}: HTTP {status} 重试耗尽（attempts={attempt + 1}），"
                    "按限流/服务端故障处理，不以无限重试绕过。"
                )
            if not (200 <= status < 300):
                # 不可重试的失败（参数错误、鉴权失败、404 等）必须立即暴露。
                raise HttpStatusError(
                    f"{context}: HTTP {status} 非成功状态，响应片段: {response.text[:200]!r}"
                )

            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResponseSchemaError(
                    f"{context}: 响应不是合法 JSON，响应片段: {response.text[:200]!r}"
                ) from exc

            if business_check is not None:
                # 业务状态与 HTTP 状态分别检查：200 不代表业务成功。
                business_check(payload)
            return payload

        # 理论不可达：循环内所有分支要么 return 要么 raise。
        raise HttpStatusError(f"{context}: 请求未能完成，最后错误: {last_error!r}")
