# -*- coding: utf-8 -*-
"""
测试公共夹具：受控 HTTP 会话、协议 fixture 构造器、固定时钟。

为什么用“假会话 + 真协议载荷”而不是直接 mock 出 DataFrame：
    Issue #5 §8 要求“让真实解析/规范化/存储运行；不要全部替换为永远正确的 DataFrame”。
    这里的 `ScriptedSession` 实现了 `requests.Session.get` 的调用契约，返回真实形状的
    JSON 载荷，因此 `HttpClient` 的超时/重试/预算、适配器的分页/解析、校验与存储
    全部走真实代码路径。

调用关系：
    各测试模块通过 `scripted_client` / `kline_row` / `funding_row` / `fixed_now` 使用。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest

from rbt_fin.http import HttpClient, RequestBudget

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeResponse:
    """最小 `requests.Response` 替身：只暴露被测代码真正使用的接口。"""

    def __init__(self, *, status_code: int, payload: Any = None, text: Optional[str] = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)

    def json(self) -> Any:
        """返回载荷；`text` 非法 JSON 时抛 ValueError，与 requests 行为一致。"""
        if self._payload is None:
            return json.loads(self.text)
        return self._payload


class ScriptedSession:
    """
    受控会话：把 `(url, params)` 交给 handler 决定响应，并记录全部调用。

    Attributes:
        calls: `[(url, params_dict), ...]`，用于断言“零网络请求”“每页都带 endTime”等。
    """

    def __init__(self, handler: Callable[[str, Dict[str, Any]], FakeResponse]) -> None:
        self._handler = handler
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.headers: Dict[str, str] = {}

    def get(self, url: str, params=None, headers=None, timeout=None) -> FakeResponse:
        """模拟 `requests.Session.get`；参数签名必须与真实实现一致。"""
        recorded = dict(params or {})
        self.calls.append((url, recorded))
        return self._handler(url, recorded)


def make_client(
    handler: Callable[[str, Dict[str, Any]], FakeResponse],
    *,
    max_requests: int = 50,
    max_retries: int = 0,
) -> Tuple[HttpClient, ScriptedSession]:
    """
    构造一个使用受控会话的 `HttpClient`。

    Args:
        handler: 响应处理器。
        max_requests: 请求预算。
        max_retries: 重试次数；默认 0 让失败立即暴露，需要测重试的用例显式提高。

    Returns:
        `(client, session)`。
    """
    session = ScriptedSession(handler)
    client = HttpClient(
        session=session,
        budget=RequestBudget(max_requests=max_requests, max_retries=max_retries, backoff_seconds=0.0),
        sleep=lambda _seconds: None,  # 测试里不真的等待
    )
    return client, session


def kline_row(*, open_time_ms: int, open_: float, high: float, low: float, close: float, volume: float) -> list:
    """
    构造一行 Binance K 线协议载荷（12 字段，与真实响应形状一致）。

    Returns:
        list：`[openTime, open, high, low, close, volume, closeTime, qav, trades, tbv, tqv, ignore]`
    """
    return [
        open_time_ms,
        f"{open_:.8f}", f"{high:.8f}", f"{low:.8f}", f"{close:.8f}", f"{volume:.8f}",
        open_time_ms + 59_999,
        "0", 1, "0", "0", "0",
    ]


def funding_row(*, funding_time_ms: int, rate: float, symbol: str = "BTCUSDT") -> dict:
    """构造一条 Binance 资金费率协议载荷。"""
    return {
        "symbol": symbol,
        "fundingTime": funding_time_ms,
        "fundingRate": f"{rate:.8f}",
        "markPrice": "30000.00000000",
    }


def to_ms(value: str) -> int:
    """把 `'2026-01-01 00:00'` 这类 UTC 文本转为毫秒时间戳（测试可读性辅助）。"""
    return int(pd.Timestamp(value, tz="UTC").value // 1_000_000)


def minute_klines(*, start: str, count: int, base_price: float = 30000.0) -> List[list]:
    """
    生成一段连续的分钟 K 线协议载荷。

    Args:
        start: 起始 UTC 文本。
        count: 条数。
        base_price: 基准价；逐分钟 +1，保证 high/low 正确包住 open/close。

    Returns:
        协议行列表。
    """
    rows: List[list] = []
    start_ms = to_ms(start)
    for index in range(count):
        price = base_price + index
        rows.append(kline_row(
            open_time_ms=start_ms + index * 60_000,
            open_=price, high=price + 5.0, low=price - 5.0, close=price + 1.0,
            volume=1.5 + index,
        ))
    return rows


def load_fixture(name: str) -> Any:
    """读取 `tests/fixtures/<name>` 的固定协议载荷。"""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def fixed_now() -> pd.Timestamp:
    """
    固定的任务截止时刻。

    为什么测试必须固定时钟：
        “一次任务只确定一次截止时刻”是数据合同的一部分；用真实时钟会让批次输入
        不可复现，也会让“已完成周期边界”的断言随运行时刻漂移。
    """
    return pd.Timestamp("2026-01-01 01:00:00", tz="UTC")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    """每个用例独立的缓存根目录（显式传入库，不依赖 cwd）。"""
    root = tmp_path / "cache"
    root.mkdir()
    return root
