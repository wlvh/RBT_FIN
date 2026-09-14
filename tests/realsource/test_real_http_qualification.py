# -*- coding: utf-8 -*-
"""
真实来源资格验证（默认不执行；需 `-m realsource`）。

为什么单独标记：
    Issue #5 §8 要求“普通离线测试禁止外部网络，真实来源测试单独标记”。
    这些用例会真的发出 HTTP 请求，结果依赖运行环境的网络与地域可达性。

执行：
    python -m pytest -m realsource -q -s

覆盖：
    - 走**新代码的实际 HTTP 路径**完成一次分页取数 -> 校验 -> 内容寻址存储 -> legacy 导出。
    - 地域限制（HTTP 451）等不可达情形被明确 skip 并打印证据，不伪装成通过。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from rbt_fin.contracts import COVERAGE_COMPLETE, DataRequest
from rbt_fin.errors import HttpStatusError, RbtFinError
from rbt_fin.exports import export_trading_csv
from rbt_fin.http import HttpClient, RequestBudget, build_session
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import (
    build_funding_dataset,
    build_klines_dataset,
    build_spot_mirror_dataset,
)

pytestmark = pytest.mark.realsource

# 固定的历史窗口：用已收盘的过去区间，保证结果可复现，也避免依赖“现在”。
WINDOW_START = pd.Timestamp("2026-01-01 00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2026-01-01 00:30", tz="UTC")


def _client(*, max_requests: int = 12) -> HttpClient:
    return HttpClient(
        session=build_session(),
        budget=RequestBudget(max_requests=max_requests, max_retries=1, read_timeout=20.0),
    )


def _skip_if_unreachable(exc: RbtFinError) -> None:
    """
    把“环境不可达”与“实现失败”分开报告。

    地域限制（451）、CloudFront 国家封锁（403）属于环境阻塞，必须 skip 并打印证据；
    其余失败是实现问题，必须让测试失败。
    """
    message = str(exc)
    if isinstance(exc, HttpStatusError) and ("HTTP 451" in message or "HTTP 403" in message):
        pytest.skip(f"来源在当前运行环境不可达（地域限制），非实现失败：{message[:200]}")
    raise exc


def test_real_klines_pagination_and_export(tmp_path, capsys):
    """
    真实 HTTP：分页取数 -> 严格网格校验 -> 内容寻址存储 -> legacy CSV 导出。

    这里用 **limit=10** 强制真实分页（30 分钟需要 3 页），
    小页跨页即可证明分页协议，不需要反复下载多年历史。
    """
    dataset = build_spot_mirror_dataset(limit=10)
    client = _client()
    try:
        result = acquire_dataset(
            dataset=dataset,
            request=DataRequest(start=WINDOW_START, end_exclusive=WINDOW_END),
            store_root=tmp_path / "cache",
            now_utc=pd.Timestamp.now(tz="UTC"),
            http=client,
        )
    except RbtFinError as exc:
        _skip_if_unreachable(exc)
        return

    assert result.coverage.status == COVERAGE_COMPLETE
    assert result.coverage.rows == 30
    assert client.requests_used >= 3, "limit=10 覆盖 30 分钟必须真的翻页"

    target = tmp_path / "BTCUSDT_spot_1m.csv"
    record = export_trading_csv(
        frame=result.frame, target_path=target, interval="1m",
        start=WINDOW_START, end_exclusive=WINDOW_END,
        source_content_sha256=result.content_sha256,
    )
    evidence = {
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": dict(result.identity),
        "requests_used": client.requests_used,
        "coverage": result.coverage.to_manifest_dict(),
        "content_sha256": result.content_sha256,
        "export_sha256": record.export_sha256,
        "first_rows": target.read_text(encoding="utf-8").split("\n")[:3],
    }
    # 脱敏证据：只有身份、范围、哈希与行数，不含任何凭据（该端点本就无需鉴权）。
    print("\nREAL_SOURCE_EVIDENCE " + json.dumps(evidence, ensure_ascii=False))


def test_real_usdm_futures_qualification(tmp_path):
    """
    生产承诺的 USDⓈ-M 期货端点资格验证。

    在被地域限制的环境里会 skip 并打印证据；**不得**用现货镜像的通过冒充它。
    """
    client = _client()
    try:
        result = acquire_dataset(
            dataset=build_klines_dataset(limit=10),
            request=DataRequest(start=WINDOW_START, end_exclusive=WINDOW_END),
            store_root=tmp_path / "cache",
            now_utc=pd.Timestamp.now(tz="UTC"),
            http=client,
        )
    except RbtFinError as exc:
        _skip_if_unreachable(exc)
        return
    assert result.coverage.status == COVERAGE_COMPLETE
    assert result.coverage.rows == 30


def test_real_funding_rate_qualification(tmp_path):
    """
    资金费率端点资格验证：固定截止、原始记录、分页范围。

    同样在地域限制下 skip；不把未执行的验证标成通过。
    """
    client = _client()
    try:
        result = acquire_dataset(
            dataset=build_funding_dataset(limit=10),
            request=DataRequest(
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end_exclusive=pd.Timestamp("2026-01-08", tz="UTC"),
                require_full_coverage=False,
            ),
            store_root=tmp_path / "cache",
            now_utc=pd.Timestamp.now(tz="UTC"),
            http=client,
        )
    except RbtFinError as exc:
        _skip_if_unreachable(exc)
        return
    assert result.coverage.rows > 0
    assert result.frame.index.is_monotonic_increasing
    assert result.frame.index.is_unique


@pytest.mark.parametrize("start_time_mode", ["unset", "zero", "positive"])
def test_real_funding_start_time_semantics(start_time_mode):
    """
    资金费率端点的 `startTime` 语义资格测试：未传 / 0 / 正值分别观察。

    Issue #5 §5 明确“K 线的 startTime=0 行为与资金费率不能相互推导”。
    这里不把转述的单次观察当证明，而是对本版本实际发出的请求做记录。
    本库的实现永远只发送正值；0 与未传只在这里作为**观察项**执行。
    """
    from rbt_fin.sources.binance_funding import FUNDING_URL

    params = {"symbol": "BTCUSDT", "limit": 5}
    if start_time_mode == "zero":
        params["startTime"] = 0
    elif start_time_mode == "positive":
        params["startTime"] = 1

    client = _client(max_requests=2)
    try:
        payload = client.get_json(url=FUNDING_URL, params=params, context=f"funding-{start_time_mode}")
    except RbtFinError as exc:
        _skip_if_unreachable(exc)
        return
    assert isinstance(payload, list)
    observed = [item["fundingTime"] for item in payload][:3]
    print(f"\nSTART_TIME_OBSERVATION mode={start_time_mode} first_funding_times={observed}")
