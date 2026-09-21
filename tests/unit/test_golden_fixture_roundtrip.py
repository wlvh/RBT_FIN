# -*- coding: utf-8 -*-
"""
固定 fixture -> 公共流程 -> golden CSV 的字节级回路（WP3 的交接件）。

为什么要有 golden 文件：
    trading#105 的“固定输入离线对照”需要一个**两边都能复现**的确定性输入。
    这里用固定协议 fixture 跑完整条真实链路（解析 -> 校验 -> 内容寻址存储 -> legacy 导出），
    并把导出文件的字节哈希固化下来。任何改动导致字节漂移都会在这里失败。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from rbt_fin.contracts import COVERAGE_COMPLETE, DataRequest
from rbt_fin.exports import export_trading_csv
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_funding_dataset, build_klines_dataset
from rbt_fin.sources.binance_funding import funding_daily_mean

from tests.conftest import FakeResponse, load_fixture, make_client

GOLDEN_CSV = Path(__file__).parent.parent / "fixtures" / "golden_BTCUSDT_future_1m.csv"

# golden 导出文件的字节哈希。它固定了：列序、无时区秒精度文本、浮点序列化与行终止符。
GOLDEN_EXPORT_SHA256 = "995275f4a8bd26c9dbac35bfc7c3c44821578590ac08320b43780169e7db2b64"


def _fixture_handler(fixture_name: str):
    """把固定 fixture 变成一个按 startTime 服务的假端点。"""
    data = load_fixture(fixture_name)
    rows = [row for page in data["pages"] for row in page["rows"]]

    def handler(url, params):
        start_ms = params["startTime"]
        end_ms = params["endTime"]
        if isinstance(rows[0], dict):
            wanted = [r for r in rows if start_ms <= r["fundingTime"] <= end_ms]
        else:
            wanted = [r for r in rows if start_ms <= r[0] <= end_ms]
        return FakeResponse(status_code=200, payload=wanted[: params["limit"]])

    return handler


def test_klines_fixture_produces_stable_golden_csv(cache_root, tmp_path):
    """
    固定 fixture 必须稳定产出同一份 legacy CSV 字节。

    这同时验证：解析 -> 校验 -> 存储 -> 导出的整条链路在离线条件下完全确定。
    """
    dataset = build_klines_dataset()
    client, _ = make_client(_fixture_handler("binance_klines_btcusdt_1m_pages.json"), max_requests=20)
    result = acquire_dataset(
        dataset=dataset,
        request=DataRequest(
            start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-01 00:10", tz="UTC"),
        ),
        store_root=cache_root,
        now_utc=pd.Timestamp("2026-01-01 12:00", tz="UTC"),
        http=client,
    )
    assert result.coverage.status == COVERAGE_COMPLETE
    assert result.coverage.rows == 10

    target = tmp_path / "BTCUSDT_future_1m.csv"
    record = export_trading_csv(
        frame=result.frame, target_path=target, interval="1m",
        start=result.requested_start, end_exclusive=result.requested_end_exclusive,
        source_content_sha256=result.content_sha256,
    )
    produced = target.read_bytes()
    assert record.export_sha256 == hashlib.sha256(produced).hexdigest()
    assert produced == GOLDEN_CSV.read_bytes(), (
        "导出字节与 golden 文件不一致；若确为有意的格式变更，必须同时更新 golden 与 "
        "csv_format_version，并通知 trading 侧重新归档批次输入。"
    )
    assert record.export_sha256 == GOLDEN_EXPORT_SHA256


def test_funding_fixture_raw_then_daily_mean(cache_root):
    """
    固定 fixture 下的资金费率：原始记录条数与日均值都是手算可核对的。

    手算：每天三条 0.0001 / 0.0002 / 0.0003 -> 日均 0.0002。
    """
    dataset = build_funding_dataset()
    client, _ = make_client(_fixture_handler("binance_funding_btcusdt_pages.json"), max_requests=20)
    result = acquire_dataset(
        dataset=dataset,
        request=DataRequest(
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end_exclusive=pd.Timestamp("2026-01-03", tz="UTC"),
            require_full_coverage=False,
        ),
        store_root=cache_root,
        now_utc=pd.Timestamp("2026-01-03 12:00", tz="UTC"),
        http=client,
    )
    assert result.coverage.rows == 6
    daily = funding_daily_mean(result.frame)
    assert len(daily) == 2
    assert daily["funding_rate"].round(12).tolist() == [0.0002, 0.0002]
