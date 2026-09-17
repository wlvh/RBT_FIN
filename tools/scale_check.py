# -*- coding: utf-8 -*-
"""
贴近真实规模的一次性运行验证（Issue #5 §6 / trading#105 §4）。

功能或目的：
    回答三个只有在真实规模下才有答案的问题：
        1. 生产规模（约 5 年 1m ≈ 300 万行）下，校验 + 内容寻址存储 + legacy 导出
           的耗时、峰值内存与磁盘占用是多少？
        2. 真实 HTTP 分页在中等规模（30 天）下需要多少请求、多长时间？
        3. 发布中途失败后，旧正式状态是否仍然完好、能否恢复？

    本脚本只做测量与断言，不产生任何生产副作用；缓存目录由 `--work-dir` 显式给出。

调用关系：
    独立执行。`--skip-network` 时完全离线（只跑规模与恢复两项）。

用法：
    python tools/scale_check.py --work-dir /tmp/scale --rows 2900000
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rbt_fin.contracts import COVERAGE_COMPLETE, DataRequest
from rbt_fin.errors import RbtFinError
from rbt_fin.exports import export_trading_csv
from rbt_fin.http import HttpClient, RequestBudget, build_session
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_klines_dataset, build_spot_mirror_dataset
from rbt_fin.storage import PHASE_BEFORE_MANIFEST_REPLACE, DatasetStore
from rbt_fin.validation import validate_strict_grid_ohlcv


def _peak_rss_mb() -> float:
    """返回本进程峰值常驻内存（MB）。`ru_maxrss` 在 Linux 上是 KB，在 macOS 上是字节，必须按平台换算。"""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _synthetic_minutes(*, start: pd.Timestamp, rows: int) -> pd.DataFrame:
    """
    生成指定行数的合法分钟 OHLCV。

    为什么用合成数据做规模测量：
        规模验证关心的是“这么多行会不会撑爆内存/磁盘/耗时”，与数值本身无关；
        真实协议正确性由 realsource 测试与离线 fixture 负责。
    """
    index = pd.date_range(start, periods=rows, freq="1min", tz="UTC", name="datetime")
    walk = np.cumsum(np.sin(np.arange(rows) / 997.0)) + 30000.0
    return pd.DataFrame(
        {
            "open": walk,
            "high": walk + 5.0,
            "low": walk - 5.0,
            "close": walk + 1.0,
            "volume": np.abs(np.cos(np.arange(rows) / 337.0)) + 0.1,
        },
        index=index,
    )


def run_scale(*, work_dir: Path, rows: int) -> dict:
    """
    生产规模的校验 -> 发布 -> 读回 -> 导出测量。

    Args:
        work_dir: 工作目录。
        rows: 行数（约 2.9e6 对应 trading 当前的 5 年 1m）。

    Returns:
        测量结果字典。
    """
    dataset = build_klines_dataset()
    start = pd.Timestamp("2021-01-01", tz="UTC")
    frame = _synthetic_minutes(start=start, rows=rows)
    request = DataRequest(start=start, end_exclusive=frame.index[-1] + pd.Timedelta(minutes=1))

    # 为什么不用 tracemalloc 测峰值：它会跟踪每一次分配，对本路径这种
    # “大量短命字符串”的负载可以把耗时放大 4 倍以上，测出来的时间不是生产真实值。
    # 峰值内存改用零开销的 ru_maxrss（内核统计），时间才是可信的。
    timings = {}

    t0 = time.monotonic()
    coverage = validate_strict_grid_ohlcv(frame, dataset=dataset, request=request)
    timings["validate_seconds"] = round(time.monotonic() - t0, 2)
    assert coverage.status == COVERAGE_COMPLETE, coverage.note

    store = DatasetStore(root=work_dir / "cache", dataset_id=dataset.dataset_id)
    t0 = time.monotonic()
    with store.lock():
        manifest = store.publish(
            frame=frame, columns=dataset.columns, time_column=dataset.time_column,
            previous=None,
            manifest_fields={
                "identity": dict(dataset.identity), "params": dict(dataset.params),
                "time_meaning": dataset.time_meaning,
                "schema_version": dataset.schema_version,
                "transform_version": dataset.transform_version,
                "actual_start": frame.index[0].isoformat(),
                "actual_end": frame.index[-1].isoformat(),
                "coverage": coverage.to_manifest_dict(),
                "fetched_at": start.isoformat(), "last_verified_at": start.isoformat(),
                "code_version": "scale-check", "point_in_time_verified": False,
                "note": "", "extra": {},
            },
        )
    timings["publish_seconds"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    readback = store.read_frame(manifest, verify_content=True)
    timings["read_verify_seconds"] = round(time.monotonic() - t0, 2)
    assert len(readback) == rows

    t0 = time.monotonic()
    record = export_trading_csv(
        frame=readback, target_path=work_dir / "BTCUSDT_future_1m.csv", interval="1m",
        start=request.start, end_exclusive=request.end_exclusive,
        source_content_sha256=manifest.content_sha256,
    )
    timings["export_seconds"] = round(time.monotonic() - t0, 2)

    return {
        "rows": rows,
        "timings": timings,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "content_bytes_mb": round(manifest.content_bytes / 1024 / 1024, 1),
        "export_bytes_mb": round(record.export_bytes / 1024 / 1024, 1),
        "content_sha256": manifest.content_sha256,
        "export_sha256": record.export_sha256,
    }


def run_failure_recovery(*, work_dir: Path) -> dict:
    """
    发布中途失败后的恢复检查：旧正式状态必须完好且可读。

    Returns:
        恢复检查结果。
    """
    dataset = build_klines_dataset(dataset_id="scale_recovery_probe")
    store = DatasetStore(root=work_dir / "cache", dataset_id=dataset.dataset_id)
    frame = _synthetic_minutes(start=pd.Timestamp("2021-01-01", tz="UTC"), rows=1000)
    fields = {
        "identity": dict(dataset.identity), "params": dict(dataset.params),
        "time_meaning": dataset.time_meaning, "schema_version": dataset.schema_version,
        "transform_version": dataset.transform_version,
        "actual_start": frame.index[0].isoformat(), "actual_end": frame.index[-1].isoformat(),
        "coverage": {"status": "complete", "checked_start": frame.index[0].isoformat(),
                     "checked_end_exclusive": frame.index[-1].isoformat(), "scan_complete": True},
        "fetched_at": "2021-01-01T00:00:00+00:00", "last_verified_at": "2021-01-01T00:00:00+00:00",
        "code_version": "scale-check", "point_in_time_verified": False, "note": "", "extra": {},
    }
    with store.lock():
        good = store.publish(
            frame=frame, columns=dataset.columns, time_column=dataset.time_column,
            previous=None, manifest_fields=fields,
        )

    def boom(phase: str) -> None:
        if phase == PHASE_BEFORE_MANIFEST_REPLACE:
            raise RuntimeError("模拟提交前失败")

    failed = False
    with store.lock():
        try:
            store.publish(
                frame=_synthetic_minutes(start=pd.Timestamp("2021-01-01", tz="UTC"), rows=2000),
                columns=dataset.columns, time_column=dataset.time_column,
                previous=good, manifest_fields=fields, phase_hook=boom,
            )
        except RuntimeError:
            failed = True
    current = store.read_manifest()
    return {
        "publish_failed_as_expected": failed,
        "manifest_still_points_to_old": current.content_sha256 == good.content_sha256,
        "old_rows_readable": len(store.read_frame(current, verify_content=True)),
    }


def run_network(*, work_dir: Path, days: int) -> dict:
    """
    真实 HTTP 中等规模分页测量。

    Args:
        work_dir: 工作目录。
        days: 天数；30 天 1m = 43200 行。

    Returns:
        测量结果；来源不可达时返回 `{"skipped": ...}`。
    """
    dataset = build_spot_mirror_dataset(limit=1000)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(days=days)
    client = HttpClient(
        session=build_session(),
        budget=RequestBudget(max_requests=days * 2 + 20, max_retries=2, read_timeout=30.0),
    )
    t0 = time.monotonic()
    try:
        result = acquire_dataset(
            dataset=dataset, request=DataRequest(start=start, end_exclusive=end),
            store_root=work_dir / "cache", now_utc=pd.Timestamp.now(tz="UTC"), http=client,
        )
    except RbtFinError as exc:
        return {"skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {
        "days": days,
        "rows": result.coverage.rows,
        "coverage_status": result.coverage.status,
        "requests_used": client.requests_used,
        "seconds": round(time.monotonic() - t0, 2),
        "content_sha256": result.content_sha256,
    }


def main() -> int:
    """解析参数、执行三项测量并打印 JSON 报告。"""
    parser = argparse.ArgumentParser(description="贴近真实规模的一次性运行验证")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--rows", type=int, default=2_900_000, help="生产规模行数（约 5 年 1m）")
    parser.add_argument("--network-days", type=int, default=30)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scale": run_scale(work_dir=work_dir, rows=args.rows),
        "failure_recovery": run_failure_recovery(work_dir=work_dir),
        "network": {"skipped": "--skip-network"} if args.skip_network
        else run_network(work_dir=work_dir, days=args.network_days),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
