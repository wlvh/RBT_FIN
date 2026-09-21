# -*- coding: utf-8 -*-
"""
逐项生成注册表接入状态矩阵（Issue #5 §9 WP2 的记录要求）。

功能或目的：
    对每个对外注册标识回答四件事，并把它们**分开**记录：
        已接入          —— 是否已绑定到公共流程（有 Dataset 定义）。
        离线结果        —— 在受控固定响应下能否 acquire -> persist -> load。
        真实验证结果    —— 用真实 HTTP 走新代码的实际路径能否取得并通过校验。
        阻塞原因        —— 缺 key / 权限 / 地域限制 / 需要本地文件，与代码失败严格分开。

    "缺 key 或来源不可达" 与 "实现有 bug" 是两件事；把它们混在一起就会让
    一次网络封锁看起来像验收通过，或者让真实缺陷藏在 "blocked" 里。

调用关系：
    独立执行。`--offline-only` 时完全不联网，只跑离线判定。

用法：
    python tools/registry_status.py --work-dir /tmp/status --out status.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import rbt_fin
from rbt_fin.contracts import DataRequest
from rbt_fin.errors import (
    AcquisitionError,
    BusinessStatusError,
    ConfigError,
    HttpStatusError,
    SourceBlockedError,
)
from rbt_fin.registry import DATASET_ALIASES, get_dataset, list_registered_ids

# 真实验证使用的小窗口。小页跨页足以证明协议，不需要反复下载多年历史。
REAL_START = pd.Timestamp("2026-01-01", tz="UTC")
REAL_END = pd.Timestamp("2026-01-08", tz="UTC")

CREDENTIAL_ENV_NAMES = ("FRED_API_KEY", "GLASSNODE_API_KEY", "CRYPTOCOMPARE_API_KEY")

# 这些 HTTP 状态表示"来源在当前环境不可达"，属于环境阻塞而不是实现失败。
ENVIRONMENT_BLOCK_MARKERS = ("HTTP 451", "HTTP 403", "HTTP 429")


def classify_failure(exc: BaseException) -> Dict[str, str]:
    """
    把一次失败归类为环境阻塞还是实现问题。

    Args:
        exc: 捕获到的异常。

    Returns:
        `{"status": ..., "reason": ...}`。
    """
    message = str(exc)
    if isinstance(exc, ConfigError) and "缺少凭据" in message:
        return {"status": "blocked_missing_credential", "reason": message[:200]}
    if isinstance(exc, ConfigError) and "local_inputs" in message:
        return {"status": "blocked_needs_local_file", "reason": message[:200]}
    if isinstance(exc, HttpStatusError) and any(m in message for m in ENVIRONMENT_BLOCK_MARKERS):
        return {"status": "blocked_source_unreachable", "reason": message[:200]}
    if isinstance(exc, SourceBlockedError):
        # HTTP 200 + 反爬页：来源侧拦截，与协议漂移是两回事。
        return {"status": "blocked_source_unreachable", "reason": message[:200]}
    if isinstance(exc, HttpStatusError) and "重试耗尽" in message:
        return {"status": "blocked_source_unreachable", "reason": message[:200]}
    if isinstance(exc, (AcquisitionError, BusinessStatusError)):
        return {"status": "source_error", "reason": f"{type(exc).__name__}: {message[:200]}"}
    return {"status": "implementation_failure", "reason": f"{type(exc).__name__}: {message[:200]}"}


def probe_real_source(*, registered_id: str, work_dir: Path) -> Dict[str, Any]:
    """
    用真实 HTTP 走一次实际取数路径。

    Args:
        registered_id: 对外注册标识。
        work_dir: 隔离工作目录（每项独立，避免互相污染缓存）。

    Returns:
        该项的真实验证结果字典。
    """
    dataset = get_dataset(registered_id)
    if not dataset.requires_network:
        return {"status": "not_applicable_local_source",
                "reason": "本地导入数据集，不联网；离线判定已覆盖其解析与存储路径。"}

    store_root = work_dir / registered_id
    if store_root.exists():
        shutil.rmtree(store_root)
    client = rbt_fin.HttpClient(
        session=rbt_fin.build_session(),
        budget=rbt_fin.RequestBudget(max_requests=12, max_retries=1, read_timeout=25.0),
    )
    credentials = {n: os.environ[n] for n in CREDENTIAL_ENV_NAMES if os.environ.get(n)}
    try:
        result = rbt_fin.acquire_dataset(
            dataset=dataset,
            request=DataRequest(
                start=REAL_START, end_exclusive=REAL_END,
                require_full_coverage=dataset.calendar_verified,
            ),
            store_root=store_root,
            now_utc=pd.Timestamp.now(tz="UTC"),
            http=client,
            credentials=credentials,
        )
    except Exception as exc:  # noqa: BLE001 - 这里的职责就是把每种失败如实归类（KeyboardInterrupt 等不吞）
        return {**classify_failure(exc), "requests_used": client.requests_used}
    return {
        "status": "verified",
        "rows": result.coverage.rows,
        "coverage": result.coverage.status,
        "requests_used": client.requests_used,
        "content_sha256": result.content_sha256,
    }


def build_status(*, work_dir: Path, offline_only: bool) -> Dict[str, Any]:
    """
    生成完整状态矩阵。

    Args:
        work_dir: 隔离工作目录。
        offline_only: True 时跳过真实来源探测。

    Returns:
        可序列化的状态报告。
    """
    items: List[Dict[str, Any]] = []
    for registered_id in list_registered_ids():
        dataset = get_dataset(registered_id)
        entry: Dict[str, Any] = {
            "registered_id": registered_id,
            "resolves_to": dataset.dataset_id,
            "alias": registered_id in DATASET_ALIASES,
            "integrated": True,  # 能解析出 Dataset 即已接入公共流程
            "collector": dataset.acquire.__name__,
            "requires_network": dataset.requires_network,
            "credential_env": dataset.params.get("credential_env")
            if isinstance(dataset.params, dict) else None,
            "calendar_verified": dataset.calendar_verified,
            # 离线结果由 tests/unit/test_full_registry.py 逐项参数化保证，
            # 这里只引用那条证据，不在工具里重复实现一份。
            "offline_evidence": "tests/unit/test_full_registry.py::"
                                "test_every_registered_id_acquires_persists_loads_and_hits_cache",
        }
        entry["real_source"] = (
            {"status": "skipped_offline_only", "reason": "--offline-only"}
            if offline_only else probe_real_source(registered_id=registered_id, work_dir=work_dir)
        )
        items.append(entry)

    summary: Dict[str, int] = {}
    for item in items:
        summary[item["real_source"]["status"]] = summary.get(item["real_source"]["status"], 0) + 1
    return {
        "rbt_fin_version": rbt_fin.__version__,
        "registered_ids": len(items),
        "real_source_summary": summary,
        "items": items,
    }


def main() -> int:
    """解析参数、生成状态矩阵并输出 JSON。"""
    parser = argparse.ArgumentParser(description="注册表接入状态矩阵")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", default=None, help="输出 JSON 路径；缺省打印到终端")
    parser.add_argument("--offline-only", action="store_true", help="不做任何真实来源探测")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = build_status(work_dir=work_dir, offline_only=args.offline_only)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"状态矩阵已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
