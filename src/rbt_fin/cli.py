# -*- coding: utf-8 -*-
"""
命令行入口：update / overview / verify / export-trading-csv。

功能或目的：
    提供 Issue #5 §7 要求的“更新/读取、验证、概览和兼容 CSV 导出能力”。
    关键边界：
        - `overview` 是**只读**的，绝不隐式更新任何来源；概览成功只代表报告生成成功，
          不授权训练继续（§3.2）。
        - 一次命令只确定一次截止时刻，向下显式传递。
        - 所有路径、时钟、HTTP 会话都在这里显式构造并注入，库内部不持有全局状态。

调用关系：
    `main()` -> 各子命令 -> `rbt_fin.pipeline` / `rbt_fin.storage` / `rbt_fin.exports`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .__about__ import __version__
from .contracts import MODE_AUTO, MODE_FORCE, MODE_OFFLINE, DataRequest
from .errors import RbtFinError
from .exports import export_trading_csv
from .http import HttpClient, RequestBudget, build_session
from .pipeline import acquire_dataset
from .registry import build_registry, get_dataset, list_registered_ids
from .storage import DatasetStore
from .timeutils import require_utc

EXIT_OK = 0
EXIT_FAILED = 1


def _parse_time(value: str, *, arg_name: str) -> pd.Timestamp:
    """
    解析命令行时间参数为 tz-aware UTC。

    为什么要求显式时区：
        naive 文本在不同机器上会被解释成不同时刻。命令行是最容易出现这个错误的边界，
        因此这里只接受带时区的 ISO-8601（例如 `2026-01-01T00:00:00Z`）。
    """
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise SystemExit(
            f"{arg_name}={value!r} 缺少时区；请使用带时区的 ISO-8601，例如 2026-01-01T00:00:00Z"
        )
    return require_utc(parsed, arg_name=arg_name)


# 本库会用到的凭据环境变量。只在 CLI 入口按名字显式读取；
# 库内部永远只拿到调用方传入的 `credentials` 映射，不自己碰 os.environ（Issue #5 §7）。
CREDENTIAL_ENV_NAMES = ("FRED_API_KEY", "GLASSNODE_API_KEY", "CRYPTOCOMPARE_API_KEY")


def _collect_credentials() -> dict:
    """
    从进程环境显式收集凭据。

    为什么放在 CLI 而不是库里：`import rbt_fin` 不得读取 `.env`、不得修改进程环境。
    凭据注入必须是入口的一个明确动作，且其值不会进入任何数据集参数或 manifest。

    Returns:
        `{环境变量名: 值}`；未设置的名字不会出现在结果里。
    """
    return {name: os.environ[name] for name in CREDENTIAL_ENV_NAMES if os.environ.get(name)}


def _build_http(args: argparse.Namespace) -> Optional[HttpClient]:
    """按模式构造 HTTP 客户端；offline 模式返回 None，保证不可能发出请求。"""
    if args.mode == MODE_OFFLINE:
        return None
    budget = RequestBudget(max_requests=args.max_requests)
    return HttpClient(session=build_session(), budget=budget)


def cmd_update(args: argparse.Namespace) -> int:
    """
    更新单个数据集并打印使用说明（JSON）。

    Returns:
        进程退出码。
    """
    dataset = get_dataset(args.dataset)
    request = DataRequest(
        start=_parse_time(args.start, arg_name="--start"),
        end_exclusive=_parse_time(args.end_exclusive, arg_name="--end-exclusive"),
        mode=args.mode,
        accept_stale_on_refresh_failure=args.accept_stale,
        max_staleness=pd.Timedelta(args.max_staleness).to_pytimedelta() if args.max_staleness else None,
        require_full_coverage=not args.allow_partial,
    )
    result = acquire_dataset(
        dataset=dataset,
        request=request,
        store_root=Path(args.cache_root),
        # 一次命令只取一次时钟：所有下游共享同一截止时刻。
        now_utc=pd.Timestamp.now(tz="UTC"),
        http=_build_http(args),
        local_inputs={"csv_path": args.local_csv} if args.local_csv else {},
        credentials=_collect_credentials(),
    )
    print(json.dumps(result.usage_notes(), ensure_ascii=False, indent=2))
    return EXIT_OK


def cmd_overview(args: argparse.Namespace) -> int:
    """
    只读概览：逐项报告状态、实际范围、最近成功核对、缓存使用和错误。

    绝不触发任何采集。`earliest_date` 表示**已取得数据**的最早日，
    不自动证明来源历史起点；采集失败的缺失率为未知（None）。
    """
    rows: List[dict] = []
    # 逐项列出**全部对外注册标识**（含别名）：基线的每个 dataset_id 都要能在概览里
    # 回答自己的状态，而不是只显示合并后的基础数据集。
    for registered_id in list_registered_ids():
        dataset = get_dataset(registered_id)
        alias_of = dataset.dataset_id if dataset.dataset_id != registered_id else None
        store = DatasetStore(root=Path(args.cache_root), dataset_id=dataset.dataset_id)
        manifest = store.read_manifest()
        if manifest is None:
            rows.append({
                "dataset_id": registered_id,
                "alias_of": alias_of,
                "state": "uninitialized",
                "identity": json.dumps(dict(dataset.identity), ensure_ascii=False, sort_keys=True),
                "earliest_obtained_date": None,
                "actual_end": None,
                "coverage_status": None,
                "coverage_checked_range": None,
                "rows": None,
                "last_verified_at": None,
                "content_sha256": None,
                "remark": dataset.remark,
            })
            continue
        coverage = dict(manifest.coverage)
        rows.append({
            "dataset_id": registered_id,
            "alias_of": alias_of,
            "state": "initialized",
            "identity": json.dumps(dict(manifest.identity), ensure_ascii=False, sort_keys=True),
            "earliest_obtained_date": manifest.actual_start,
            "actual_end": manifest.actual_end,
            "coverage_status": coverage.get("status"),
            "coverage_checked_range": (
                f"[{coverage.get('checked_start')}, {coverage.get('checked_end_exclusive')})"
            ),
            "rows": manifest.rows,
            "last_verified_at": manifest.last_verified_at,
            "content_sha256": manifest.content_sha256,
            "remark": manifest.note,
        })
    frame = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"只读概览已写入 {args.out}（共 {len(frame)} 项）。注意：概览成功不代表数据合格。")
    else:
        print(frame.to_string(index=False))
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """
    验证缓存：重算内容哈希、核对 manifest 与内容一致性。

    Returns:
        全部通过返回 0；任一数据集损坏返回 1。
    """
    failures = 0
    for dataset in build_registry():
        if args.dataset and dataset.dataset_id != args.dataset:
            continue
        store = DatasetStore(root=Path(args.cache_root), dataset_id=dataset.dataset_id)
        manifest = store.read_manifest()
        if manifest is None:
            print(f"[skip] {dataset.dataset_id}: 尚未初始化。")
            continue
        try:
            frame = store.read_frame(manifest, verify_content=True)
        except RbtFinError as exc:
            failures += 1
            print(f"[FAIL] {dataset.dataset_id}: {type(exc).__name__}: {exc}")
            continue
        print(
            f"[ok] {dataset.dataset_id}: rows={len(frame)} sha256={manifest.content_sha256} "
            f"range=[{manifest.actual_start}, {manifest.actual_end}]"
        )
    return EXIT_FAILED if failures else EXIT_OK


def cmd_export_trading_csv(args: argparse.Namespace) -> int:
    """从缓存导出 trading 兼容 CSV，并打印导出记录（含导出文件自身哈希）。"""
    dataset = get_dataset(args.dataset)
    store = DatasetStore(root=Path(args.cache_root), dataset_id=dataset.dataset_id)
    manifest = store.read_manifest()
    if manifest is None:
        print(f"[FAIL] {dataset.dataset_id}: 尚未初始化，无法导出。", file=sys.stderr)
        return EXIT_FAILED
    frame = store.read_frame(manifest, verify_content=True)
    record = export_trading_csv(
        frame=frame,
        target_path=Path(args.target),
        interval=dataset.params["interval"],
        start=_parse_time(args.start, arg_name="--start"),
        end_exclusive=_parse_time(args.end_exclusive, arg_name="--end-exclusive"),
        source_content_sha256=manifest.content_sha256,
    )
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(prog="rbt-fin", description="统一数据层命令行入口")
    parser.add_argument("--version", action="version", version=f"rbt-fin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cache_root(target: argparse.ArgumentParser) -> None:
        target.add_argument("--cache-root", required=True, help="缓存根目录（显式传入，不扫描 cwd）")

    update = sub.add_parser("update", help="取得/刷新单个数据集")
    add_cache_root(update)
    update.add_argument("--dataset", required=True)
    update.add_argument("--start", required=True, help="左闭端点，带时区 ISO-8601")
    update.add_argument("--end-exclusive", required=True, help="右开端点，带时区 ISO-8601")
    update.add_argument("--mode", choices=(MODE_AUTO, MODE_FORCE, MODE_OFFLINE), default=MODE_AUTO)
    update.add_argument("--accept-stale", action="store_true", help="刷新失败时允许使用仍合格的旧缓存")
    update.add_argument("--max-staleness", default=None, help="陈旧上限，例如 '2h'；与 --accept-stale 配套")
    update.add_argument("--allow-partial", action="store_true", help="允许非完整覆盖（非严格任务）")
    update.add_argument("--max-requests", type=int, default=400, help="本次请求预算")
    update.add_argument("--local-csv", default=None, help="本地导入数据集的源文件路径")
    update.set_defaults(func=cmd_update)

    overview = sub.add_parser("overview", help="只读概览（不触发任何采集）")
    add_cache_root(overview)
    overview.add_argument("--out", default=None, help="输出 CSV 路径；缺省打印到终端")
    overview.set_defaults(func=cmd_overview)

    verify = sub.add_parser("verify", help="重算内容哈希并核对 manifest")
    add_cache_root(verify)
    verify.add_argument("--dataset", default=None, help="仅验证指定数据集")
    verify.set_defaults(func=cmd_verify)

    export = sub.add_parser("export-trading-csv", help="导出 trading 兼容 CSV")
    add_cache_root(export)
    export.add_argument("--dataset", required=True)
    export.add_argument("--target", required=True)
    export.add_argument("--start", required=True)
    export.add_argument("--end-exclusive", required=True)
    export.set_defaults(func=cmd_export_trading_csv)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI 主入口。

    Args:
        argv: 参数序列；默认取 `sys.argv[1:]`。

    Returns:
        进程退出码：0 成功，1 失败（数据失败必须让调用方 shell 能据此停止后续步骤）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RbtFinError as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
