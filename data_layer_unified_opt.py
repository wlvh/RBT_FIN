#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：对全部注册项走 `rbt_fin` 公共流程采集一次，并输出原版 10 列概览 CSV。

功能或目的：
    原脚本自带整套采集器/解析器；WP2 之后这些实现已全部迁入 `rbt_fin.sources` 并接入
    公共"缓存资格 → 采集 → 规范化/去重/校验 → 内容寻址发布"流程（RBT_FIN#5 §2-§6）。
    本文件只保留 `--out/--limit` 的原有用途与输出列，不再维护第二份采集/解析实现：
    保留死实现只会让两份代码在数值合同、分页与去重语义上悄悄漂移。

刷新语义（显式）：
    - 默认 `--mode auto`：每个注册项都按公共流程判断缓存资格；缓存合格且未到重新检查时机
      则零请求，否则真实采集并发布到 `--cache-root`。
    - `--mode force`：无视缓存重新查来源；`--mode offline`：零请求，缓存不足的项记录为错误行。
    - 概览成功只代表报告生成成功；失败行的缺失率为未知（`na_ratio` 为空），不授权训练继续。

输出列（与原版一致）：
    dataset_id, category, endpoint, earliest_date, sample_start, sample_end,
    sample_rows, na_cells, na_ratio, remark
    其中 `category` 由数据身份派生（来源/交易所 + 值含义），`endpoint` 记录实现该数据集的
    `rbt_fin` 采集函数（协议 URL 由各来源模块持有，不在此重复）。

用法：
    python data_layer_unified_opt.py --out datasets_unified_summary.csv --limit 300 \\
        --start-date 2020-01-01 [--end-date 2026-04-23] [--cache-root rbt_fin_cache] \\
        [--mode auto|force|offline] [--max-requests 400] [--env-file .env]

调用关系：
    main() -> load_credentials() -> run_overview() -> rbt_fin.acquire_dataset()（每个注册项）
           -> summarize_dataset() -> CSV
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

import rbt_fin
from rbt_fin.cli import CREDENTIAL_ENV_NAMES
from rbt_fin.registry import list_registered_ids

SUMMARY_COLUMNS = (
    "dataset_id", "category", "endpoint", "earliest_date", "sample_start", "sample_end",
    "sample_rows", "na_cells", "na_ratio", "remark",
)
EXIT_OK = 0
EXIT_USAGE = 2


def load_credentials(*, env_file: Optional[Path], environ: Mapping[str, str]) -> Dict[str, str]:
    """
    显式收集凭据：先取进程环境中的已知凭据名，再用 `env_file`（若存在）补充缺失项。

    为什么不写回 `os.environ`：凭据注入是入口的一个明确动作，值只交给 `acquire_dataset(credentials=...)`，
    不进入进程环境、数据集参数或 manifest。

    Args:
        env_file: `KEY=VALUE` 文本文件路径；None 或不存在时跳过。
        environ: 进程环境映射（显式传入便于测试）。

    Returns:
        `{凭据环境变量名: 值}`，只包含注册表声明过的凭据名。
    """
    credentials = {name: environ[name] for name in CREDENTIAL_ENV_NAMES if name in environ and environ[name]}
    if env_file is not None and env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key in CREDENTIAL_ENV_NAMES and key not in credentials and value:
                credentials[key] = value
    return credentials


def _category(dataset) -> str:
    """由数据身份派生分类文本：<来源或交易所>/<值含义或时间含义>。"""
    identity = dataset.identity
    if "exchange" in identity:
        source = identity["exchange"]
    elif "source" in identity:
        source = identity["source"]
    else:
        source = "local"
    meaning = identity["value_meaning"] if "value_meaning" in identity else dataset.time_meaning
    return f"{source}/{meaning}"


def summarize_dataset(*, registered_id: str, dataset, frame: Optional[pd.DataFrame], limit: int,
                      error: Optional[str], end_exclusive: pd.Timestamp) -> dict:
    """
    把一次采集结果压成原版概览行。

    Args:
        registered_id: 对外注册标识（含别名）。
        dataset: 解析后的 Dataset 定义。
        frame: 采集/读取得到的数据（tz-aware UTC 索引）；失败时为 None。
        limit: 自最早日期起抽样的 UTC 日数。
        error: 失败说明；None 表示成功。
        end_exclusive: 本次统一右开截止。

    Returns:
        SUMMARY_COLUMNS 顺序的字典；失败行 `na_ratio` 为 None（缺失率未知）。
    """
    remark_parts = [dataset.remark, f"complete_through={(end_exclusive - pd.Timedelta(days=1)).date().isoformat()}"]
    row = {"dataset_id": registered_id, "category": _category(dataset),
           "endpoint": f"rbt_fin.{dataset.acquire.__module__.split('.')[-1]}:{dataset.acquire.__name__}",
           "earliest_date": None, "sample_start": None, "sample_end": None,
           "sample_rows": 0, "na_cells": None, "na_ratio": None}
    if error is not None:
        remark_parts.append(f"error={error}")
    elif frame is not None and len(frame):
        dates = frame.index.tz_convert("UTC").normalize()
        earliest = dates.min()
        in_sample = dates < earliest + pd.Timedelta(days=limit)
        values = frame.loc[in_sample, list(dataset.value_columns)]
        na_cells = int(values.isna().to_numpy().sum())
        row.update({
            "earliest_date": earliest.date().isoformat(),
            "sample_start": earliest.date().isoformat(),
            "sample_end": dates[in_sample].max().date().isoformat(),
            "sample_rows": int(in_sample.sum()), "na_cells": na_cells,
            "na_ratio": round(na_cells / max(values.size, 1), 6),
        })
    remark_parts.append(f"params={json.dumps(dict(dataset.params), ensure_ascii=False, sort_keys=True)}")
    row["remark"] = " | ".join(part for part in remark_parts if part)
    return row


def run_overview(*, cache_root: Path, start: pd.Timestamp, end_exclusive: pd.Timestamp, limit: int, mode: str,
                 http_factory: Callable[[], Optional[rbt_fin.HttpClient]], credentials: Mapping[str, str],
                 now_utc: pd.Timestamp, registered_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    对每个注册项执行一次公共流程采集并汇总。

    Args:
        cache_root: 内容寻址缓存根目录。
        start / end_exclusive: 统一请求区间 [start, end_exclusive)，tz-aware UTC。
        limit: 抽样日数。
        mode: rbt_fin.MODE_AUTO / MODE_FORCE / MODE_OFFLINE。
        http_factory: 每个数据集调用一次以取得 HttpClient；offline 时应返回 None。
        credentials: 凭据映射（只含注册表声明的名字）。
        now_utc: 本次统一时钟。
        registered_ids: 限定的注册标识；None 表示全部。

    Returns:
        按 (category, dataset_id) 排序的概览 DataFrame。

    Raises:
        rbt_fin 之外的异常原样传播（这些不是"来源失败"，而是程序错误）。
    """
    rows: List[dict] = []
    for registered_id in (registered_ids or list_registered_ids()):
        dataset = rbt_fin.get_dataset(registered_id)
        frame, error = None, None
        try:
            result = rbt_fin.acquire_dataset(
                dataset=dataset,
                request=rbt_fin.DataRequest(start=start, end_exclusive=end_exclusive, mode=mode,
                                            require_full_coverage=False),
                store_root=cache_root, now_utc=now_utc, http=http_factory(), credentials=credentials,
            )
            frame = result.frame
        except rbt_fin.RbtFinError as exc:
            # 来源失败/缓存不足/缺凭据都是概览要如实记录的行级事实，不是程序错误。
            error = f"{type(exc).__name__}: {exc}"
        rows.append(summarize_dataset(registered_id=registered_id, dataset=dataset, frame=frame, limit=limit,
                                      error=error, end_exclusive=end_exclusive))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).sort_values(["category", "dataset_id"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RBT_FIN 兼容概览入口（走 rbt_fin 公共流程；会按 --mode 触发采集）")
    parser.add_argument("--out", default="datasets_unified_summary.csv", help="输出 CSV 路径")
    parser.add_argument("--limit", type=int, default=300, help="从最早日期起抽取的 UTC 日样本长度")
    parser.add_argument("--start-date", required=True, help="统一起始日期，例如 2020-01-01（UTC）")
    parser.add_argument("--end-date", default=None, help="统一结束日期（含）；缺省为最新已完成 UTC 日")
    parser.add_argument("--cache-root", default="rbt_fin_cache", help="内容寻址缓存根目录")
    parser.add_argument("--mode", choices=(rbt_fin.MODE_AUTO, rbt_fin.MODE_FORCE, rbt_fin.MODE_OFFLINE), default=rbt_fin.MODE_AUTO)
    parser.add_argument("--max-requests", type=int, default=400, help="每个数据集的请求预算")
    parser.add_argument("--env-file", default=".env", help="显式加载的 KEY=VALUE 凭据文件；不修改进程环境")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        print("--limit 必须为正整数", file=sys.stderr)
        return EXIT_USAGE
    now_utc = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(args.start_date, tz="UTC")
    # 结束日为"含"语义：右开截止取该日次日 00:00；缺省只用最新已完成 UTC 日，避免混入未收盘数据。
    end_exclusive = (pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)) if args.end_date else now_utc.normalize()
    if end_exclusive <= start:
        print(f"结束日期必须晚于起始日期：start={start.date()} end_exclusive={end_exclusive.date()}", file=sys.stderr)
        return EXIT_USAGE
    print(f"[注意] 兼容入口按 --mode={args.mode} 走 rbt_fin 公共流程；只读概览请使用 `rbt-fin overview --cache-root <dir>`。")
    credentials = load_credentials(env_file=Path(args.env_file) if args.env_file else None, environ=os.environ)

    def http_factory() -> Optional[rbt_fin.HttpClient]:
        if args.mode == rbt_fin.MODE_OFFLINE:
            return None
        return rbt_fin.HttpClient(session=rbt_fin.build_session(), budget=rbt_fin.RequestBudget(max_requests=args.max_requests))

    frame = run_overview(cache_root=Path(args.cache_root), start=start, end_exclusive=end_exclusive, limit=args.limit,
                         mode=args.mode, http_factory=http_factory, credentials=credentials, now_utc=now_utc)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    failures = int(frame["na_ratio"].isna().sum())
    print(f"Wrote {len(frame)} rows to {args.out}（失败/缺失率未知的行：{failures}）")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
