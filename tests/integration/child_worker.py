# -*- coding: utf-8 -*-
"""
受控子进程工作器：用于“提交前/提交后中断”与“真实双写者”集成测试。

为什么必须是真实子进程：
    Issue #5 §8 要求两个关键中断测试用真实进程终止来验证提交语义。
    在同一进程里抛异常只能证明代码路径，证明不了“进程在提交点被杀死后磁盘上剩下什么”。

角色（第一个命令行参数）：
    publish-kill  在指定发布同步点用 SIGKILL 杀死自己。
    writer        正常跑一次 `acquire_dataset`；可在发布前与父进程做同步点握手。
    lock-prober   先证明数据集写锁确实被他人持有，再发起一次阻塞式 `acquire_dataset`。

所有角色都使用同一个内置的确定性假端点，不产生任何真实网络请求。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from rbt_fin.contracts import DataRequest
from rbt_fin.http import HttpClient, RequestBudget
from rbt_fin.pipeline import acquire_dataset
from rbt_fin.registry import build_klines_dataset
from rbt_fin.storage import DatasetStore

# 子进程之间只通过文件系统上的“同步点文件”握手，不依赖 sleep 猜测时序。
SYNC_POLL_SECONDS = 0.01
SYNC_TIMEOUT_SECONDS = 30.0


class _FakeResponse:
    """最小响应替身，接口与 `requests.Response` 中被使用的部分一致。"""

    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)
        # 真实 requests.Response 总有 headers；限速层读取 X-MBX-USED-WEIGHT-1M（缺失即不校正）。
        self.headers = {}

    def json(self):
        return self._payload


class _FakeSession:
    """确定性假端点：按 `startTime` 返回连续分钟 K 线，不联网。"""

    def __init__(self, *, base_ms: int, count: int):
        self.headers = {}
        self._rows = [
            [
                base_ms + i * 60_000,
                f"{30000.0 + i:.8f}", f"{30005.0 + i:.8f}", f"{29995.0 + i:.8f}",
                f"{30001.0 + i:.8f}", f"{1.0 + i:.8f}",
                base_ms + i * 60_000 + 59_999, "0", 1, "0", "0", "0",
            ]
            for i in range(count)
        ]

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        if url.endswith("/fapi/v1/exchangeInfo"):
            # 0.3.3 起适配器在第一次联网取数前读一次实际限额：回真实形状的最小 rateLimits 载荷。
            return _FakeResponse({"rateLimits": [
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
            ]})
        start_ms = params["startTime"]
        end_ms = params["endTime"]
        wanted = [row for row in self._rows if start_ms <= row[0] <= end_ms]
        return _FakeResponse(wanted[: params["limit"]])


def _wait_for(path: Path) -> None:
    """轮询等待同步点文件出现；超时即失败，避免测试永久挂起。"""
    deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"等待同步点文件超时: {path}")
        time.sleep(SYNC_POLL_SECONDS)


def _touch(path: Path) -> None:
    """创建同步点文件（原子性足够：只用于存在性判断）。"""
    path.write_text("1", encoding="utf-8")


def _client(base: pd.Timestamp, count: int) -> HttpClient:
    return HttpClient(
        session=_FakeSession(base_ms=int(base.value // 1_000_000), count=count),
        budget=RequestBudget(max_requests=50, max_retries=0, backoff_seconds=0.0),
        sleep=lambda _s: None,
    )


def _request(start: str, end: str) -> DataRequest:
    return DataRequest(
        start=pd.Timestamp(start, tz="UTC"), end_exclusive=pd.Timestamp(end, tz="UTC")
    )


def main() -> int:
    """解析角色参数并执行。返回进程退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("publish-kill", "writer", "lock-prober"))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--source-base", required=True, help="假端点数据起点")
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--kill-phase", default=None)
    parser.add_argument("--signal-file", default=None, help="进入同步点后创建的文件")
    parser.add_argument("--wait-file", default=None, help="进入同步点后等待的文件")
    args = parser.parse_args()

    dataset = build_klines_dataset()
    cache_root = Path(args.cache_root)
    now = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    source_base = pd.Timestamp(args.source_base, tz="UTC")

    if args.role == "lock-prober":
        # 先证明锁确实被他人持有，再发起阻塞式获取。这样“第二写者从新状态更新”
        # 这一断言就不依赖 sleep 猜测，而是有明确的因果顺序。
        import fcntl

        store = DatasetStore(root=cache_root, dataset_id=dataset.dataset_id)
        store.dataset_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
        while True:
            handle = store.lock_path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                break  # 锁被他人持有 —— 正是我们要等到的状态
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            if time.monotonic() > deadline:
                raise TimeoutError("等待另一个写者持锁超时")
            time.sleep(SYNC_POLL_SECONDS)
        if args.signal_file:
            _touch(Path(args.signal_file))

    phase_hook = None
    if args.role == "publish-kill":
        def phase_hook(phase: str) -> None:
            if phase != args.kill_phase:
                return
            if args.signal_file:
                _touch(Path(args.signal_file))
            # 立刻用 SIGKILL 结束自己：模拟进程在提交点被强杀，不给任何清理机会。
            os.kill(os.getpid(), signal.SIGKILL)

    elif args.role == "writer" and (args.signal_file or args.wait_file):
        def phase_hook(phase: str) -> None:
            # 在锁内、提交前握手：保证另一个写者确实经历了“锁被占用”。
            if phase != "before_content_write":
                return
            if args.signal_file:
                _touch(Path(args.signal_file))
            if args.wait_file:
                _wait_for(Path(args.wait_file))

    result = acquire_dataset(
        dataset=dataset,
        request=_request(args.start, args.end_exclusive),
        store_root=cache_root,
        now_utc=now,
        http=_client(source_base, args.source_count),
        phase_hook=phase_hook,
    )
    print(json.dumps({"rows": result.coverage.rows, "sha256": result.content_sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
