# -*- coding: utf-8 -*-
"""
导入副作用约束：`import rbt_fin` 不联网、不读 cwd 的 `.env`、不修改进程环境、无反向依赖。

为什么用子进程 + 审计钩子而不是直接在测试进程里 import：
    测试进程早已导入过本包，无法再观察“首次导入时发生了什么”。子进程里用
    `sys.addaudithook` 记录 `open` / `socket.connect` 事件，才是对副作用的真实观测。
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

PROBE = textwrap.dedent(
    '''
    import json, os, sys

    opened = []
    connected = []

    def _audit(event, args):
        # open 事件的第一个参数是路径；socket.connect 表示真的发起了网络连接。
        if event == "open":
            opened.append(str(args[0]))
        elif event in ("socket.connect", "socket.getaddrinfo"):
            connected.append(event)

    sys.addaudithook(_audit)

    env_before = dict(os.environ)
    import rbt_fin
    import rbt_fin.cli
    import rbt_fin.registry
    import rbt_fin.sources.binance_klines
    import rbt_fin.sources.binance_funding
    import rbt_fin.sources.local_csv
    env_after = dict(os.environ)

    print(json.dumps({
        "opened": opened,
        "connected": connected,
        "env_changed": env_before != env_after,
        "modules": sorted(m for m in sys.modules if "." not in m),
        "version": rbt_fin.__version__,
    }))
    '''
)


def _run_probe(cwd: Path) -> dict:
    """在指定工作目录下的干净子进程里导入本包，并返回审计结果。"""
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(cwd), capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_import_does_not_read_cwd_dotenv_or_mutate_environment(tmp_path):
    """
    `import rbt_fin` 不得扫描 cwd 的 `.env`，也不得修改进程环境。

    基线 `data_layer_unified_opt.py` 在模块导入时就调用 `load_env_from_dotenv()`，
    把 cwd 的 `.env` 注入 `os.environ`——这正是本约束要消除的行为。
    """
    (tmp_path / ".env").write_text("RBT_FIN_SENTINEL=should-not-be-loaded\n", encoding="utf-8")
    report = _run_probe(tmp_path)

    dotenv_reads = [path for path in report["opened"] if path.endswith(".env")]
    assert dotenv_reads == [], f"导入阶段不得读取 .env，实际读取: {dotenv_reads}"
    assert report["env_changed"] is False, "导入阶段不得修改 os.environ"


def test_import_does_not_touch_network(tmp_path):
    """导入阶段不得发起任何网络连接或域名解析。"""
    report = _run_probe(tmp_path)
    assert report["connected"] == [], f"导入阶段发生了网络活动: {report['connected']}"


def test_import_does_not_read_market_data_files(tmp_path):
    """
    导入阶段不得读取行情/缓存文件。

    只允许打开 Python 自身与依赖包的文件；任何 `.csv` / `manifest.json` 都是违规。
    """
    (tmp_path / "BTCUSDT_future_1m.csv").write_text("datetime,open\n", encoding="utf-8")
    report = _run_probe(tmp_path)
    data_reads = [
        path for path in report["opened"]
        if path.endswith(".csv") or path.endswith("manifest.json")
    ]
    assert data_reads == [], f"导入阶段读取了数据文件: {data_reads}"


def test_no_reverse_dependency_on_trading_or_optimizer(tmp_path):
    """
    无反向依赖：本包不得导入 trading、Optuna 或回测引擎。

    这是跨仓库分工的硬边界——数据层被 trading 消费，不能反过来依赖它。
    """
    report = _run_probe(tmp_path)
    forbidden = {
        "optuna", "numpy_backtest", "trading_core", "trading_data_process",
        "DIY_Volume_numpy", "diy_volume_numpy", "extra_info", "binance_data_handler",
        "sqlalchemy", "yfinance", "ccxt",
    }
    leaked = forbidden.intersection(report["modules"])
    assert leaked == set(), f"检测到反向/重量级依赖: {sorted(leaked)}"


def test_optional_dependency_absence_does_not_block_core(tmp_path):
    """
    来源可选依赖缺失不阻止安装/使用核心行情能力。

    断言方式：核心模块导入后，可选依赖（yfinance）并不在已加载模块中。
    """
    report = _run_probe(tmp_path)
    assert "yfinance" not in report["modules"]
    assert report["version"]
