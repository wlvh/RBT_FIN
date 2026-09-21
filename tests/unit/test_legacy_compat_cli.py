# -*- coding: utf-8 -*-
"""
兼容入口 `data_layer_unified_opt.py` 收敛为薄包装后的合同（Issue #5 §6 / trading#105 S2 清理项）。

功能或目的：
    - 全部注册项（含别名）走 `rbt_fin` 公共流程一次并产出原版 10 列概览；受控端点复用
      `tests/unit/test_full_registry.py` 的按族协议样例，凭据为虚构值。
    - offline 模式 + 空缓存：零请求，每行记录错误、`na_ratio` 为空（缺失率未知），入口仍返回报告。
    - 缓存合格后 offline 复跑：零请求且能读出概览。
    - 死实现已删除：模块不再自带任何采集器/HTTP 调用，也不在导入时读 `.env` 或改环境。
    - 凭据只从进程环境与显式 env 文件收集，值不进入 params/remark。

调用关系：
    pytest 收集本文件；被测模块以仓库根目录脚本形式导入。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import rbt_fin
from rbt_fin.registry import list_registered_ids
from tests.conftest import make_client
from tests.unit.test_full_registry import END, FAKE_CREDENTIALS, NOW, START, _build_session_for

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "data_layer_unified_opt.py"


@pytest.fixture(scope="module")
def compat():
    spec = importlib.util.spec_from_file_location("data_layer_unified_opt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NETWORK_IDS = [i for i in list_registered_ids() if rbt_fin.get_dataset(i).requires_network]


def _controlled_http_factory_for(ids):
    """按数据集族返回受控端点：`run_overview` 对 `ids` 逐个调用工厂一次，这里按同一顺序匹配。"""
    state = {"index": 0}

    def factory():
        dataset = rbt_fin.get_dataset(ids[state["index"]])
        state["index"] += 1
        return make_client(_build_session_for(dataset), max_requests=60)[0]
    return factory


def test_overview_covers_every_registered_id_with_legacy_columns(compat, tmp_path):
    frame = compat.run_overview(
        cache_root=tmp_path / "cache", start=START, end_exclusive=END, limit=300, mode=rbt_fin.MODE_AUTO,
        http_factory=_controlled_http_factory_for(NETWORK_IDS), credentials=FAKE_CREDENTIALS, now_utc=NOW,
        registered_ids=NETWORK_IDS,
    )
    assert list(frame.columns) == list(compat.SUMMARY_COLUMNS)
    assert set(frame["dataset_id"]) == set(NETWORK_IDS) and len(frame) == len(NETWORK_IDS)
    failures = frame[frame["na_ratio"].isna()]
    assert failures.empty, failures[["dataset_id", "remark"]].to_string()
    assert (frame["sample_rows"] > 0).all() and frame["earliest_date"].notna().all()
    assert frame["endpoint"].str.startswith("rbt_fin.").all()
    for value in FAKE_CREDENTIALS.values():
        assert not frame["remark"].str.contains(value).any(), "凭据值不得进入概览"


def test_offline_with_empty_cache_records_errors_instead_of_raising(compat, tmp_path):
    ids = list(list_registered_ids())[:3]
    frame = compat.run_overview(
        cache_root=tmp_path / "cache", start=START, end_exclusive=END, limit=300, mode=rbt_fin.MODE_OFFLINE,
        http_factory=lambda: None, credentials={}, now_utc=NOW, registered_ids=ids,
    )
    assert len(frame) == len(ids) and frame["na_ratio"].isna().all()
    assert frame["remark"].str.contains("error=").all()


def test_offline_rerun_reads_cache_without_requests(compat, tmp_path):
    ids = ["binance_funding_btcusdt"]
    first = compat.run_overview(cache_root=tmp_path / "cache", start=START, end_exclusive=END, limit=300, mode=rbt_fin.MODE_AUTO,
                                http_factory=_controlled_http_factory_for(ids), credentials={}, now_utc=NOW, registered_ids=ids)
    assert first["na_ratio"].notna().all()
    second = compat.run_overview(cache_root=tmp_path / "cache", start=START, end_exclusive=END, limit=300, mode=rbt_fin.MODE_OFFLINE,
                                 http_factory=lambda: None, credentials={}, now_utc=NOW, registered_ids=ids)
    assert second["na_ratio"].notna().all() and second["sample_rows"].iloc[0] == first["sample_rows"].iloc[0]


def test_module_has_no_duplicate_collectors_or_import_side_effects(monkeypatch, tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert names == {"load_credentials", "_category", "summarize_dataset", "run_overview", "build_parser", "main"}
    assert "requests" not in {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
    # 导入不读 cwd 的 .env、不改环境
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FRED_API_KEY=should-not-be-loaded-on-import\n", encoding="utf-8")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("data_layer_unified_opt_import_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import os
    assert "FRED_API_KEY" not in os.environ


def test_load_credentials_prefers_environ_and_reads_env_file_without_mutating(compat, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nFRED_API_KEY='from-file'\nGLASSNODE_API_KEY=from-file-glassnode\nUNRELATED=1\n", encoding="utf-8")
    monkeypatch.delenv("GLASSNODE_API_KEY", raising=False)
    creds = compat.load_credentials(env_file=env_file, environ={"FRED_API_KEY": "from-env"})
    assert creds == {"FRED_API_KEY": "from-env", "GLASSNODE_API_KEY": "from-file-glassnode"}
    import os
    assert "GLASSNODE_API_KEY" not in os.environ


def test_main_rejects_bad_limit_and_inverted_dates(compat, tmp_path, capsys):
    assert compat.main(["--start-date", "2026-01-01", "--limit", "0", "--cache-root", str(tmp_path)]) == compat.EXIT_USAGE
    assert compat.main(["--start-date", "2026-01-05", "--end-date", "2026-01-01", "--cache-root", str(tmp_path)]) == compat.EXIT_USAGE
