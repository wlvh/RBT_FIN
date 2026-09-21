# -*- coding: utf-8 -*-
"""
rbt_fin：统一数据层的公开入口。

功能或目的：
    对外暴露公共流程、合同类型与导出能力。

导入副作用约束（Issue #5 §7，由 `tests/unit/test_import_side_effects.py` 守护）：
    `import rbt_fin` 不联网、不读行情、不扫描 cwd 的 `.env`、不修改进程环境。
    路径、时钟、HTTP 会话与凭据全部在调用入口显式传入，模块级不持有任何这些对象。

调用关系：
    contracts（合同） -> validation（校验） -> sources（来源适配） -> storage（提交）
    -> pipeline（编排） -> exports（legacy CSV 导出） / cli（命令行）。
"""

from __future__ import annotations

from .__about__ import LEGACY_CSV_FORMAT_VERSION, __version__
from .contracts import (
    COVERAGE_COMPLETE,
    COVERAGE_PARTIAL,
    COVERAGE_UNVERIFIED,
    MODE_AUTO,
    MODE_FORCE,
    MODE_OFFLINE,
    AcquireContext,
    AcquireResult,
    CoverageReport,
    DataRequest,
    DataResult,
    Dataset,
    GridSpec,
)
from .errors import (
    AcquisitionError,
    BudgetExhaustedError,
    BusinessStatusError,
    CacheCorruptionError,
    CacheEligibilityError,
    ConfigError,
    ContractError,
    CoverageError,
    HttpStatusError,
    LockError,
    OfflineUnavailableError,
    PaginationError,
    PublishError,
    PublishedUnverifiedError,
    RbtFinError,
    ResponseSchemaError,
    RevisionConflictError,
    SourceBannedError,
    SourceBlockedError,
    StaleCacheError,
    StorageError,
    TimeContractError,
    ValueContractError,
)
from .exports import ExportRecord, export_trading_csv
from .http import HttpClient, RequestBudget, build_session
from .ratelimit import RateLimitConfig, RateLimitRegistry, discover_binance_futures_limit, shared_registry
from .pipeline import TaskOutcome, acquire_dataset, run_task
from .registry import build_registry, get_dataset
from .storage import DatasetStore, Manifest, atomic_write_bytes, exclusive_lock
from .timeutils import complete_end_exclusive, expected_grid, require_utc
from .validation import validate_ohlcv_values

__all__ = [
    "__version__",
    "LEGACY_CSV_FORMAT_VERSION",
    # 合同
    "Dataset", "DataRequest", "DataResult", "CoverageReport", "GridSpec",
    "AcquireContext", "AcquireResult",
    "COVERAGE_COMPLETE", "COVERAGE_PARTIAL", "COVERAGE_UNVERIFIED",
    "MODE_AUTO", "MODE_FORCE", "MODE_OFFLINE",
    # 流程
    "acquire_dataset", "run_task", "TaskOutcome",
    "build_registry", "get_dataset",
    # 存储与导出（`exclusive_lock` / `atomic_write_bytes` 是供外部消费者复用的写安全原语）
    "DatasetStore", "Manifest", "export_trading_csv", "ExportRecord",
    "exclusive_lock", "atomic_write_bytes",
    # 校验（`validate_ohlcv_values` 是不依赖 Dataset/DataRequest 的纯数值合同）
    "validate_ohlcv_values",
    # HTTP
    "HttpClient", "RequestBudget", "build_session", "RateLimitConfig", "RateLimitRegistry", "discover_binance_futures_limit", "shared_registry",
    # 时间
    "complete_end_exclusive", "expected_grid", "require_utc",
    # 异常
    "RbtFinError", "ConfigError", "AcquisitionError", "HttpStatusError",
    "BusinessStatusError", "ResponseSchemaError", "PaginationError", "BudgetExhaustedError", "SourceBannedError",
    "ContractError", "TimeContractError", "ValueContractError", "CoverageError",
    "RevisionConflictError", "SourceBlockedError", "StorageError", "LockError", "CacheCorruptionError",
    "PublishError", "PublishedUnverifiedError", "CacheEligibilityError", "OfflineUnavailableError", "StaleCacheError",
]
