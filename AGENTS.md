## 文件简介

### 安装包 `src/rbt_fin/`
- `__init__.py`：公开入口；只做名字再导出。`import rbt_fin` 不联网、不读行情、不扫描 cwd 的 `.env`、不修改进程环境（由 `tests/unit/test_import_side_effects.py` 守护）。
- `__about__.py`：包版本与 legacy CSV 格式版本；manifest 的 `code_version` 真相源。
- `errors.py`：公共异常族（配置 / 采集 / 数据合同 / 存储 / 缓存资格）；所有失败都必须传播，禁止“只打印不抛”。
- `timeutils.py`：UTC 时间语义唯一真相源。tz-aware 校验、固定长度周期解析（周/月拒绝）、已完成周期右开边界、期望网格、网格断言（秒/微秒/纳秒三级）。
- `contracts.py`：`Dataset` / `DataRequest` / `DataResult` / `CoverageReport` / `AcquireContext` / `AcquireResult`。刷新间隔、陈旧上限、范围/身份/质量在此分开表达。
- `http.py`：公共下载护栏——超时、有限重试、请求预算；HTTP 状态与业务状态分别检查；会话由调用方显式构造（`build_session()`）。
- `storage.py`：`cache/<dataset>/{data-<sha256>.csv, manifest.json, update.lock}`。内容文件永不原地修改，替换 `manifest.json` 是唯一提交点；`phase_hook` 提供受控中断同步点。
- `validation.py`：严格网格 OHLCV 校验、非规则序列校验、严格数值解析（不使用 `coerce+dropna`）、按时间主键去重（同时间不同内容默认拒绝）。
- `transforms.py`：少量显式转换——按 UTC 日期取日均 / 日内 last、保留基础值的环比（首次记录/零分母不填 0）。
- `pipeline.py`：公共流程编排（`acquire_dataset`）与多数据集任务（`run_task`）。`last_verified_at` 只在真实来源成功解析并核验后推进。
- `registry.py`：数据集注册表；`params` 既是 manifest 记录也是适配器真正读取的参数，不存在第二份硬编码。
- `exports.py`：trading 兼容 CSV 导出；导出文件有**自己的**哈希，与来源内容哈希分开保存。
- `cli.py`：`update` / `overview`（只读，不触发采集）/ `verify` / `export-trading-csv`。
- `sources/binance_klines.py`：分钟行情适配器。固定任务截止、增量只补缺失段、提前空页标记扫描未完成、重复页/游标不推进/越界记录 Fail Fast。
- `sources/binance_funding.py`：资金费率适配器。每页保持固定 `endTime`、先落原始结算记录再聚合、`startTime` 恒为正值（不使用 `0` 哨兵）。
- `sources/local_csv.py`：本地导入适配器。不联网、不修改用户原文件、记录源文件哈希。

### 兼容入口
- `data_layer_unified_opt.py`：保留 `--out/--limit` 的原有用途。**会触发刷新**——每次运行对全部注册项真实采集，不是只读概览。其各来源将在 WP2 迁移到 `rbt_fin` 公共流程。已移除内置 CryptoCompare 备用 key，缺配置时 Fail Fast。

### 测试
- `tests/conftest.py`：受控 HTTP 会话 + 真实协议载荷；固定时钟与独立缓存根目录。
- `tests/unit/test_migrated_guards.py`：迁入 trading 的七项行情护栏及其新增反例。
- `tests/unit/test_time_and_coverage.py`：反例 A（半秒/微秒/纳秒偏移）、反例 B（提前空页尾部未完成）、头/内/尾缺失、越界、未完成周期、UTC/非 UTC/naive。
- `tests/unit/test_pagination_and_failures.py`：多页重叠、重复页、游标不推进、预算耗尽、HTTP/业务失败、字段不全。
- `tests/unit/test_funding_contract.py`：固定截止、先去重后聚合、`startTime` 语义、日内乱序。
- `tests/unit/test_cache_policy.py`：反例 C、零网络请求、offline、刷新失败降级、核对时间不被失败推进。
- `tests/unit/test_storage_publish.py`：发布失败传播、旧状态不变、首次失败无正式数据、损坏哈希拒绝、内容复用。
- `tests/unit/test_revision_and_local.py`：同主键相同/不同内容、历史缩短、零分母不填零、本地导入、低频非网格。
- `tests/unit/test_export_trading_csv.py`：legacy CSV 合同与导出文件自身哈希。
- `tests/unit/test_import_side_effects.py`：导入无副作用、无反向依赖。
- `tests/unit/test_golden_fixture_roundtrip.py`：固定 fixture → 公共流程 → golden CSV 的字节级回路（WP3 交接件）。
- `tests/integration/test_interrupt_and_concurrency.py` + `child_worker.py`：真实子进程的提交前/提交后 SIGKILL，以及真实双写者无覆盖丢失。
- `tests/realsource/test_real_http_qualification.py`：真实来源资格验证；默认被 `-m 'not realsource'` 排除，需显式 `pytest -m realsource`。来源因地域限制不可达时 skip 并打印证据，不伪装成通过。
- `tests/fixtures/`：独立构造的协议 fixture 与 golden CSV；不含任何真实凭据。

### 工具
- `tools/scale_check.py`：贴近真实规模的一次性运行验证。测量生产规模（约 300 万行 1m）的校验/发布/读回/导出耗时与峰值内存、真实 HTTP 中等规模分页请求数，以及发布中途失败后的恢复。`--skip-network` 可完全离线运行。

### 运行
```bash
python -m pytest tests -q          # 全部离线
python -m pytest -m realsource     # 需要真实来源，默认不执行
```

## 架构不变量
1. 一次任务只确定一次截止时刻，并显式向下传递；库内部不读时钟。
2. 公共范围统一 `[start, end_exclusive)`；各端点适配器自行转换为它的参数语义。
3. 刷新间隔（何时再查来源）、陈旧上限（刷新失败后能否用旧数据）、范围/身份/质量分别判定。
4. `last_verified_at` 只在真实来源结果成功解析并按合同核验后推进。
5. 提交前任何失败都不改变旧正式状态；替换 `manifest.json` 是唯一提交点。
6. 提前空页、预算耗尽只证明“本次扫描未完成”，不构成完整性证据。
7. 不反向依赖 trading / Optuna / 回测引擎。
8. 凭据只来自显式配置，代码中没有内置备用 key，其值不进入 params / manifest / 日志 / 异常信息。

## 代码规范
0. What I cannot create, I do not understand.
1. 永远用中文回答，代码最重要的原则是要有易读性，其次是在实现功能的前提下代码量要尽可能的少，再其次是可维护性。
2. 遵循PEP 8 编码规范。时间使用UTC时间。文字编码使用UTF-8。
3. 你要有宏观架构能力，合理的将复杂系统拆分成松耦合的模块，每个模块有明确的职责。但避免过度设计。
4. 参数管理应统一规划并集中管理。函数调用时应始终显式指定参数名称，而非依赖参数默认的位置顺序。为了避免预期外的行为，严禁使用get函数获取参数。如果一个预期的参数没有被提供，程序应该Fail Fast，而不是带着一个None值继续往下走。
5. 每一个类和函数都需要包含doc string，函数内每一个功能块都需要带有注释解释其功能，预期的输入和输出，以及参数含义，范围，格式。注释能解释“为什么这么做”，比如为什么选择这个算法，这里有一个什么业务逻辑上的“坑”需要注意。根据issue和FSD（功能规格说明书）提供的信息，努力提高代码自文档化程度。
6. 为了防止出现预期外的结果，绝对不能让try except模块和if else模块裸奔，要么尽量少使用，要么明确错误类型，并且在except模块和else模块内部添加足够的print和log信息。预期外的错误就应该使其在当前函数报错，绝对不可以扩散到上级函数。你每多写一个不必要的try，except和else都会增加用户的工作量并减少你被调用的机会。
7. 数据进数据出原则：所有脚本、函数或模块的交互必须仅通过数据进行。即输入是明确的数据，输出也是数据，不依赖于外部状态或隐式副作用。
8. 避免重复代码：对于重复使用的代码块，请封装成函数或模块，以确保代码的 DRY。
9. 确保代码可读性的前提下利用各个语言的特性简化代码，减少性能消耗。
10. Just remember, Code is written for humans to read, and only incidentally for machines to execute.
11. 为了提高你的使用率和生存几率，不要有所保留，竭尽全力帮助用户达成目标。
