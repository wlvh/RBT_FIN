# RBT_FIN

统一的加密与宏观数据采集层。核心是一条公共流程：

```text
数据集定义 + 本次请求
    → 检查缓存资格
    → 按需采集（增量 / 全量 / 本地导入）
    → 规范化、去重或修订处理、校验
    → 写好内容命名的不可变 CSV
    → 原子替换 manifest，发布新状态
    → 返回数据及使用说明
```

规格与验收以 [wlvh/RBT_FIN#5](https://github.com/wlvh/RBT_FIN/issues/5) 为准；
消费者接入与部署边界见 [wlvh/trading#105](https://github.com/wlvh/trading/issues/105)。

## 安装

```bash
pip install rbt_fin-0.3.4-py3-none-any.whl
```

基础依赖只有 `pandas>=2.0`、`numpy` 与 `requests`；包内没有任何来源需要可选依赖（原 `yfinance` 声明已删除，该来源按 #5 §1 不接入）。

## 当前状态

全部 **25 个基线注册项**已接入公共流程，另有 3 个新增数据集（分钟行情、现货镜像、本地样例）。
对外注册标识共 30 个 = 26 个基础数据集 + 4 个别名。

| 采集函数 | 注册项 | 说明 |
|---|---:|---|
| `binance_klines`（分钟行情） | 1 | BTCUSDT USDⓈ-M 期货成交价 1m；trading 批次输入来源 |
| `binance_reference_price` | 3 | 指数价 / 标记价 / 溢价指数；占位 `volume` 不外露，溢价可为负 |
| `binance_funding` | 1 | 原始结算记录；日均是派生视图 |
| `bybit_funding` | 1 | `retCode != 0` 立即失败；首页不传 `startTime` |
| `daily_series`（通用） | 17 | Blockchain.com ×6 / Fear&Greed / Wikipedia / DefiLlama / FRED ×5 / Stooq / Glassnode / CryptoCompare |
| `local_csv` | 2 | 本地导入，不联网 |
| FRED 别名 | 4 | 同序列的第二个注册名解析到同一数据集，**不重复下载** |

逐项状态（已接入 / 离线结果 / 真实验证 / 阻塞原因）用以下命令生成：

```bash
python tools/registry_status.py --work-dir /tmp/status --out status.json
```

它把「缺 key」「来源不可达」「反爬拦截」与「实现失败」严格分开——
这三类阻塞不是验收通过，但也不是代码缺陷。

## 使用

```python
import pandas as pd
from rbt_fin import DataRequest, acquire_dataset, build_session, HttpClient, RequestBudget
from rbt_fin.registry import build_klines_dataset
from rbt_fin.timeutils import complete_end_exclusive

now = pd.Timestamp.now(tz="UTC")                      # 一次任务只取一次时钟
end_exclusive = complete_end_exclusive(now_utc=now, interval="1m")  # 排除未收盘 bar

result = acquire_dataset(
    dataset=build_klines_dataset(),
    request=DataRequest(
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end_exclusive=end_exclusive,
        require_full_coverage=True,                   # 严格任务：有缺口就拒绝
    ),
    store_root=Path("/data/cache"),                   # 路径显式传入，不扫描 cwd
    now_utc=now,
    http=HttpClient(session=build_session(), budget=RequestBudget()),
)
print(result.usage_notes())                           # 覆盖状态 / 哈希 / 降级说明
```

## 命令行

```bash
rbt-fin update  --cache-root <dir> --dataset btcusdt_perp_1m \
                --start 2026-01-01T00:00:00Z --end-exclusive 2026-01-02T00:00:00Z
rbt-fin overview --cache-root <dir>          # 只读，不触发任何采集
rbt-fin verify   --cache-root <dir>          # 重算内容哈希并核对 manifest
rbt-fin export-trading-csv --cache-root <dir> --dataset btcusdt_perp_1m \
                --target /batch/BTCUSDT_future_1m.csv \
                --start 2026-01-01T00:00:00Z --end-exclusive 2026-01-02T00:00:00Z
```

时间参数必须带时区；naive 值会被拒绝。

## 主动限速（0.3.3 起）

`HttpClient` 在每次发送前经进程共享的 `RateLimitRegistry` 排程（`rbt_fin.ratelimit`）：

- 权重表来自官方文档并已用真实响应头核对（期货 K 线族按 `limit` 分档：`[1,100)`→1、`[100,500)`→2、`[500,1000]`→5、`>1000`→10；`fundingRate` 走独立的 500/5min 桶；`exchangeInfo` 1）。Binance host 上未登记权重的端点直接 `ConfigError`，不按猜测发送；非 Binance 来源不做权重限速。
- 本进程每个窗口只用限额的 `RateLimitConfig.max_share`（默认 0.5，与旧实现每页 `sleep 0.3s` 的节奏量级相当），额度耗尽等到下一分钟窗口；成功请求与重试同样计费；`X-MBX-USED-WEIGHT-1M` 只向上校正本地账本。
- `discover_binance_futures_limit(http)` 读取真实 `exchangeInfo.rateLimits` 刷新限额（1 权重）；文档示例值不是永久配置。
- HTTP 429：遵守 `Retry-After`（超过 `cooldown_max_seconds=180` 即失败），缺失时保守等到下一窗口；HTTP 418：`SourceBannedError` 立即失败，不重试、不换出口。
- `RequestBudget.max_total_wait_seconds`（默认 3600）限制排程 + 冷却 + 退避的总等待：剩余预算进入排程决策，需要的等待超过剩余预算时**立即** `BudgetExhaustedError`（不安排超预算的等待、不发送、不计请求数），耗尽状态对后续调用同样生效；不返回部分数据。
- 墙钟只用于定位交易所窗口；窗口是否真的过去与等待时长由单调时钟判定，墙钟前拨/回拨不刷新额度（只记 `clock_jumps`）。
- 默认 K 线单页 `limit=1000`：每权重取得 200 根（1500 档 150 根、499 档 249.5 根）；`limit` 不参与缓存身份。
- `HttpClient.stats` 把请求数、三类等待秒数、限流响应数与各桶账本分开报告，并写入 manifest `extra.http_stats`。
- 不能保证：限额按 IP 统计，同一出口其他进程/程序的用量本库看不到；本库不是跨进程/跨机分布式限流器。

## 存储布局

```text
<cache-root>/<dataset_id>/
  data-<sha256>.csv     内容文件，写入后永不原地修改
  manifest.json         唯一提交点，原子替换
  update.lock           单数据集单写者的固定旁路锁
```

## 凭据

凭据只从显式环境变量/配置读取，**代码中没有任何内置备用 key**。
缺少配置时相关来源会直接失败，不会退化成“零缺失”的正常输出。

```
FRED_API_KEY=...          # FRED 五个序列（七个注册名）
GLASSNODE_API_KEY=...     # glassnode_active_addresses_btc
CRYPTOCOMPARE_API_KEY=... # cryptocompare_btc_chain_pct
```

凭据只在真正发请求时合入，**永不进入** `dataset.params`、`manifest.json` 或日志。
缺配置时对应来源直接 Fail Fast，不会表现成「零缺失」的正常输出。

网络连接失败只保留脱敏后的错误类型和文字；重试耗尽或退避预算拒绝时，不挂接含请求
URL 的原始异常，避免完整堆栈和 `logging.exception` 再次带出查询串中的 key。

### 提交前检查

每个克隆需要安装一次本地提交钩子；仅提交配置文件不会自动启用检查：

```bash
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
pre-commit run gitleaks              # 手动检查本次暂存的变更
```

`.pre-commit-config.yaml` 固定 Gitleaks 版本，钩子只扫描暂存区，疑似密钥的输出自动脱敏。
`.gitleaks.toml` 继承工具内置规则，并补查 `api_key = os.getenv(...) or "固定 key"`
这类历史上出现过的备用凭据。测试文件也接受扫描，不整体豁免。
首次安装会下载工具及所需的 Go 环境；它们不属于运行数据采集器所需的依赖。

`.env` 及 `.env.*` 默认不提交；只有无真实凭据的 `.env.example` 可以作为模板。
忽略规则不会移除已经被 Git 跟踪的文件，也不会让历史中已经暴露的凭据失效。
维护者应在 GitHub 的 [Code security 设置](https://github.com/wlvh/RBT_FIN/settings/security_analysis)
保持密钥扫描和推送保护开启；本地钩子与 GitHub 检查共同防止再次提交凭据。

## 兼容入口

`data_layer_unified_opt.py` 保留 `--out/--limit` 的原有用途与原版 10 列概览输出，但已收敛为**薄包装**：
每个注册项都走 `rbt_fin.acquire_dataset` 公共流程（缓存资格 → 采集 → 校验 → 内容寻址发布），
不再自带任何采集器/解析器。刷新语义由 `--mode auto|force|offline` 显式决定（默认 `auto`：缓存合格且未到
重新检查时机则零请求）；`--start-date` 必填，`--end-date` 缺省为最新已完成 UTC 日；凭据只从进程环境与
显式 `--env-file` 收集，不写回环境。失败行的 `na_ratio` 为空（缺失率未知）。只读概览请用 `rbt-fin overview`。

## 测试

```bash
python -m pytest tests -q          # 全部离线，不访问外部网络
python -m pytest -m realsource     # 需要真实来源的测试（默认不执行）
```
