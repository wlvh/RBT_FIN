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
pip install rbt_fin-0.2.0-py3-none-any.whl
```

基础依赖只有 `pandas` 与 `requests`。个别低频来源的可选依赖（如 `yfinance`）不影响
安装与核心行情能力。

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

## 兼容入口

`data_layer_unified_opt.py` 保留 `--out/--limit` 的原有用途。
**它会触发刷新**：每次运行都对全部注册项真实发起采集。只读概览请用 `rbt-fin overview`。

## 测试

```bash
python -m pytest tests -q          # 全部离线，不访问外部网络
python -m pytest -m realsource     # 需要真实来源的测试（默认不执行）
```
