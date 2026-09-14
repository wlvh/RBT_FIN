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
pip install rbt_fin-0.1.0-py3-none-any.whl
```

基础依赖只有 `pandas` 与 `requests`。个别低频来源的可选依赖（如 `yfinance`）不影响
安装与核心行情能力。

## 当前状态（WP1）

已通过公共流程接入并有离线证据的数据集：

| dataset_id | 说明 | 覆盖判定 |
|---|---|---|
| `btcusdt_perp_1m` | BTCUSDT USDⓈ-M 期货成交价 1m（trading 批次输入的行情来源） | 严格分钟网格 |
| `binance_funding_btcusdt_raw` | Binance 资金费率**原始结算记录**（日均是派生视图） | 事件流，日历未验证 |
| `local_metrics_sample` | 本地 CSV 导入样例（不联网） | 非规则序列，日历未验证 |

其余注册项在 WP2 接入，见 Issue #5 §9。

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
FRED_API_KEY=...
GLASSNODE_API_KEY=...
CRYPTOCOMPARE_API_KEY=...
```

## 兼容入口

`data_layer_unified_opt.py` 保留 `--out/--limit` 的原有用途。
**它会触发刷新**：每次运行都对全部注册项真实发起采集。只读概览请用 `rbt-fin overview`。

## 测试

```bash
python -m pytest tests -q          # 全部离线，不访问外部网络
python -m pytest -m realsource     # 需要真实来源的测试（默认不执行）
```
