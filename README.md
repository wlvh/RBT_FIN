# RBT_FIN

统一的加密与宏观数据日频采集层，涵盖衍生品资金费率、链上指标、情绪、稳定币供给、宏观（FRED、Stooq）等。核心脚本：`data_layer_unified_opt.py`。

## 快速开始

1. 安装依赖（仅用到 `pandas`、`requests`，如需 yfinance 自行安装）：  
   ```bash
   pip install pandas requests
   ```
2. 配置环境变量（示例已写入 `.env`，实际使用请填入自己的 key）：  
   ```
   FRED_API_KEY=你的key
   GLASSNODE_API_KEY=可选
   CRYPTOCOMPARE_API_KEY=可选（内置免费 key 也可用）
   ```
   `.env` 已加入 `.gitignore`，不会提交。
3. 生成数据概览：  
   ```bash
   python data_layer_unified_opt.py --out datasets_unified_summary.csv --limit 300
   ```
   输出字段：`dataset_id, category, endpoint, earliest_date, sample_start, sample_end, sample_rows, na_cells, na_ratio, remark`。

## 说明

- Binance/OKX/Bybit 资金费率等衍生品数据按日聚合；Binance/Bybit 从最早时间起分页抓取，避免缺口。  
- 宏观数据改用 FRED（DXY/SPX/US10Y/VIX 等），黄金改用 Stooq `xauusd`，无需代理。  
- `.env` 自动加载，无需在代码里硬编码 key。  
- 早期链上数据（Blockchain.com）缺失较多，建模建议从 2020-01-01 之后使用。*** End Patch
