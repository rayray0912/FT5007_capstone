# Bitfinex 永续合约 MBO 数据集说明

> 数值截至 2026-09-13 11:45 UTC，采集仍在持续。文末附刷新命令。

---

## 0. 先纠正一处理解偏差

**当前采集的不是 HYPE/USDT，是 BTC / ETH / XAUT 三个 USDT 保证金永续合约。**

HYPE-PERP 在选标的阶段被排除，原因是流动性不足：Bitfinex 上它 24h 成交量仅约 **13,675 美元**（UNI-PERP 约 10,474 美元），全天成交仅几笔到几十笔。AS 做市模型需要从成交流中标定订单到达强度 λ(δ)，这个样本量无法支撑参数估计，也无法做有意义的成交回测。

同期对比（Bitfinex 永续，24h 成交量 USD）：

| 合约 | 成交量 | 合约 | 成交量 |
|---|---:|---|---:|
| **BTC-PERP** | **12,972,509** | LTC-PERP | 152,834 |
| **XAUT-PERP** | **1,847,590** | ADA-PERP | 101,013 |
| **ETH-PERP** | **1,830,890** | LINK-PERP | 70,940 |
| ETH/BTC-PERP | 508,342 | HYPE-PERP | 13,675 |
| DOT-PERP | 315,642 | UNI-PERP | 10,474 |

最终选定 BTC / ETH / XAUT，即该场所仅有的三个具备真实成交流的标的。XAUT（代币化黄金永续）作为非加密资产的对照组保留，其做市商报价结构与 BTC/ETH 差异显著，适合做跨品种的微观结构对比。

---

## 1. 数据源与采集方式

| 项目 | 内容 |
|---|---|
| 交易所 | Bitfinex |
| 接口 | WebSocket v2 Raw Books，`prec=R0`，`len=250` |
| 鉴权 | 无（公开接口） |
| 粒度 | **Market-by-order（逐订单）**，非价位聚合 |
| 标的 | `tBTCF0:USTF0`、`tETHF0:USTF0`、`tXAUTF0:USTF0` |
| 连接结构 | 每标的一条独立 WS 连接，故障隔离 |
| 起始时间 | 2026-08-30 06:29:47 UTC |
| 运行方式 | 7×24 持续采集，Docker 容器，自动重连 |

选择 Bitfinex 的理由：在主流中心化交易所中，它是**唯一同时满足「订单级粒度 + 永续合约 + 公开无鉴权」**的行情源。代价是流动性低于 Binance / Bybit，这一点在外部效度部分讨论。

`len=250` 指每侧跟踪 250 个**订单**（非 250 个价位），是该接口的最大档位。

---

## 2. 数据内容与字段

### 2.1 `book_mbo` — 订单级盘口事件

每一行是一次盘口变动。按 `(symbol, epoch, seq)` 顺序回放可精确重建任意时刻的订单簿状态。

| 字段 | 类型 | 说明 |
|---|---|---|
| `symbol` | LowCardinality(String) | 合约代码 |
| `ts_exch` | DateTime64(3) | 交易所时间戳（毫秒） |
| `ts_local` | DateTime64(6) | 本地接收时间戳（微秒） |
| `epoch` | UInt32 | 连续段编号，每次重连递增 |
| `seq` | UInt64 | 交易所全局序号，用于丢包检测 |
| `order_id` | UInt64 | 订单 ID |
| `side` | Enum8 | bid / ask |
| `action` | Enum8 | snapshot / add / update / delete |
| `price` | Float64 | 价格 |
| `amount` | Float64 | 数量（买正卖负） |

**删除事件保留删除前的价格与数量**。协议原始消息在撤单时只发 `price=0`，采集器回查该订单的最后已知状态再落库，因此可直接计算队列消耗（queue depletion）而无需自行维护状态机。

### 2.2 `trades` — 逐笔成交

`symbol`、`ts_exch`、`ts_local`、`trade_id`、`price`、`amount`、`side`。

采用 `te`（低延迟）消息而非 `tu`，ReplacingMergeTree 按 `(symbol, trade_id)` 去重。

### 2.3 `collector_events` — 采集完整性日志

记录 connect / subscribed / snapshot / disconnect / seq_gap / checksum_fail / resync / error。**这是判断数据可用区间的权威依据**，使用数据前应先审计此表。

---

## 3. 数据规模

采集起点 2026-08-30 06:29:47 UTC，截至 2026-09-13 11:45 UTC，连续运行 **14.2 天**。

| 表 | 行数 | 压缩后 | 未压缩 | 压缩比 | 字节/行 |
|---|---:|---:|---:|---:|---:|
| `book_mbo` | 114,992,905 | 1.71 GiB | 5.87 GiB | 3.43 | 15.96 |
| `trades` | 531,045 | 7.68 MiB | 21.27 MiB | 2.77 | 15.17 |
| `collector_events` | 810 | 9.06 KiB | 30.67 KiB | 3.39 | — |

分标的盘口事件：

| 标的 | 行数 | 占比 |
|---|---:|---:|
| `tBTCF0:USTF0` | 52,714,817 | 45.8% |
| `tETHF0:USTF0` | 34,677,447 | 30.2% |
| `tXAUTF0:USTF0` | 27,600,760 | 24.0% |

**增长速率约 810 万行/天、123 MB/天**，年化约 45 GB。在 120 GB 的专用卷上，6 个月 TTL 的稳态占用约 22 GB，容量不构成约束。

**成交约 37,400 笔/天**（三标的合计），累计 53.1 万笔。

---

## 4. 数据质量

### 4.1 三重校验机制

采集器通过 Bitfinex `conf` flag `229376` 同时启用三项独立校验：

| 机制 | flag | 检测目标 |
|---|---|---|
| `SEQ_ALL` 序号 | 65536 | 每条消息带单调计数器，跳号即为静默丢包 |
| `OB_CHECKSUM` 校验和 | 131072 | 每次盘口迭代下发前 25 档 CRC32；R0 模式基于 ORDER_ID 与 AMOUNT 计算 |
| 读超时 | — | 心跳约 15 秒，20 秒静默判定连接失效 |

任一机制触发即：`epoch += 1` → 清空本地簿 → 重新拉取快照。

### 4.2 实测结果（14.2 天）

| 指标 | 结果 |
|---|---|
| Checksum 校验次数 | 截至 09-07 日志记录 761,766 次；按同速率推算累计约 134 万次 |
| **Checksum 失败** | **0** |
| **序号跳变（丢包）** | **0** |
| 连接中断 | 129 次 |
| 对应重同步 | 129 次（100% 正确处理） |
| 重新拉取快照 | 138 次 |
| 数据污染 | 无 |

129 次中断全部为 TCP 连接层事件（服务端定期重启 + 家宽抖动），折合每标的每天约 3 次，与第一周的速率一致，无劣化趋势。

**零 seq_gap 与零 checksum_fail 意味着不存在「连接保持但数据错误」这一类隐性污染**——所有中断点都是显式的、可定位的。这是 MBO 数据最关键的质量指标：本地簿一旦漂移而未被发现，后续所有回放都是错的，且难以事后察觉。

### 4.3 连续可用区间

| 标的 | 连续段数 | 平均时长 | 最长段 | >2 小时的段 |
|---|---:|---:|---:|---:|
| BTC | 45 | 7.6 h | **40 h** | 33 |
| ETH | 49 | 6.9 h | **54 h** | 29 |
| XAUT | 41 | 8.3 h | **54 h** | 29 |

每标的有 29–33 段超过 2 小时，最长连续段 40–54 小时，且经逐小时行数验证确为真实连续（无空洞小时）。

**分段核算自洽**：以 BTC 为例，45 段 × 454.2 分钟 = 20,439 分钟，实际运行 20,475 分钟，差值 36 分钟正是断点空洞本身；ETH 与 XAUT 同样吻合。这验证了 epoch 计数器在进程重启后从数据库续接（而非归零）的修复是有效的，`v_epochs` 报告的段边界可直接采信。

**使用约定**：跨 epoch 回放会产生"幽灵订单"（删除消息落在数据空洞中的订单永远无法移除），因此所有回放与因子计算必须限定在单个 epoch 内。

### 4.4 采集延迟

新加坡 → Bitfinex 撮合引擎：

| 标的 | p50 | p99 |
|---|---:|---:|
| BTC | 79 ms | 231 ms |
| ETH | 80 ms | 81 ms |
| XAUT | 79 ms | 90 ms |

三标的 p50 高度一致，表明瓶颈为网络物理传播而非本地处理能力。BTC 的 p99 偏高源于行情爆发时的突发批量。

**此延迟仅影响时间戳精度的解释，不影响数据完整性**——`ts_exch` 为交易所时间戳，研究中应以此为准；`ts_local` 保留用于诊断。

---

## 5. 已识别的数据特征

### 5.1 日内活跃度周期

| UTC 时段 | 行数/小时 | 说明 |
|---|---:|---|
| 09:00–10:00 | 46k–58k | 全天最低 |
| 14:00 | 165k | 美股开盘 |
| 23:00–02:00 | 160k–171k | 全天最高 |

**峰谷差 3.7 倍。** Bitfinex 活跃度跟随美国时段，与其用户结构一致。

**方法论影响**：因子研究与参数标定必须按时段分层，或将时段作为控制变量。不分层的话，报价密度、价差、深度的日内周期性会污染 IC 估计。这是薄流动性市场特别容易踩的坑。

### 5.2 XAUT 的特殊性

XAUT 的盘口更新频率高于 BTC（单位时间事件数更多），但成交显著更少。说明其做市商采取高频报价更新、低成交的模式，与 BTC/ETH 的行为结构不同。作为跨资产类别对照组价值较高。

---

## 6. 存储与访问

| 项目 | 内容 |
|---|---|
| 数据库 | ClickHouse 24.8 |
| 宿主 | NUC13 Pro（12 核 / 32 GB），Ubuntu Server 24.04，新加坡 |
| 存储 | 独立 LVM 逻辑卷 120 GB，挂载于 `/data/clickhouse` |
| 表引擎 | MergeTree（book_mbo）/ ReplacingMergeTree（trades） |
| 分区 | 按月 `toYYYYMM(ts_exch)` |
| 排序键 | `(symbol, ts_exch, seq)` |
| 列编码 | DoubleDelta + ZSTD(3)（时间戳/ID）、Gorilla + ZSTD(3)（价格/数量） |
| TTL | 6 个月自动过期 |
| 网络访问 | Tailscale 内网，`100.76.49.84:8123`（HTTP）、`:9000`（native） |

访问示例：

```python
import clickhouse_connect
ch = clickhouse_connect.get_client(host="100.76.49.84", port=8123, database="bfx")
df = ch.query_df("SELECT * FROM bfx.v_epochs WHERE minutes > 120")
```

内置审计视图：`v_storage`（存储占用）、`v_daily`（日行数）、`v_epochs`（连续段清单）。

---

## 7. 是否足以支撑 AS 回测

### 7.1 满足的部分

AS 模型的核心输入，本数据集均可直接导出：

| AS 所需 | 本数据集 | 状态 |
|---|---|---|
| 中间价序列与波动率 σ | 由 MBO 精确重建 | ✅ |
| 订单到达强度 λ(δ) = A·exp(−κδ) | 成交时刻回溯当时盘口，直接测距 | ✅ |
| 库存动态 | 可模拟 | ✅ |
| **成交模拟的队列位置** | **MBO 提供订单级队列序，这是 L2 聚合数据做不到的** | ✅ |

**队列位置是本数据集相对于常规 L2 深度数据的核心优势。** 聚合数据只能知道某价位有多少量，无法知道自己的模拟订单排在第几位，成交判定只能用「价位被击穿」这种粗糙规则。MBO 可以精确追踪队列前方的撤单与成交，做出显著更真实的 fill 判定——这直接关系到做市策略回测的可信度。

样本量方面：三标的合计 53.1 万笔成交（约 37,400 笔/天）。即便按最保守的分层方案——10 个 δ 分桶 × 6 个时段 × 3 个标的共 180 格——平均每格仍有约 2,900 个样本，足以稳定估计 κ 并保留分层后的统计功效。盘口侧 1.15 亿条订单级事件对撤单率、报价寿命、队列动态等指标更是充裕。

（各标的成交笔数的精确拆分见文末刷新命令；XAUT 的报价更新占比高于其成交占比，实际分布与盘口事件占比不完全一致。）

### 7.2 需要补充的部分

**缺少永续合约特有的三项数据：资金费率、标记价格、指数价格。**

这是当前数据集最明确的缺口。永续做市的库存持有成本中，资金费率是重要组成部分——多头库存在正资金费率时持续付费，空头则持续收费。缺少这一项会导致：

- 库存惩罚项 γ 的标定失去一个真实的经济锚点
- 回测的 PnL 归因不完整，库存成本被系统性低估

这与此前 Report F 中主动撤回初步正收益结论的原因之一（sweep summary 未净额计算库存持有成本）直接相关。补上资金费率数据可以让该项修正从「已识别的方法论缺陷」变成「已量化的成本项」。

Bitfinex 的 `status` 频道（key 为 `deriv:tBTCF0:USTF0`）提供标记价格、指数价格、当期资金费率、下期资金费率时点、未平仓合约量，同样是公开无鉴权接口。**建议尽快补采**——历史资金费率虽可通过 REST 回补，但标记价格的高频序列无法回补，越早开始损失越小。

### 7.3 外部效度的限制

需在研究结论中明确说明：**本数据集反映的是 Bitfinex 这一中等流动性场所的微观结构**。BTC-PERP 约 1,300 万美元日成交量，相较 Binance 同类合约低 2–3 个数量级。

由此标定出的 κ（订单到达的价格敏感度）、最优价差、库存周转速度，**不能直接外推到主流场所**。在薄流动性市场中做市商的报价行为、撤单率、队列竞争强度都与厚簿市场存在系统性差异。

这不是数据缺陷，而是需要在方法论部分主动声明的适用范围。反过来说，薄簿环境下单个做市商的行为更容易识别，对于研究做市商报价策略本身反而是更干净的样本。

### 7.4 结论

**核心结论：当前数据集足以支撑完整的 AS 回测，且在成交模拟的真实性上优于常规 L2 数据集。**

两项待办：

1. **补采 `status:deriv` 频道**（资金费率 / 标记价格 / 指数价格）——影响库存成本的完整性，建议立即执行
2. **回测设计中按时段分层**——应对 3.7 倍的日内活跃度差异

---

## 附：数值刷新命令

发出此文档前，可用以下命令更新第 3、4 节的数字：

```bash
cd ~/bfx-mbo-collector

# 第 3 节：规模
docker compose exec clickhouse clickhouse-client -q \
  "SELECT * FROM bfx.v_storage FORMAT Pretty"

docker compose exec clickhouse clickhouse-client -q \
  "SELECT symbol, count() AS rows, min(ts_exch) AS since, max(ts_exch) AS latest
   FROM bfx.book_mbo GROUP BY symbol ORDER BY symbol FORMAT Pretty"

# 第 7.1 节：分标的成交笔数
docker compose exec clickhouse clickhouse-client -q \
  "SELECT symbol, count() AS trades, round(count() / 14.2, 0) AS per_day
   FROM bfx.trades GROUP BY symbol ORDER BY symbol FORMAT Pretty"

# 第 4.2 节：完整性
docker compose exec clickhouse clickhouse-client -q \
  "SELECT kind, count() FROM bfx.collector_events
   GROUP BY kind ORDER BY count() DESC FORMAT Pretty"

# 第 4.3 节：连续段
docker compose exec clickhouse clickhouse-client -q \
  "SELECT symbol, count() AS segments, round(avg(minutes),1) AS avg_min,
          max(minutes) AS longest_min, countIf(minutes > 120) AS segs_over_2h
   FROM bfx.v_epochs GROUP BY symbol FORMAT Pretty"
```
