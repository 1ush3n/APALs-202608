# r3：历史无效数据归档（不可运行）

`r3` 的训练图缺少正式 `专业编码` 与 `工种` 语义字段，历史加载器曾将它们静默降级为工种 0。
因此 r3 是 `historical_invalid_schema` 归档：不得用于训练、正式评估或统计；只能保留作问题溯源证据。

## Layout and naming

- `t/`: 30 fixed APAL training graphs.
- `b/t/`: initial baseline schedules for the 30 training graphs.
- `b/r/`: initial baseline schedules for `real_283`, `real_680`, `real_2338`, and `real_3182`.
- `s/`: fixed low/medium/high perturbation scenario CSVs and metadata. Each real instance has 60 scenario IDs, 20 per level.
- `m.json`：历史 manifest，仅用于追溯；任何生产入口都必须拒绝它。
- `integrity_check.json`: SHA-256 copy-verification record for the 74 copied assets.

## Reproducibility constraints

- Scenario and training-graph seed: `20260701`.
- Warm-start checkpoint: `checkpoints/init/g15.ckpt`; bytes: `127124342`; SHA-256: `9e8f9136ac99eaaff7efe1e8bbb14612e6327b03ff69c36f9fee1b6d1b6a3225`.
- Real-instance worker mapping: `283 -> 80`, `680 -> 100`, `2338 -> 140`, `3182 -> 160`. All runtime entry points must apply this mapping before environment reset.

## Required invocation paths

禁止使用 `data/r3/t` 或 `data/r3/m.json` 启动任何训练或正式评估。当前正式重调度资产为 `data/r4/m.json`，协议为 `explicit_fiveskill_v1`。
