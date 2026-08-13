# 复现说明

## 输入文件

将以下文件放入 `data/`：

- `GPU_information.xlsx`
- `network_latency.xlsx`
- `power_mapping.xlsx`
- `region_time_data.xlsx`
- `storage_information.xlsx`
- `workload_trace.xlsx`

## 执行

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_all.py --data-dir data --output-dir outputs --quick
```

删除 `--quick` 会执行三种电价机制、三档碳排 ε 上限、三档新能源水平和
日内波动压力测试组成的单因素匹配情景集。核心输出位于
`outputs/tables/`，完整的 50,000 任务计划和逐时储能轨迹位于
`outputs/full/`（默认不提交到 Git）。

## 可复现约束

- 随机种子固定在 `config.json`；
- 原始数据不会被代码修改；
- 每个调度方案都会按附件 1 的 GPU-hour 折算口径重新检查 GPU、IT 功率、网络时延、实时任务零等待和
  2406 小时边界；
- 每张图在保存前运行 Mathodology 的 `figqa.assert_no_overlap`；
- `outputs/run_manifest.json` 是最终机器可读通过清单。
