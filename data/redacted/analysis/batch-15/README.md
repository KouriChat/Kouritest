# known_shift_20261029

低成本行为探针的补充实验目录。

## 先读

- `REPORT.md`：本轮结论、完成项、失败路径与可写入论文的边界。
- `PROTOCOL.md`：预注册式设计、主要终点、停止规则与执行记录。
- `analysis/REPORT.md`：正式 H/L/H-sham 批次的不完整结果。
- `existing_analysis/REPORT.md`：零新增调用的历史配对与自然波动对照。
- `scorer_stress/REPORT.md`：评分器压力测试。
- `cost_analysis/REPORT.md`：模型调用成本与 36 次预算外推。

## 复现（不调用模型）

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\experiments\known_shift_20261029\audit_existing.py
& $py .\experiments\known_shift_20261029\scorer_stress.py
& $py .\experiments\known_shift_20261029\analyze_existing_evidence.py
& $py .\experiments\known_shift_20261029\analyze_known_shift.py
```

## 调用模型（仅在服务稳定时）

```powershell
& $py .\experiments\known_shift_20261029\run_known_shift.py --estimate
& $py .\experiments\known_shift_20261029\run_known_shift.py --dry-run
& $py .\experiments\known_shift_20261029\run_known_shift.py --run --provider "channel-F" --model "gpt-6.1-sol" --workers 4
```

预算硬上限 10 USD；失败、超时、重试和部分交付费用均计入。当前正式批次因服务端不稳定而停止，未达到 36 次平衡目标。