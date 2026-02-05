# MetaGen-AI（中文说明）

- 英文主文档：`README.md`
- 数据默认放置路径：`data/datasets/<name>.jsonl`
- 运行入口脚本在：`scripts/`

常用命令：

```bash
export OPENAI_API_KEY="YOUR_KEY"

python scripts/run_dataset_eval.py --config configs/default.yaml --datasets arith_grid --max_examples 100 --out_csv logs/metrics/run_arith_grid.csv
python scripts/run_baselines.py --config configs/default.yaml --dataset mnli_dev_matched --baseline cot --max_examples 200 --out_csv logs/metrics/baseline_cot_mnli.csv
```

准备数据集可参考 `scripts/prepare_*.py` 的 `--help` 说明。
