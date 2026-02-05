#!/usr/bin/env python
                       

"""
把 HuggingFace 上的 GLUE MNLI 数据集下载并转换成
metagen-ai 统一使用的 jsonl 格式（多选题 schema，复用 MMLU 流水线）。

输出格式（每行一个样本）大致为：
{
  "id": "mnli_validation_matched_0",
  "dataset": "mnli",
  "split": "validation_matched",
  "premise": "...",
  "hypothesis": "...",
  "question": "Premise: ...\\nHypothesis: ...\\nWhat is the relationship ...?",
  "choices": ["entailment", "neutral", "contradiction"],
  "answer": "A"   # or "B"/"C"
}
"""

import os
import json
import argparse
from typing import Dict, Any, Iterable

from datasets import load_dataset

                                   
LABEL_ID_TO_TEXT = {
    0: ("entailment", "A"),
    1: ("neutral", "B"),
    2: ("contradiction", "C"),
}

CHOICES = ["entailment", "neutral", "contradiction"]


def iter_mnli(
    split: str = "validation_matched",
    hf_name: str = "glue",
    subset: str = "mnli",
) -> Iterable[Dict[str, Any]]:
    """
    迭代 MNLI 样本，并转成 metagen-ai 统一 schema：
      - question: 把 premise + hypothesis 拼成一个问题描述
      - choices: 3 个字符串（entailment / neutral / contradiction）
      - answer: 'A'/'B'/'C'，对应三个真实标签
    """
    ds = load_dataset(hf_name, subset, split=split)

    for i, ex in enumerate(ds):
        label_id = int(ex["label"])
        if label_id not in LABEL_ID_TO_TEXT:
            continue

        label_text, letter = LABEL_ID_TO_TEXT[label_id]

        premise = ex["premise"]
        hypothesis = ex["hypothesis"]

        q = (
            "You are doing a natural language inference (NLI) task.\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Decide the relationship between the hypothesis and the premise.\n"
            "Options:\n"
            "A. entailment\n"
            "B. neutral\n"
            "C. contradiction\n\n"
            "Answer with one line: 'Final answer: <LETTER>' where <LETTER> is A, B, or C."
        )

        out: Dict[str, Any] = {
            "id": f"{subset}_{split}_{i}",
            "dataset": "mnli",
            "split": split,
            "premise": premise,
            "hypothesis": hypothesis,
            "question": q,
            "choices": CHOICES,
            "answer": letter,          
        }
        yield out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="validation_matched",
        help="MNLI split，通常用 validation_matched / validation_mismatched / train 等",
    )
    parser.add_argument(
        "--subset",
        default="mnli",
        help="GLUE 子任务名（一般不用改）",
    )
    parser.add_argument(
        "--hf_name",
        default="glue",
        help="HuggingFace 数据集名称（默认 glue）",
    )
    parser.add_argument(
        "--out",
        default="data/datasets/mnli_dev_matched.jsonl",
        help="输出 jsonl 路径",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in iter_mnli(split=args.split, hf_name=args.hf_name, subset=args.subset):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

    print(f"[prepare_mnli] wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
