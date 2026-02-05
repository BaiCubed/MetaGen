#!/usr/bin/env python
                       
"""
把 HuggingFace 上的 MMLU 数据集（默认为 cais/mmlu）下载并转换成
metagen-ai 统一使用的 jsonl 格式。

输出格式（每行一个样本）：
{
  "id": "abstract_algebra_test_0",
  "subject": "abstract_algebra",
  "question": "...",
  "choices": ["opt A", "opt B", "opt C", "opt D"],
  "answer": "A"   # 标准化成 A/B/C/D
}
"""

import os
import json
import argparse
from typing import List, Dict, Any, Iterable, Optional

from datasets import load_dataset


                             
SUBJECTS_DEFAULT: List[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _norm_answer_to_letter(ans_raw: Any) -> str:
    """
    把各种形式的答案（0/1/2/3 或 1/2/3/4 或 'A'..'D' 或 '0'..'3'）统一成 'A'..'D'
    """
           
    if isinstance(ans_raw, str):
        s = ans_raw.strip()
        u = s.upper()
        if u in {"A", "B", "C", "D"}:
            return u
                 
        try:
            idx = int(s)
        except Exception:
            raise ValueError(f"Cannot interpret answer string: {ans_raw!r}")
    else:
        idx = int(ans_raw)

             
    if 0 <= idx <= 3:
        return "ABCD"[idx]
             
    if 1 <= idx <= 4:
        return "ABCD"[idx - 1]

    raise ValueError(f"Answer index out of range: {ans_raw!r}")


def _extract_choices(row: Dict[str, Any]) -> List[str]:
    """
    适配常见几种字段命名，把选项统一成 list[str] 长度 4。
    尝试顺序：
      - row["choices"] 为 list/tuple
      - row["options"] 为 list/tuple
      - row["A"], row["B"], row["C"], row["D"]
      - row["choice1"]..row["choice4"]
    """
    if "choices" in row and isinstance(row["choices"], (list, tuple)):
        return list(row["choices"])
    if "options" in row and isinstance(row["options"], (list, tuple)):
        return list(row["options"])

                                          
    if all(k in row for k in ["A", "B", "C", "D"]):
        return [row["A"], row["B"], row["C"], row["D"]]
    if all(k in row for k in ["choice1", "choice2", "choice3", "choice4"]):
        return [row["choice1"], row["choice2"], row["choice3"], row["choice4"]]

    raise ValueError(f"Cannot find choices fields in row keys={list(row.keys())}")


def iter_mmlu(
    split: str = "test",
    subjects: Optional[List[str]] = None,
    hf_name: str = "cais/mmlu",
) -> Iterable[Dict[str, Any]]:
    """
    遍历所有 subject + 指定 split，yield metagen-ai 标准样本字典。
    """
    if subjects is None:
        subjects = SUBJECTS_DEFAULT

    for subj in subjects:
        ds = load_dataset(hf_name, subj, split=split)
        for i, row in enumerate(ds):
            q = row.get("question") or row.get("text")
            if not q:
                raise ValueError(f"Missing 'question' field in row: keys={list(row.keys())}")

            choices = _extract_choices(row)
            ans_letter = _norm_answer_to_letter(row.get("answer"))

            yield {
                "id": f"{subj}_{split}_{i}",
                "dataset": "mmlu",
                "subject": subj,
                "question": q,
                "choices": choices,
                "answer": ans_letter,                         
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        help="HF split to use, e.g. 'test' or 'validation' (默认 test)",
    )
    parser.add_argument(
        "--out",
        default="data/mmlu/mmlu_test.jsonl",
        help="输出 jsonl 路径",
    )
    parser.add_argument(
        "--hf_name",
        default="cais/mmlu",
        help="HuggingFace 数据集名称（默认 cais/mmlu；如需用 hendrycks_test 可改）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in iter_mmlu(split=args.split, hf_name=args.hf_name):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

    print(f"[prepare_mmlu] wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
