                         
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset


def extract_answers_from_example(ex: Dict[str, Any]) -> List[str]:
    """
    尽可能从 HF DROP 的一个样本里抽取所有 gold 答案：
    - 首选: answer["spans"] 或 answers_spans["spans"]
    - 退路: answer["number"]
    - 再退路: answer["date"] (year / month / day)
    返回: 去重后的字符串列表
    """
    answers: List[str] = []

                               
    ans = ex.get("answer") or {}

                                            
    spans = ans.get("spans")
    if not spans:
                                 
        spans_container = ex.get("answers_spans") or {}
        spans = spans_container.get("spans")

    if spans:
                                                        
        if isinstance(spans, list):
            for s in spans:
                if isinstance(s, str):
                    s_ = s.strip()
                    if s_:
                        answers.append(s_)
                elif isinstance(s, (list, tuple)):
                    for ss in s:
                        if isinstance(ss, str):
                            ss_ = ss.strip()
                            if ss_:
                                answers.append(ss_)

                           
    num = ans.get("number")
    if isinstance(num, str):
        num_ = num.strip()
        if num_:
            answers.append(num_)

                              
    date = ans.get("date") or {}
    y = date.get("year")
    m = date.get("month")
    d = date.get("day")
                  
    if y not in (None, "", 0):
        parts = []
                                  
        if m not in (None, "", 0):
            parts.append(str(m))
        if d not in (None, "", 0):
            parts.append(str(d))
        parts.append(str(y))
        answers.append(" ".join(parts))

                      
        answers.append(str(y))

             
    seen = set()
    uniq_answers: List[str] = []
    for a in answers:
        if a not in seen:
            seen.add(a)
            uniq_answers.append(a)

    return uniq_answers


def convert_split(
    hf_split,
    out_path: str,
    split_name: str,
) -> int:
    """
    把一个 HF split 转成 JSONL 文件。
    返回写出的样本数。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(hf_split):
            passage = ex.get("passage") or ex.get("context") or ""
            question = ex.get("question") or ""
            if not passage or not question:
                           
                continue

            answers = extract_answers_from_example(ex)
                      
            if not answers:
                                          
                continue

            ex_id = (
                ex.get("id")
                or ex.get("_id")
                or f"{split_name}_{i:06d}"
            )

            item = {
                "id": str(ex_id),
                "passage": str(passage),
                "question": str(question),
                "answers": answers,
                                                   
                "answer": answers[0],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[DROP] wrote {n} examples -> {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default="data/datasets",
        help="输出 JSONL 所在目录，默认 data/datasets",
    )
    ap.add_argument(
        "--include_train",
        action="store_true",
        help="是否同时导出 train split（默认只导出 dev/validation）",
    )
    args = ap.parse_args()

    out_dir = args.out_dir

                                 
         
                                          
                                                    
    print("[DROP] loading dataset from HuggingFace ...")
    try:
        ds = load_dataset("drop", "drop")
    except Exception:
                                
        ds = load_dataset("drop")

                                            
                           
    if "validation" in ds:
        dev_split = ds["validation"]
    elif "dev" in ds:
        dev_split = ds["dev"]
    else:
        raise RuntimeError("Cannot find dev/validation split in DROP dataset.")

    dev_out = os.path.join(out_dir, "drop_dev.jsonl")
    convert_split(dev_split, dev_out, "dev")

                 
    if args.include_train:
        if "train" not in ds:
            print("[DROP] no train split found; skip.")
        else:
            train_split = ds["train"]
            train_out = os.path.join(out_dir, "drop_train.jsonl")
            convert_split(train_split, train_out, "train")

    print("[DROP] done.")


if __name__ == "__main__":
    main()
