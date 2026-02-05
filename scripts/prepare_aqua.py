#!/usr/bin/env python3
"""Prepare AQuA (AQuA-RAT) dataset into metagen-ai jsonl format.

This repo's dataset loader reads local jsonl files:
  data/datasets/aqua_train.jsonl
  data/datasets/aqua_dev.jsonl
  data/datasets/aqua_test.jsonl

Each line should be a JSON object with at least:
  - id
  - question
  - options (list)
  - correct (A..E)
  - rationale (optional)

The script downloads AQuA-RAT via Hugging Face `datasets` and converts to this format.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


HF_CANDIDATES: List[str] = [
                                                     
    "deepmind/aqua_rat",
    "aqua_rat",
    "Chinar/AQuA-RAT",
    "MU-NLPC/Calc-aqua_rat",                                            
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        s = s.upper()
        if s in {"A", "B", "C", "D", "E"}:
            return s
                                            
        if len(s) >= 1 and s[0] in "ABCDE":
            return s[0]
        return None
    if isinstance(x, int):
        if 0 <= x <= 4:
            return chr(65 + x)
    return None


def _normalize_options(opts: Any) -> List[str]:
    """Return options as a list of strings, preferably 'A) ...'.."""
    if opts is None:
        return []
    if isinstance(opts, dict):
                                                              
        out: List[str] = []
        for k in "ABCDE":
            if k in opts:
                out.append(f"{k}) {opts[k]}")
            elif f"{k})" in opts:
                out.append(f"{k}) {opts[f'{k})']}")
            elif f"{k}." in opts:
                out.append(f"{k}) {opts[f'{k}.']}")
                                                                            
        if not out:
            for k in sorted(opts.keys(), key=lambda z: str(z)):
                out.append(str(opts[k]))
        return out

    if isinstance(opts, (list, tuple)):
        return [str(x) for x in opts]

                                                      
    return [str(opts)]


def _pick_split(ds_dict, split: str):
                                     
    want = split.lower()
    aliases = {
        "dev": ["validation", "valid", "dev"],
        "val": ["validation", "valid", "dev"],
        "validation": ["validation", "valid", "dev"],
        "test": ["test"],
        "train": ["train"],
    }
    for key in aliases.get(want, [want]):
        if key in ds_dict:
            return ds_dict[key], key
    raise KeyError(f"Split '{split}' not found. Available: {list(ds_dict.keys())}")


def _load_from_hf() -> Tuple[Any, str]:
    last_err: Optional[Exception] = None
    for name in HF_CANDIDATES:
        try:
            ds = load_dataset(name)
            return ds, name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load AQuA-RAT from Hugging Face candidates: {HF_CANDIDATES}. Last error: {last_err}")


def convert_and_write(split: str, out_path: str, max_examples: Optional[int] = None) -> None:
    ds, src_name = _load_from_hf()
    split_ds, resolved = _pick_split(ds, split)

    _ensure_dir(os.path.dirname(out_path))

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(split_ds):
            if max_examples is not None and i >= max_examples:
                break

            q = ex.get("question") or ex.get("prompt") or ex.get("query") or ""
            opts = ex.get("options") or ex.get("choices") or ex.get("answer_choices") or ex.get("candidates")
            opts_list = _normalize_options(opts)

                                                                                                                  
            corr = (
                ex.get("correct")
                or ex.get("answer")
                or ex.get("label")
                or ex.get("gold")
                or ex.get("target")
            )
            corr_letter = _to_letter(corr)

            rid = ex.get("id") or ex.get("qid") or ex.get("uid") or f"aqua_{resolved}_{i}"

            item: Dict[str, Any] = {
                "id": str(rid),
                "question": str(q),
                "options": opts_list,
                "correct": corr_letter if corr_letter is not None else str(corr),
                "rationale": ex.get("rationale") or ex.get("explanation") or "",
                "source": src_name,
                "split": resolved,
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[prepare_aqua] source={src_name} split={resolved} wrote {n} examples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "validation", "test", "val"], help="Which split to export") 
    ap.add_argument("--out", default=None, help="Output jsonl path. Default: data/datasets/aqua_{train|dev|test}.jsonl") 
    ap.add_argument("--max_examples", type=int, default=None, help="Optional cap for quick debugging") 
    args = ap.parse_args()

    split = args.split.lower()
    if split in {"val", "validation"}:
        split_key = "dev"                           
    else:
        split_key = split

    out = args.out
    if out is None:
        out = os.path.join("data", "datasets", f"aqua_{split_key}.jsonl")

    convert_and_write(split=split, out_path=out, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
