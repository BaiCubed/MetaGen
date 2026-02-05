                                   
from __future__ import annotations
import os, json
from typing import Dict, Iterable, List, Any

                                        
DATA_DIR = os.path.join("data", "datasets")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL 文件，每行是一个样本字典。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _builtin_arith_grid() -> List[Dict[str, Any]]:
    """
    简单算术网格，用于测试数值推理 / 性能：
      {"question": "Compute 2 + 3.", "answer": "5"}
    """
    out: List[Dict[str, Any]] = []
    for a in [1, 2, 3, 4, 5]:
        for b in [6, 7, 8]:
            out.append({"question": f"Compute {a} + {b}.", "answer": str(a + b)})
    for a in [2, 3, 4]:
        for b in [3, 4, 5]:
            out.append({"question": f"Compute {a} * {b}.", "answer": str(a * b)})
    return out

def _iter_aqua(path: str) -> Iterable[Dict[str, Any]]:
    """
    专门读取 AQuA (AQuA-RAT) 的 jsonl，并转成 MMLU 风格的多选题：
      - question: 题干 + Options 列表（方便 LLM 看）
      - choices: 选项列表
      - answer: 正确选项字母（A/B/C/D/E）
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            q = (ex.get("question") or "").strip()
            options = ex.get("options") or []
            correct = (ex.get("correct") or "").strip()

                        
            if options:
                                                      
                opt_lines = [str(opt) for opt in options]
                q_full = q + "\n\nOptions:\n" + "\n".join(opt_lines)
            else:
                q_full = q

            item = {
                "id": ex.get("id", f"aqua-{idx}"),
                "dataset": "aqua",
                "question": q_full,
                "choices": options,                      
                "answer": correct,                                          
                "rationale": ex.get("rationale", ""),
            }
            out.append(item)
    return out

def load_dataset(name: str) -> Iterable[Dict[str, Any]]:
    """
    返回一个可迭代对象，每个元素为一个 task 字典。

    统一支持的几类 schema：

      1) Short-answer / 算术类：
         {
           "question": str,
           "answer": str  # 数字或字符串
         }

      2) MMLU 多选题：
         {
           "question": str,
           "choices": ["A. ...","B. ...","C. ...","D. ..."],  # 或者纯文本选项
           "answer": "A"  # 或者 "B"/"C"/"D"，judge_correct 里会做字母归一
         }

      3) HumanEval 代码生成：
         {
           "task_id": str,
           "prompt": str,
           "tests": str,
           "entry_point": str
         }

      4) DROP / Span QA：
         {
           "id": str,
           "passage": str,          # 或 "context"
           "question": str,
           "answers": [str, ...]    # one or more gold spans
           # 可选地额外带一个 "answer": 主答案
         }

    解析顺序：

      - 若存在 data/datasets/<name>.jsonl 则直接加载；
      - 否则若 name 在 alias_to_file 里且文件存在，则加载别名路径；
      - 否则若 name 本身就是文件路径（相对或绝对），也允许加载；
      - 找不到则报错并提示支持的 schema。
    """
    name = str(name).strip()

    if name in ("aqua_test", "aqua_dev", "aqua_train"):
        aqua_path = os.path.join(DATA_DIR, f"{name}.jsonl")
        if os.path.isfile(aqua_path):
            return _iter_aqua(aqua_path)

                                            
    jsonl_path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if os.path.isfile(jsonl_path):
        return _iter_jsonl(jsonl_path)

                       
