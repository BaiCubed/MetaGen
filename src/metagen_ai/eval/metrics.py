                                
from __future__ import annotations
import os, re, sys, tempfile, subprocess
from typing import Dict, Any, Optional

                         
                         
                         
_NUM_PAT = re.compile(
    r"(?:####|Final answer|Answer|Result|Profit|Total)\s*(?:=|:|：)?\s*"
    r"(?:\\\(\s*)?(?:\\boxed\{\s*)?"
    r"([-+]?\$?\s*\d+(?:\.\d+)?|[-+]?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
    r"(?:\s*\})?(?:\s*\\\))?",
    re.IGNORECASE,
)
_CURRENCY_NUM = re.compile(r"[-+]?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_SIMPLE_NUM   = re.compile(r"[-+]?\$?\s*\d+(?:\.\d+)?")
_ONLY_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DIGIT_RUN    = re.compile(r"\d[\d,\.]*")

def _norm_num(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
                                            
    t = re.sub(r"[^\d\.\-\+]", "", str(s))
    if not t or t in {"+", "-", ".", "+.", "-."}:
        return None
    parts = t.split(".")
    if len(parts) > 2:
                                          
        t = parts[0] + "." + "".join(parts[1:])
    try:
        if "." in t:
            v = float(t)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return str(v).rstrip("0").rstrip(".")
        else:
            return str(int(t))        
    except Exception:
        return None

def _gold_to_letter(gold: Any) -> Optional[str]:
    """将 gold 转为 'A'|'B'|'C'|'D'|'E'；兼容 0–4（0基）、1–5（1基）、或字母本身。"""
    if gold is None:
        return None
                
    if isinstance(gold, str) and gold.strip().upper() in {"A", "B", "C", "D", "E"}:
        return gold.strip().upper()
               
    try:
        idx = int(str(gold).strip())
    except Exception:
        return None
                                      
    if 0 <= idx <= 4:
        return "ABCDE"[idx]
                                      
    if 1 <= idx <= 5:
        return "ABCDE"[idx - 1]
    return None

def _extract_number(txt: str) -> Optional[str]:
    """Normalized 抽取：依次尝试 _NUM_PAT -> 货币/千分位 -> 简单数字 -> 最长数字兜底。"""
    if not txt:
        return None
                                                          
    m = _NUM_PAT.search(txt)
    if m:
        return _norm_num(m.group(1))

               
    cands = _CURRENCY_NUM.findall(txt)
    if cands:
        v = _norm_num(cands[-1])
        if v is not None:
            return v

                    
    cands = _SIMPLE_NUM.findall(txt)
    if cands:
        v = _norm_num(cands[-1])
        if v is not None:
            return v

                                       
    runs = _DIGIT_RUN.findall(txt)
    if runs:
        best = None
        best_len = -1
        for r in runs:
            rr = re.sub(r"[^\d]", "", r)              
            if len(rr) > best_len:
                nn = _norm_num(r)
                if nn is not None:
                    best, best_len = nn, len(rr)
        if best is not None:
            return best

    return None

                                                                       


def _normalize_answer_drop(s: str) -> str:
    """
    DROP 风格的文本归一化：
    - 小写
    - 去掉标点
    - 去掉冠词 a/an/the
    - 折叠多余空格
    """
    s = s.lower()

                         
    s = re.sub(r"[^a-z0-9\s]", " ", s)

         
    s = re.sub(r"\b(a|an|the)\b", " ", s)

          
    s = " ".join(s.split())
    return s


def _drop_em_and_f1(pred: str, golds) -> tuple[float, float]:
    """
    pred: 预测字符串
    golds: str 或 list[str]
    返回: (best_em, best_f1)，在所有 gold 中取最优
    """
    if pred is None:
        return 0.0, 0.0

              
    if isinstance(golds, str):
        gold_list = [golds]
    elif isinstance(golds, (list, tuple)):
        gold_list = [g for g in golds if g is not None]
        if not gold_list:
            return 0.0, 0.0
    else:
        return 0.0, 0.0

    pred_norm = _normalize_answer_drop(str(pred))

    def _f1_single(p: str, g: str) -> tuple[float, float]:
        p_toks = p.split()
        g_toks = g.split()
        if not p_toks or not g_toks:
            return 0.0, 0.0
                     
        common = {}
        for t in p_toks:
            if t in g_toks:
                common[t] = common.get(t, 0) + 1
        num_same = sum(min(p_toks.count(t), g_toks.count(t)) for t in common)
        if num_same == 0:
            return 0.0, 0.0
        prec = num_same / len(p_toks)
        rec = num_same / len(g_toks)
        f1 = 2 * prec * rec / (prec + rec)
        em = 1.0 if p == g else 0.0
        return em, f1

    best_em, best_f1 = 0.0, 0.0
    for g in gold_list:
        g_norm = _normalize_answer_drop(str(g))
        em, f1 = _f1_single(pred_norm, g_norm)
        if f1 > best_f1:
            best_f1 = f1
        if em > best_em:
            best_em = em

    return best_em, best_f1


def _extract_span_for_drop(txt: str) -> str:
    """
    从模型输出中抽取用于 DROP 评测的答案 span：
    - 若包含 'Final answer:' 前缀，则取其后文本
    - 否则取全文，做 strip
    """
    if not txt:
        return ""
    m = re.search(r"(?i)final answer\s*:\s*(.+)", txt.strip())
    if m:
        return m.group(1).strip()
    return txt.strip()

_STRICT_LASTLINE = re.compile(r"(?m)^\s*####\s*(-?\d+(?:\.\d+)?)\s*$")

def _extract_number_strict(txt: str) -> Optional[str]:
    if not txt:
        return None
    m = _STRICT_LASTLINE.search(txt)
    return m.group(1) if m else None

def _extract_option_letter(txt: str, letters: str = "ABCD") -> Optional[str]:
    """Extract a single option letter from model output.

    For backwards-compatibility with existing MMLU runs, this keeps the
    original behavior: return the *first* standalone letter match.
    """
    if not txt:
        return None
    letters = str(letters or "ABCD").upper()
                                             
    m = re.search(rf"\b([{letters}])\b", txt.upper())
    return m.group(1) if m else None

                         
                     
                         
_PY_RUN = sys.executable

def _extract_solution_code(final_text: str) -> str:
                                              
    if not final_text:
        return ""
    m = re.search(r"```python(.*?)```", final_text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", final_text, re.S)
    if m:
        return m.group(1).strip()
    return final_text.strip()

def _run_humaneval_simple(solution_text: str, tests: str, timeout_s: int = 20) -> Dict[str, Any]:
    code = f"# --- Solution ---\n{solution_text}\n\n# --- Tests ---\n{tests}\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "heval_run.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run([_PY_RUN, path],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout_s, check=False, text=True)
            return {
                "passed": proc.returncode == 0,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "returncode": proc.returncode,
                "backend": "simple"
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stderr": "Timeout", "returncode": -1, "backend": "simple"}

def _run_humaneval_official(task_id: str, completion: str, timeout_s: int = 20) -> Dict[str, Any]:
               
    try:
        from .humaneval_official import evaluate_one_official
    except Exception as e:
        raise RuntimeError(f"Official HumanEval adapter not available: {e}")

    res = evaluate_one_official(task_id, completion, timeout=timeout_s, n_workers=1, k=1)
    return {"passed": bool(res.get("passed", False)), "backend": res.get("backend", "official"), "raw": res}

def _should_use_official() -> bool:
                                             
    env = os.environ.get("METAGEN_USE_OFFICIAL_HUMANEVAL", "").strip()
    if env in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    try:
        import importlib
        if importlib.util.find_spec("evalplus") is not None:
            return True
        if importlib.util.find_spec("human_eval") is not None:
            return True
    except Exception:
        pass
    return False

def judge_correct_strict(task: Dict[str, Any], final_text: str) -> bool:
    """显式严格评测：仅当最后一行形式 '#### <number>' 匹配 gold 时为 True。"""
                            
    is_he = ("task_id" in task) and ("prompt" in task)
    if is_he:
        return judge_correct(task, final_text)                       
    if isinstance(task.get("choices"), list) and task.get("choices"):
        return judge_correct(task, final_text)        
    gold = task.get("answer")
    if gold is None:
        return False
    pred = _extract_number_strict(final_text or "")
    return (pred is not None) and (pred.strip() == str(gold).strip())


                         
      
                         
def judge_correct(task: Dict[str, Any], final_text: str) -> bool:
    """
    统一判分入口：
      - HumanEval：默认优先使用官方（EvalPlus/human-eval），不可用时退回简易执行器；
      - DROP：基于 span 的 EM/F1（Final answer: <span> 或全文）；
      - MMLU：选项字母与 gold 比较；
      - 默认（short-answer QA 等）：抽数值与 gold 比较（货币/千分位/等号均鲁棒）。
    """
                     
    is_he = ("task_id" in task) and ("prompt" in task)
    if is_he:
        if _should_use_official():
            completion = _extract_solution_code(final_text)
            if not completion.strip():
                completion = str(task.get("prompt", ""))
            try:
                res = _run_humaneval_official(task_id=str(task["task_id"]), completion=completion, timeout_s=20)
                return bool(res.get("passed", False))
            except Exception:
                pass
        completion = _extract_solution_code(final_text)
        if not completion.strip():
            completion = str(task.get("prompt", ""))
        res = _run_humaneval_simple(completion, str(task.get("tests", "")), timeout_s=20)
        return bool(res.get("passed", False))

                                                     
    passage = task.get("passage") or task.get("context")
    answers = task.get("answers")
    answer = task.get("answer")

    is_drop_like = bool(passage) and (answers is not None or answer is not None)
    if is_drop_like:
                 
        if answers is not None:
            golds = answers
        else:
                                        
            golds = answer if isinstance(answer, (list, tuple)) else [answer]

                                               
        pred_span = _extract_span_for_drop(final_text or "")
        em, f1 = _drop_em_and_f1(pred_span, golds)
                                      
        return f1 >= 0.5

                
    if isinstance(task.get("choices"), list) and task.get("choices"):
        gold_letter = _gold_to_letter(task.get("answer"))
        if gold_letter is None:
            return False
        letters = "ABCDE" if len(task.get("choices") or []) >= 5 else "ABCD"
        pred = _extract_option_letter(final_text, letters=letters)
        return (pred is not None) and (pred == gold_letter)

                                           
    gold = task.get("answer")
    if gold is None:
        return False

                                                 
    use_strict = os.environ.get("METAGEN_QA_STRICT", "").strip().lower() in {"1","true","yes"}

    if use_strict:
        pred = _extract_number_strict(final_text or "")
        return (pred is not None) and (pred.strip() == str(gold).strip())

                             
    pred = _extract_number(final_text or "")
    return (_norm_num(pred) is not None) and (_norm_num(pred) == _norm_num(str(gold)))
