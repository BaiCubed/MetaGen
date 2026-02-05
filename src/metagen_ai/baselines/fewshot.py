                                     
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import random
import re
import time

from metagen_ai.utils.llm import LLMClient


                          
                                                           
                          

def _is_code_task(task: Dict[str, Any]) -> bool:
    return isinstance(task, dict) and (
        "entry_point" in task or "canonical_solution" in task or "tests" in task
    )


def _is_mc_task(task: Dict[str, Any]) -> bool:
    return isinstance(task, dict) and isinstance(task.get("choices"), list) and len(task["choices"]) > 0


def _extract_num(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Final answer\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1).strip()
    m2 = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return m2[-1] if m2 else None


def _extract_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Final answer\s*:\s*([A-Z])", text)
    return m.group(1).strip().upper() if m else None


def _format_mc_question(stem: str, choices: List[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opt_lines = []
    for i, c in enumerate(choices):
        if i >= len(letters):
            break
        opt_lines.append(f"{letters[i]}. {c}")
    return (
        f"{stem.strip()}\n\n"
        "Options:\n" + "\n".join(opt_lines) + "\n\n"
        "Answer with exactly one line: Final answer: <LETTER>"
    )


def _needs_mc_options(q: str) -> bool:
    """Detect whether question text already includes option lines."""
    if not q:
        return True
    if "Options:" in q or "OPTIONS:" in q:
        return False
    if re.search(r"(?m)^\s*[A-E][\).]\s+", q):
        return False
    return True


                          
                         
                                                               
                          

_ARITH_FEWSHOT: List[Dict[str, str]] = [
    {
        "q": "If there are 12 cookies and you eat 5, how many are left?",
        "r": "Start with 12 cookies. Eat 5. Remaining = 12 - 5 = 7.",
        "a": "7",
    },
    {
        "q": "A notebook costs 3 dollars. If you buy 4 notebooks, how much do you pay?",
        "r": "Each costs 3. For 4 notebooks, total = 3 * 4 = 12.",
        "a": "12",
    },
    {
        "q": "Tom has 9 marbles and gives 2 to Anna. How many does Tom have now?",
        "r": "Tom gives away 2 from 9, so 9 - 2 = 7.",
        "a": "7",
    },
    {
        "q": "A box holds 6 oranges. How many oranges are in 5 boxes?",
        "r": "Each box has 6. With 5 boxes, total = 6 * 5 = 30.",
        "a": "30",
    },
    {
        "q": "Sara read 18 pages on Monday and 15 pages on Tuesday. How many pages did she read total?",
        "r": "Add the pages: 18 + 15 = 33.",
        "a": "33",
    },
]

_MNLI_FEWSHOT: List[Dict[str, Any]] = [
    {
        "stem": "Premise: The man is playing a guitar.\nHypothesis: The man is making music.",
        "choices": ["entailment", "neutral", "contradiction"],
        "r": "Playing a guitar is a way of making music, so the hypothesis follows.",
        "a": "A",
    },
    {
        "stem": "Premise: A woman is sleeping on a bench.\nHypothesis: A woman is running a marathon.",
        "choices": ["entailment", "neutral", "contradiction"],
        "r": "Sleeping on a bench is incompatible with running a marathon right now.",
        "a": "C",
    },
    {
        "stem": "Premise: Two kids are sitting at a table.\nHypothesis: The kids are eating dinner.",
        "choices": ["entailment", "neutral", "contradiction"],
        "r": "Sitting at a table does not imply they are eating; it could be many activities.",
        "a": "B",
    },
]

_MMLU_FEWSHOT_FALLBACK: List[Dict[str, Any]] = [
    {
        "stem": "Which of the following is a prime number?",
        "choices": ["9", "15", "17", "21"],
        "r": "17 has no positive divisors other than 1 and 17, so it is prime.",
        "a": "C",
    },
    {
        "stem": "Water boils at what temperature at sea level (in Celsius)?",
        "choices": ["0", "50", "100", "150"],
        "r": "At standard atmospheric pressure, water boils at 100°C.",
        "a": "C",
    },
]

_HUMANEVAL_FEWSHOT: List[Dict[str, Any]] = [
    {
        "prompt": (
            "Write a function add_two that takes an integer x and returns x + 2.\n\n"
            "def add_two(x: int) -> int:\n"
            "    \"\"\"Return x plus 2.\"\"\"\n"
        ),
        "entry_point": "add_two",
        "code": (
            "```python\n"
            "def add_two(x: int) -> int:\n"
            "    \"\"\"Return x plus 2.\"\"\"\n"
            "    return x + 2\n"
            "```"
        ),
    }
]


def run_fewshot_cot(
    llm: LLMClient,
    task: Dict[str, Any],
    *,
    shots_pool: Optional[List[Dict[str, Any]]] = None,
    n_shots: int = 5,
    seed: int = 0,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """Few-shot CoT baseline.

    Notes:
    - For MMLU, you can pass a pool loaded from mmlu_val (recommended) via shots_pool.
    - For other datasets, we use small built-in pools by default.
    """
    t0 = time.time()
    rnd = random.Random(seed)

    dataset = str(task.get("dataset") or "").lower()
    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)

    usage = 0

                              
                               
                              
    if is_code:
        pool = shots_pool if shots_pool is not None else _HUMANEVAL_FEWSHOT

                          
        exemplars = list(pool)
        rnd.shuffle(exemplars)
        exemplars = exemplars[: max(0, min(n_shots, len(exemplars)))]

        prompt = task.get("prompt") or task.get("question") or ""
        entry = task.get("entry_point", "solution")

        ex_blocks = []
        for ex in exemplars:
            ex_blocks.append(
                "### Example\n"
                f"Problem:\n{ex.get('prompt','')}\n\n"
                f"Solution:\n{ex.get('code','')}\n"
            )

        user = (
            "You are a Python programmer. Solve the coding task.\n"
            "Return ONLY a single Python code block.\n\n"
            + "\n\n".join(ex_blocks)
            + "\n\n### Task\n"
            f"Problem:\n{prompt}\n\n"
            f"Function name: {entry}\n\n"
            "Return ONLY one Python code block with your implementation."
        )

        resp = llm.chat(
            [
                {"role": "system", "content": "You write correct and readable Python code."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage += resp.get("usage", {}).get("total_tokens", 0)
        dt = time.time() - t0
        return resp.get("text", ""), {"total_tokens": usage}, dt

                              
                                            
                              
    if is_mc:
                                                                        
        if shots_pool is not None:
            pool = shots_pool
        else:
            pool = _MNLI_FEWSHOT if dataset == "mnli" else _MMLU_FEWSHOT_FALLBACK

        exemplars = list(pool)
        rnd.shuffle(exemplars)
        exemplars = exemplars[: max(0, min(n_shots, len(exemplars)))]

                          
        blocks = []
        for ex in exemplars:
            stem = ex.get("question") or ex.get("stem") or ""
            choices = ex.get("choices") or task.get("choices") or []
            if _needs_mc_options(stem) and choices:
                q_ex = _format_mc_question(stem, list(choices))
            else:
                q_ex = stem
            rationale = (ex.get("r") or ex.get("rationale") or "").strip()
            ans = ex.get("answer") or ex.get("a") or ""

                                                                          
            if rationale:
                a_ex = f"Let's think step by step.\n{rationale}\nFinal answer: {ans}"
            else:
                a_ex = f"Final answer: {ans}"
            blocks.append(f"Q: {q_ex}\nA: {a_ex}")

        stem = str(task.get("question") or "").strip()
        if _needs_mc_options(stem):
            stem = _format_mc_question(stem, list(task.get("choices") or []))

        user = (
            "You are solving multiple-choice questions.\n"
            "Use the examples as a guide. For the final question, think step by step and end with\n"
            "Final answer: <LETTER>\n\n"
            + "\n\n".join(blocks)
            + "\n\n"
            + f"Q: {stem}\nA: Let's think step by step."
        )

        resp = llm.chat(
            [
                {"role": "system", "content": "You are a careful reasoner."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage += resp.get("usage", {}).get("total_tokens", 0)
        txt = resp.get("text", "")
        letter = _extract_letter(txt) or "A"
        dt = time.time() - t0
        return letter, {"total_tokens": usage}, dt

                              
                                   
                              
    pool = shots_pool if shots_pool is not None else _ARITH_FEWSHOT
    exemplars = list(pool)
    rnd.shuffle(exemplars)
    exemplars = exemplars[: max(0, min(n_shots, len(exemplars)))]

    blocks = []
    for ex in exemplars:
        q_ex = ex.get("question") or ex.get("q") or ""
        r = (ex.get("r") or ex.get("rationale") or "").strip()
        a = ex.get("answer") or ex.get("a") or ""
        blocks.append(
            "Q: " + q_ex + "\n" +
            "A: Let's think step by step.\n" + r + "\n" +
            f"Final answer: {a}"
        )

    q = task.get("question") or ""
    user = (
        "You are solving math word problems.\n"
        "Use the examples as a guide. For the last question, think step by step and finish with\n"
        "Final answer: <number>\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + f"Q: {q}\nA: Let's think step by step."
    )

    resp = llm.chat(
        [
            {"role": "system", "content": "You are a careful math tutor."},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage += resp.get("usage", {}).get("total_tokens", 0)
    txt = resp.get("text", "")
    num = _extract_num(txt)
    final = f"Final answer: {num}" if num is not None else txt
    dt = time.time() - t0
    return final, {"total_tokens": usage}, dt
