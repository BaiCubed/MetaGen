                                              
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


def _needs_mc_options(q: str) -> bool:
    if not q:
        return True
    if "Options:" in q or "OPTIONS:" in q:
        return False
    if re.search(r"(?m)^\s*[A-E][\).]\s+", q):
        return False
    return True


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


def _task_to_prompt(task: Dict[str, Any]) -> Tuple[str, str]:
    """Return (system, user) prompt pair for the task, with strict output formats."""
    if _is_code_task(task):
        prompt = task.get("prompt") or task.get("question") or ""
        entry = task.get("entry_point", "solution")
        sys = "You are an expert Python programmer. Write correct, efficient, readable code."
        user = (
            "Solve the coding task.\n"
            "Return ONLY a single Python code block.\n\n"
            f"Problem:\n{prompt}\n\n"
            f"Function name: {entry}\n\n"
            "Return ONLY one Python code block with your implementation."
        )
        return sys, user

    if _is_mc_task(task):
        stem = str(task.get("question") or "").strip()
        if _needs_mc_options(stem):
            stem = _format_mc_question(stem, list(task.get("choices") or []))
        sys = "You are an expert at multiple-choice reasoning."
        user = (
            "Answer the question. Think step by step, but end with exactly one line:\n"
            "Final answer: <LETTER>\n\n"
            f"{stem}"
        )
        return sys, user

                        
    q = task.get("question") or ""
    sys = "You are a careful math tutor."
    user = (
        "Solve the problem step by step, then end with exactly one line:\n"
        "Final answer: <number>\n\n"
        f"Problem: {q}"
    )
    return sys, user


def _postprocess(task: Dict[str, Any], text: str) -> str:
    """Normalize outputs to match judge_correct expectations."""
    if _is_code_task(task):
                                                                                 
        if not text:
            return ""
        m = re.search(r"```python(.*?)```", text, re.S | re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"```(.*?)```", text, re.S)
        if m:
            return m.group(1).strip()
        return text.strip()
    if _is_mc_task(task):
        return _extract_letter(text) or (text.strip()[:1].upper() if text else "")
             
    num = _extract_num(text)
    return f"Final answer: {num}" if num is not None else (text or "")


def run_chain_graph(
    llm: LLMClient,
    task: Dict[str, Any],
    *,
    steps: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """Chain graph baseline: iterative refinement (sequential DAG).

    Step1: solve; Step2..K: critique+fix previous attempt.
    """
    t0 = time.time()
    total = 0

    sys0, user0 = _task_to_prompt(task)
    resp = llm.chat(
        [{"role": "system", "content": sys0}, {"role": "user", "content": user0}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    total += resp.get("usage", {}).get("total_tokens", 0)
    cur = resp.get("text", "")

    for _ in range(max(0, steps - 1)):
        sys = "You are a strict reviewer who fixes mistakes and improves the answer."
        if _is_code_task(task):
            user = (
                "You will be given a coding task and a draft solution.\n"
                "Fix any bugs, ensure it passes the tests, and return ONLY one Python code block.\n\n"
                f"Task:\n{task.get('prompt') or task.get('question') or ''}\n\n"
                f"Draft solution:\n{cur}\n\n"
                "Return ONLY one Python code block with the corrected implementation."
            )
        elif _is_mc_task(task):
            stem = str(task.get("question") or "").strip()
            if _needs_mc_options(stem):
                stem = _format_mc_question(stem, list(task.get("choices") or []))
            user = (
                "You will be given a multiple-choice problem and a draft answer.\n"
                "Critique the reasoning, correct any mistakes, and output ONLY:\n"
                "Final answer: <LETTER>\n\n"
                f"Problem:\n{stem}\n\n"
                f"Draft answer:\n{cur}"
            )
        else:
            user = (
                "You will be given a math word problem and a draft solution.\n"
                "Fix mistakes and output ONLY one final line: Final answer: <number>\n\n"
                f"Problem: {task.get('question') or ''}\n\n"
                f"Draft solution:\n{cur}"
            )

        resp = llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total += resp.get("usage", {}).get("total_tokens", 0)
        cur = resp.get("text", "")

    final = _postprocess(task, cur)
    dt = time.time() - t0
    return final, {"total_tokens": total}, dt


def run_complete_graph(
    llm: LLMClient,
    task: Dict[str, Any],
    *,
    n_agents: int = 3,
    temperature: float = 0.5,
    judge_temperature: float = 0.2,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """Complete graph baseline (approximated as: propose -> cross-review -> judge)."""
    t0 = time.time()
    total = 0

    sys0, user0 = _task_to_prompt(task)
    drafts: List[str] = []
    for i in range(max(1, n_agents)):
        resp = llm.chat(
            [{"role": "system", "content": sys0}, {"role": "user", "content": user0}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total += resp.get("usage", {}).get("total_tokens", 0)
        drafts.append(resp.get("text", ""))

                                                                                 
    revised: List[str] = []
    for i, d in enumerate(drafts):
        others = [drafts[j] for j in range(len(drafts)) if j != i]
        sys = "You are a critical reviewer. Improve the draft using the other candidates as references."
        if _is_code_task(task):
            user = (
                "You will be given a coding task, your draft solution, and other candidates.\n"
                "Fix bugs and produce the BEST final solution. Return ONLY one Python code block.\n\n"
                f"Task:\n{task.get('prompt') or task.get('question') or ''}\n\n"
                f"Your draft:\n{d}\n\n"
                "Other candidates:\n" + "\n\n---\n\n".join(others)
                + "\n\nReturn ONLY one Python code block."
            )
        elif _is_mc_task(task):
            stem = str(task.get("question") or "").strip()
            if _needs_mc_options(stem):
                stem = _format_mc_question(stem, list(task.get("choices") or []))
            user = (
                "You will be given a multiple-choice problem, your draft, and other candidates.\n"
                "Decide the correct option. End with exactly one line: Final answer: <LETTER>\n\n"
                f"Problem:\n{stem}\n\n"
                f"Your draft:\n{d}\n\n"
                "Other candidates:\n" + "\n\n---\n\n".join(others)
            )
        else:
            user = (
                "You will be given a math problem, your draft, and other candidates.\n"
                "Produce the correct final answer. End with: Final answer: <number>\n\n"
                f"Problem: {task.get('question') or ''}\n\n"
                f"Your draft:\n{d}\n\n"
                "Other candidates:\n" + "\n\n---\n\n".join(others)
            )

        resp = llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total += resp.get("usage", {}).get("total_tokens", 0)
        revised.append(resp.get("text", ""))

                 
    sys = "You are a strict judge. Select (or synthesize) the best final answer."
    if _is_code_task(task):
        user = (
            "Choose the best solution that will pass the tests.\n"
            "Return ONLY one Python code block.\n\n"
            f"Task:\n{task.get('prompt') or task.get('question') or ''}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
            + "\n\nReturn ONLY one Python code block."
        )
    elif _is_mc_task(task):
        stem = str(task.get("question") or "").strip()
        if _needs_mc_options(stem):
            stem = _format_mc_question(stem, list(task.get("choices") or []))
        user = (
            "Choose the correct option. Output ONLY: Final answer: <LETTER>\n\n"
            f"Problem:\n{stem}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
        )
    else:
        user = (
            "Choose the correct final numeric answer. Output ONLY one line: Final answer: <number>\n\n"
            f"Problem: {task.get('question') or ''}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
        )

    resp = llm.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=judge_temperature,
        max_tokens=max_tokens,
    )
    total += resp.get("usage", {}).get("total_tokens", 0)
    final = _postprocess(task, resp.get("text", ""))
    dt = time.time() - t0
    return final, {"total_tokens": total}, dt


def run_random_graph(
    llm: LLMClient,
    task: Dict[str, Any],
    *,
    n_agents: int = 3,
    edge_prob: float = 0.5,
    seed: int = 0,
    temperature: float = 0.5,
    judge_temperature: float = 0.2,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """Random DAG baseline: propose -> partial cross-review (random incoming edges) -> judge."""
    t0 = time.time()
    rnd = random.Random(seed)
    total = 0

    sys0, user0 = _task_to_prompt(task)
    drafts: List[str] = []
    for _ in range(max(1, n_agents)):
        resp = llm.chat(
            [{"role": "system", "content": sys0}, {"role": "user", "content": user0}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total += resp.get("usage", {}).get("total_tokens", 0)
        drafts.append(resp.get("text", ""))

                                                          
    incoming: List[List[int]] = [[] for _ in range(len(drafts))]
    for i in range(len(drafts)):
        for j in range(i):
            if rnd.random() < max(0.0, min(1.0, edge_prob)):
                incoming[i].append(j)

    revised: List[str] = []
    for i, d in enumerate(drafts):
        neigh = [drafts[j] for j in incoming[i]]
        if not neigh:
            revised.append(d)
            continue

        sys = "You are a critical reviewer. Improve the draft using the neighbor candidates as references."
        if _is_code_task(task):
            user = (
                "You will be given a coding task, your draft, and neighbor candidates.\n"
                "Fix bugs and return ONLY one Python code block.\n\n"
                f"Task:\n{task.get('prompt') or task.get('question') or ''}\n\n"
                f"Your draft:\n{d}\n\n"
                "Neighbor candidates:\n" + "\n\n---\n\n".join(neigh)
                + "\n\nReturn ONLY one Python code block."
            )
        elif _is_mc_task(task):
            stem = str(task.get("question") or "").strip()
            if _needs_mc_options(stem):
                stem = _format_mc_question(stem, list(task.get("choices") or []))
            user = (
                "You will be given a multiple-choice problem, your draft, and neighbor candidates.\n"
                "Decide the correct option. End with: Final answer: <LETTER>\n\n"
                f"Problem:\n{stem}\n\n"
                f"Your draft:\n{d}\n\n"
                "Neighbor candidates:\n" + "\n\n---\n\n".join(neigh)
            )
        else:
            user = (
                "You will be given a math problem, your draft, and neighbor candidates.\n"
                "Fix mistakes and output Final answer: <number>\n\n"
                f"Problem: {task.get('question') or ''}\n\n"
                f"Your draft:\n{d}\n\n"
                "Neighbor candidates:\n" + "\n\n---\n\n".join(neigh)
            )

        resp = llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total += resp.get("usage", {}).get("total_tokens", 0)
        revised.append(resp.get("text", ""))

                                  
    sys = "You are a strict judge. Select (or synthesize) the best final answer."
    if _is_code_task(task):
        user = (
            "Choose the best solution that will pass the tests.\n"
            "Return ONLY one Python code block.\n\n"
            f"Task:\n{task.get('prompt') or task.get('question') or ''}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
            + "\n\nReturn ONLY one Python code block."
        )
    elif _is_mc_task(task):
        stem = str(task.get("question") or "").strip()
        if _needs_mc_options(stem):
            stem = _format_mc_question(stem, list(task.get("choices") or []))
        user = (
            "Choose the correct option. Output ONLY: Final answer: <LETTER>\n\n"
            f"Problem:\n{stem}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
        )
    else:
        user = (
            "Choose the correct final numeric answer. Output ONLY one line: Final answer: <number>\n\n"
            f"Problem: {task.get('question') or ''}\n\n"
            "Candidates:\n" + "\n\n---\n\n".join(revised)
        )

    resp = llm.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=judge_temperature,
        max_tokens=max_tokens,
    )
    total += resp.get("usage", {}).get("total_tokens", 0)
    final = _postprocess(task, resp.get("text", ""))
    dt = time.time() - t0
    return final, {"total_tokens": total}, dt
