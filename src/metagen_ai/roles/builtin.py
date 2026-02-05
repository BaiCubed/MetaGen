                                 
from __future__ import annotations

import re, ast
from typing import Dict, Any, List, Optional
from collections import Counter
from .schema import RoleProfile
from typing import Optional
__all__ = ["BUILTIN_ROLES"]

                               
                               
_TAG_PATTERNS = [
    ("DECISION",      re.compile(r"(?i)\bDECISION\s*:\s*([ABCD])\b")),
    ("Final answer",  re.compile(r"(?i)\bFinal answer\s*:\s*([ABCD])\b")),
    ("RECOMMEND",     re.compile(r"(?i)\bRECOMMEND\s*:\s*([ABCD])\b")),
    ("CANDIDATE",     re.compile(r"(?i)\bCANDIDATE\s*:\s*([ABCD])\b")),
    ("Helper answer", re.compile(r"(?i)\bHelper answer\s*:\s*([ABCD])\b")),
]
_BARE_LETTER = re.compile(r"(?im)^\s*([ABCD])\s*$")


                                        
_AQUA_TAG_PATTERNS = [
    ("DECISION",      re.compile(r"(?i)\bDECISION\s*:\s*([ABCDE])\b")),
    ("Final answer",  re.compile(r"(?i)\bFinal answer\s*:\s*([ABCDE])\b")),
    ("RECOMMEND",     re.compile(r"(?i)\bRECOMMEND\s*:\s*([ABCDE])\b")),
    ("CANDIDATE",     re.compile(r"(?i)\bCANDIDATE\s*:\s*([ABCDE])\b")),
    ("Helper answer", re.compile(r"(?i)\bHelper answer\s*:\s*([ABCDE])\b")),
]
_AQUA_BARE_LETTER = re.compile(r"(?im)^\s*([ABCDE])\s*$")

def _aqua_final_handler(ctx):
    """Deterministically produce: Final answer: <LETTER> for AQuA-style MCQ.

    Supports both 4-way (A–D) and 5-way (A–E) variants by reading task['choices'].
    """
    task = ctx.get("task", {}) or {}
    choices = task.get("choices") or task.get("options") or []
    n = len(choices) if isinstance(choices, (list, tuple)) else 0
    allowed = "ABCDE"[:5] if n <= 0 else "ABCDE"[:min(5, max(2, n))]
    allowed_set = set(allowed)
    inputs = ctx.get("inputs", {}) or {}

                                     
    arb = _to_text(inputs.get("aqua_arbiter"))
    if arb:
        dec = _last_match(_AQUA_TAG_PATTERNS[0][1], arb)            
        if dec in allowed_set:
            return f"Final answer: {dec}"

                                         
    texts = []
    if isinstance(inputs, dict):
        for k, v in sorted(inputs.items(), key=lambda x: str(x[0])):
            texts.append(_to_text(v))
    else:
        texts.append(_to_text(inputs))
    joined = "\n\n".join(t for t in texts if t)

    for _tag, pat in _AQUA_TAG_PATTERNS[1:]:
        letter = _last_match(pat, joined)
        if letter in allowed_set:
            return f"Final answer: {letter}"

    bare = _last_match(_AQUA_BARE_LETTER, joined)
    if bare in allowed_set:
        return f"Final answer: {bare}"

                                                    
    votes = []
    for _tag, pat in _AQUA_TAG_PATTERNS:
        for m in pat.finditer(joined or ""):
            votes.append(m.group(1).upper())
    for m in _AQUA_BARE_LETTER.finditer(joined or ""):
        votes.append(m.group(1).upper())

    votes = [v for v in votes if v in allowed_set]
    if votes:
        c = Counter(votes)
        best_cnt = c.most_common(1)[0][1]
        tied = [k for k, cnt in c.items() if cnt == best_cnt]
        for k in ["A", "B", "C", "D", "E"]:
            if k not in allowed_set:
                continue
            if k in tied:
                return f"Final answer: {k}"

                                
    return f"Final answer: {allowed[0] if allowed else 'A'}"

def _to_text(v: Any) -> str:
    """Make ctx.inputs values robust: accept str / dict(message) / list(messages) / others."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
                               
        for k in ("content", "text", "message"):
            if k in v and isinstance(v[k], str):
                return v[k]
                                                           
        return str(v)
    if isinstance(v, list):
        return "\n".join(_to_text(x) for x in v)
    return str(v)

def _last_match(pat: re.Pattern, text: str) -> Optional[str]:
    m = None
    for m in pat.finditer(text or ""):
        pass
    return (m.group(1).upper() if m else None)

def _collect_votes_from_text(text: str):
    votes = []
    for tag, pat in _TAG_PATTERNS:
        for m in pat.finditer(text or ""):
            votes.append((tag, m.group(1).upper()))
    for m in _BARE_LETTER.finditer(text or ""):
        votes.append(("BARE", m.group(1).upper()))
    return votes

def _mmlu_final_handler(ctx):
    """
    Deterministically produce: Final answer: <LETTER>

    Strong rule:
      1) If mmlu_arbiter exists, ONLY trust its DECISION.
      2) Otherwise fall back to tagged votes (Final answer/RECOMMEND/CANDIDATE/Helper answer).
      3) Otherwise majority vote; tie -> A>B>C>D.
    """
    inputs = ctx.get("inputs", {}) or {}

                                    
    arb = _to_text(inputs.get("mmlu_arbiter"))
    if arb:
        dec = _last_match(_TAG_PATTERNS[0][1], arb)            
        if dec in ("A", "B", "C", "D"):
            return f"Final answer: {dec}"

                                               
    texts = []
    if isinstance(inputs, dict):
                      
        for k, v in sorted(inputs.items(), key=lambda x: str(x[0])):
            texts.append(_to_text(v))
    else:
        texts.append(_to_text(inputs))

    joined = "\n\n".join(t for t in texts if t)

    for _tag, pat in _TAG_PATTERNS[1:]:                                   
        letter = _last_match(pat, joined)
        if letter in ("A","B","C","D"):
            return f"Final answer: {letter}"

    bare = _last_match(_BARE_LETTER, joined)
    if bare in ("A","B","C","D"):
        return f"Final answer: {bare}"

    votes = [v for _tag, v in _collect_votes_from_text(joined) if v in ("A","B","C","D")]
    if votes:
        c = Counter(votes)
        best_cnt = c.most_common(1)[0][1]
        tied = [k for k, cnt in c.items() if cnt == best_cnt]
        for k in ["A","B","C","D"]:
            if k in tied:
                return f"Final answer: {k}"

                                
    return f"Final answer: {allowed[0] if allowed else 'A'}"

def _extract_numbers(s: str) -> List[float | int]:
    return [float(x) if "." in x else int(x) for x in _NUMBER_RE.findall(str(s))]

def _gather_text(ctx: Dict[str, Any]) -> str:
    task = ctx.get("task", {}) or {}
    ins = ctx.get("inputs", {}) or {}
    prev = ctx.get("prev_summary", "") or ""
    parts: List[str] = []
    parts.append(str(task.get("question") or task.get("prompt") or task))
    parts.append(str(prev))
    if isinstance(ins, dict):
        parts.extend([str(v) for v in ins.values()])
    return "\n".join(p for p in parts if p)

def _extract_code_blocks(text: str) -> List[str]:
    if not text:
        return []
    blocks = []
    for m in re.finditer(r"```(?:python)?\s*(.*?)```", text, re.S | re.I):
        code = (m.group(1) or "").strip()
        if code:
            blocks.append(code)
    return blocks

def _first_func_name_from_prompt(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    m = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", prompt)
    return m.group(1) if m else None

                                                                    

def _safe_eval(expr: str):
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
        ast.Constant, ast.Tuple
    )
    node = ast.parse(expr, mode="eval")

    def _check(n):
        if not isinstance(n, allowed):
            raise ValueError(f"Unsafe node: {type(n).__name__}")
        for child in ast.iter_child_nodes(n):
            _check(child)
    _check(node)
    return eval(compile(node, "<safe-eval>", "eval"), {"__builtins__": {}}, {})

def _code_evaluator_handler(ctx: Dict[str, Any]) -> str:
    task = ctx.get("task", {}) or {}
    ins = ctx.get("inputs", {}) or {}
    prompt_text = str(task.get("prompt") or task.get("question") or "")
    entry_point = task.get("entry_point") or _first_func_name_from_prompt(prompt_text)
    upstream_text = "\n".join([str(v) for v in ins.values()]) if ins else ""
    blocks = _extract_code_blocks(upstream_text)

    def _has_entry(code: str, name: Optional[str]) -> bool:
        if not name:
            return False
        pat = rf"(?m)^\s*def\s+{re.escape(name)}\s*\("
        return re.search(pat, code) is not None

    picked: Optional[str] = None
    if blocks:
        if entry_point:
            with_entry = [b for b in blocks if _has_entry(b, entry_point)]
            if with_entry:
                with_entry.sort(key=lambda x: x.count("\n"), reverse=True)
                picked = with_entry[0]
        if picked is None:
            blocks.sort(key=lambda x: len(x), reverse=True)
            picked = blocks[0]
    if picked is None:
        name = entry_point or "solution"
        picked = f"def {name}(*args, **kwargs):\n    # TODO: implement\n    raise NotImplementedError"
    return f"```python\n{picked}\n```"

def _static_check_handler(ctx: Dict[str, Any]) -> str:
    """
    HumanEval 静态检查：不执行代码，只看是否安全 & 是否包含入口函数。
    输出一行：
      - 'CHECK: OK'
      - 或 'REPAIR: <简短提示>'
    """
    task = ctx.get("task", {}) or {}
    ins  = ctx.get("inputs", {}) or {}

    prompt_text = str(task.get("prompt") or task.get("question") or "")
    entry_point = task.get("entry_point") or _first_func_name_from_prompt(prompt_text)

    upstream_text = "\n".join([str(v) for v in ins.values()]) if ins else ""
    blocks = _extract_code_blocks(upstream_text)
    if not blocks:
        return "REPAIR: no code block found"

    code = blocks[-1]
    bad_tokens = [
        r"\beval\s*\(", r"\bexec\s*\(", r"__import__", r"\bopen\s*\(",
        r"\bsubprocess\b", r"\bsocket\b", r"\brequests\b", r"\bos\b", r"\bsys\b",
    ]
    for pat in bad_tokens:
        if re.search(pat, code):
            return "REPAIR: disallow unsafe or external dependencies; use pure Python only"

    if entry_point:
        if not re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", code):
            return f"REPAIR: missing required entry function `{entry_point}`"

    return "CHECK: OK"

                                      

BUILTIN_ROLES: Dict[str, RoleProfile] = {
                             
                   
                             

    "policy_guard": RoleProfile(
        name="policy_guard",
        description="Detect and fix bad percentage base patterns; if needed, rewrite EQUATION.",
        system_template=(
            "You verify the percentage base policy:\n"
            "- Base for 'increased by X%' is the pre-event asset value.\n"
            "- Do NOT include costs/repairs in the multiplicative base.\n"
            "If the provided EQUATION violates the policy (e.g., (P + R) * (1 + RATE)), rewrite it.\n"
            "Output EXACTLY one line:\n"
            "EQUATION: <correct_expression_with_numbers>"
        ),
        user_template=(
            "Upstream:\n{inputs}\n"
            "If the previous EQUATION already satisfies the policy, repeat it unchanged.\n"
            "Otherwise rewrite it so that only P is the base of the percentage increase.\n"
            "Output exactly:\n"
            "EQUATION: <expression>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["verify"]
    ),






                             
                       
                             
    "programmer": RoleProfile(
        name="programmer",
        description="Writes the full Python implementation for the given function signature and docstring.",
        system_template=(
            "You are a senior Python engineer.\n"
            "Return ONLY a Python code block implementing the requested function.\n"
            "Do not include any explanations or text outside the code block."
        ),
        user_template=(
            "Implement the function described below.\n"
            "Constraints:\n"
            "1) Keep the SAME function name and parameters as given.\n"
            "2) Pure Python; no external packages.\n"
            "3) Include any helpers inside the same file.\n"
            "4) If tests imply edge cases, handle them.\n"
            "\n"
            "Function prompt:\n{task}\n"
        ),
        local_handler=None,
        temperature=0.1,
        capabilities=["compute"]
    ),

    "code_evaluator": RoleProfile(
        name="code_evaluator",
        description="Selects the best code implementation from upstream candidates or synthesizes a stub; outputs a single Python code block.",
        system_template=(
            "You are a strict code selector. "
            "Your output MUST be exactly one Python code block with the final solution. "
            "Prefer a block that defines the required entry point function if known."
        ),
        user_template=(
            "Task:\n{task}\n"
            "Upstream candidates (may include multiple code blocks):\n{inputs}\n"
            "Return ONLY one Python code block. No extra text."
        ),
        local_handler=_code_evaluator_handler,
        temperature=0.0,
        capabilities=["verify"]
    ),

    "spec_parser": RoleProfile(
        name="spec_parser",
        description="Extract entry function name (ENTRY) and hard constraints (CONSTRAINTS) from HumanEval prompt.",
        system_template=(
            "You are a spec extractor. Read the HumanEval prompt and output exactly two lines:\n"
            "ENTRY: <function_name>\n"
            "CONSTRAINTS: <short bullet-like text; avoid extra punctuation>\n"
            "If unsure about constraints, keep them minimal."
        ),
        user_template=(
            "Prompt:\n{task}\n"
            "Output exactly two lines:\n"
            "ENTRY: <function_name>\n"
            "CONSTRAINTS: <short text>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["parse"]
    ),

    "static_checker": RoleProfile(
        name="static_checker",
        description="Deterministic linter to verify safety and presence of entry function. Returns one line.",
        system_template=(
            "You are a deterministic static checker.\n"
            "You must return exactly one line:\n"
            "  CHECK: OK\n"
            "or\n"
            "  REPAIR: <short hint>\n"
        ),
        user_template=(
            "Upstream may contain one or more Python code blocks:\n{inputs}\n"
            "Check safety and required entry function (if provided in the prompt).\n"
            "Return exactly one line: 'CHECK: OK' or 'REPAIR: <short hint>'."
        ),
        local_handler=_static_check_handler,
        temperature=0.0,
        capabilities=["verify"]
    ),

    "repairer": RoleProfile(
        name="repairer",
        description="If static check requests REPAIR, produce a single fixed Python code block; otherwise pass-through best candidate.",
        system_template=(
            "You are a strict code fixer.\n"
            "Rules:\n"
            "- If upstream contains a line 'REPAIR: <hint>', you MUST output a SINGLE Python code block implementing the function, applying the hint.\n"
            "- If upstream says 'CHECK: OK', you should forward the best candidate code as-is (one Python code block only).\n"
            "- DO NOT output any extra text."
        ),
        user_template=(
            "Task:\n{task}\n"
            "Upstream:\n{inputs}\n"
            "If there is a 'REPAIR: ...' line, fix the code accordingly and output ONE Python code block only.\n"
            "If 'CHECK: OK', output the best candidate code as ONE Python code block only."
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["compute", "verify"]
    ),

                             
                  
                             

    "drop_reader": RoleProfile(
        name="drop_reader",
        description="Few-shot style extractive reader for DROP: given passage + question, output a short answer span from the passage.",
        system_template=(
            "You answer DROP-style reading comprehension questions.\n"
            "You MUST ALWAYS answer using a SHORT SPAN copied from the passage.\n"
            "Do NOT invent names or numbers that do not appear in the passage.\n"
            "Do NOT answer 'I don't know' or 'not given' — the answer is always present.\n"
            "\n"
            "Examples:\n"
            "Example 1\n"
            "Passage: John went to the store. He bought 3 apples and 2 bananas.\n"
            "Question: What did John buy 3 of?\n"
            "Answer: 3 apples\n"
            "\n"
            "Example 2\n"
            "Passage: The Raiders scored first with a 20-yard touchdown pass from JaMarcus Russell to Chaz Schilens.\n"
            "Question: Who scored the first touchdown?\n"
            "Answer: Chaz Schilens\n"
            "\n"
            "You MUST follow the pattern above: a short text span copied from the passage, with no explanation.\n"
        ),
        user_template=(
            "You will see a passage and a question inside the upstream messages.\n"
            "They may be formatted like:\n"
            "  Passage: <text>\n"
            "  Question: <text>\n"
            "\n"
            "Upstream:\n{inputs}\n"
            "\n"
            "Now answer the question with a SHORT SPAN copied from the passage.\n"
            "Output exactly ONE line in the format:\n"
            "SPAN CANDIDATE: <answer_span>"
        ),
        local_handler=None,
        temperature=0.1,
        capabilities=["read", "extract"]
    ),

    "drop_span_proposer": RoleProfile(
        name="drop_span_proposer",
        description="Pass-through / light cleaner for span candidates in DROP; keeps a single SPAN CANDIDATE line.",
        system_template=(
            "You are a light cleaner.\n"
            "You will receive some text containing a line like:\n"
            "  SPAN CANDIDATE: <answer_span>\n"
            "Your job is to output exactly ONE line in the same format, copying the <answer_span>.\n"
            "If there are multiple SPAN CANDIDATE lines, use the LAST one.\n"
            "Do NOT output explanations.\n"
        ),
        user_template=(
            "Upstream:\n{inputs}\n"
            "Extract the LAST 'SPAN CANDIDATE: <text>' line and output exactly:\n"
            "SPAN CANDIDATE: <text>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["aggregate", "extract"]
    ),

    "drop_span_aggregator": RoleProfile(
        name="drop_span_aggregator",
        description="Aggregate one or more span candidates for DROP and output a single CANDIDATE ANSWER line.",
        system_template=(
            "You aggregate span candidates for DROP.\n"
            "You will see one or more lines of the form:\n"
            "  SPAN CANDIDATE: <text>\n"
            "If there are several, choose the BEST one (usually they are identical or similar).\n"
            "Then output exactly ONE line:\n"
            "CANDIDATE ANSWER: <text>\n"
            "Do NOT answer 'I don't know'. Always choose some candidate text."
        ),
        user_template=(
            "Upstream:\n{inputs}\n"
            "From all lines of the form 'SPAN CANDIDATE: <text>', choose the best candidate span.\n"
            "Output exactly one line:\n"
            "CANDIDATE ANSWER: <text>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["aggregate", "extract"]
    ),

    "drop_answerer": RoleProfile(
        name="drop_answerer",
        description="Finalizer for DROP: wrap the chosen span as 'Final answer: <span>'.",
        system_template=(
            "You finalize answers for DROP.\n"
            "You will see upstream text that contains a line like:\n"
            "  CANDIDATE ANSWER: <text>\n"
            "or, if that is missing, a line like:\n"
            "  SPAN CANDIDATE: <text>\n"
            "\n"
            "You MUST:\n"
            "1) Find the LAST such line.\n"
            "2) Copy <text> EXACTLY (do not paraphrase, do not translate, do not add words).\n"
            "3) Output exactly ONE line:\n"
            "   Final answer: <text>\n"
            "\n"
            "You MUST NOT answer 'I don't know', 'not given', or similar. Always copy the candidate span."
        ),
        user_template=(
            "Upstream messages:\n{inputs}\n"
            "Find the LAST line that looks like either:\n"
            "  CANDIDATE ANSWER: <text>\n"
            "or\n"
            "  SPAN CANDIDATE: <text>\n"
            "Copy <text> exactly and output:\n"
            "Final answer: <text>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["finalize", "extract"]
    ),

    "qa_answerer": RoleProfile(
        name="qa_answerer",
        description="Answer short-answer questions and output a single final line.",
        system_template=(
            "You answer questions concisely and correctly.\n"
            "Output EXACTLY one line: Final answer: <answer>\n"
            "Do not add any other text."
        ),
        user_template=(
            "Task: {task}\n"
            "Question: {question}\n\n"
            "Return exactly one line: Final answer: <answer>"
        ),
        local_handler=None,
        temperature=0.2,
        capabilities=["answer"],
    ),

                             
                     
                             
    "mmlu_direct_answerer": RoleProfile(
        name="mmlu_direct_answerer",
        description="Answer multiple-choice questions like MMLU by selecting one option letter.",
        system_template=(
            "You are a high-accuracy multiple-choice question solver.\n"
            "You will be given a task dictionary that includes:\n"
            "- a question\n"
            "- a list of answer choices\n"
            "- optionally other metadata (subject, id, etc.).\n\n"
            "You may also receive upstream helper analyses in the context. "
            "Helpers may end their outputs with lines like 'Helper answer: <LETTER>'.\n\n"
            "Your job is to choose the SINGLE BEST answer.\n"
            "Rules:\n"
            "1) Silently read the question and the helper outputs.\n"
            "2) You MAY use the helper answers as a hint, but you must still ensure the choice is logically correct.\n"
            "3) You MUST output exactly one line in the form:\n"
            "   Final answer: <LETTER>\n"
            "   where <LETTER> is exactly one of: A, B, C, D.\n"
            "4) Do NOT output any reasoning or extra text, only that single line."
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream context (may be empty):\n{inputs}\n\n"
            "The task dict contains a field 'question' and a field 'choices', "
            "where 'choices' is an ordered list of answer options.\n"
            "Map choices as:\n"
            "A -> choices[0]\n"
            "B -> choices[1]\n"
            "C -> choices[2]\n"
            "D -> choices[3]\n\n"
            "You MUST output exactly one line: Final answer: <LETTER>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["compute", "verify"]
    ),
    "mmlu_reasoner": RoleProfile(
        name="mmlu_reasoner",
        description="Deep reasoning for MMLU multiple-choice; end with CANDIDATE letter.",
        system_template=(
            "You are a careful solver for MMLU-style multiple-choice questions.\n"
            "You will be given ONE question and its options.\n"
            "Use only this question; do NOT invent other problems.\n\n"
            "Output format:\n"
            "- Write your reasoning (can be detailed).\n"
            "- The LAST line MUST be exactly: CANDIDATE: <A|B|C|D>\n"
            "Do NOT write 'Final answer'."
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream (helpers may exist):\n{inputs}\n\n"
            "Solve ONLY this task. Do NOT switch to unrelated problems.\n"
            "The task dict contains 'question' and 'choices' (ordered list).\n"
            "Map letters: A->choices[0], B->choices[1], C->choices[2], D->choices[3].\n\n"
            "Output format:\n"
            "- Write your reasoning.\n"
            "- The LAST line MUST be exactly: CANDIDATE: <A|B|C|D>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["analyze", "compute"]
    ),
    "mmlu_critic": RoleProfile(
        name="mmlu_critic",
        description="Critique and recommend a final choice letter; end with RECOMMEND.",
        system_template=(
            "You are a strict verifier.\n"
            "Given the question/options and an upstream solver's reasoning, check for mistakes.\n"
            "If solver is wrong, override.\n\n"
            "Output format:\n"
            "- Up to 4 short sentences.\n"
            "- The LAST line MUST be exactly: RECOMMEND: <A|B|C|D>\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream messages:\n{inputs}\n\n"
            "Verify the upstream reasoning strictly. If inconsistent, override.\n"
            "The task dict contains 'question' and 'choices'. Use A/B/C/D mapping by index.\n"
            "Output format:\n"
            "- Up to 4 short sentences.\n"
            "- The LAST line MUST be exactly: RECOMMEND: <A|B|C|D>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["verify"]
    ),
    "mmlu_refiner": RoleProfile(
        name="mmlu_refiner",
        description="Refine final decision using critic feedback; end with CANDIDATE.",
        system_template=(
            "You refine the final choice based on critic feedback.\n"
            "Return a short justification (optional) but MUST end with:\n"
            "CANDIDATE: <A|B|C|D>\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream (reasoner + critic):\n{inputs}\n\n"
            "Resolve disagreements and choose the best option.\n"
            "The task dict contains 'question' and 'choices'. Use A/B/C/D mapping by index.\n"
            "Return optional short justification, but MUST end with:\n"
            "CANDIDATE: <A|B|C|D>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["refine"]
    ),
    "mmlu_arbiter": RoleProfile(
        name="mmlu_arbiter",
        description="Arbitrate among multiple solvers for MMLU; output ONE strict decision line.",
        system_template=(
            "You are the FINAL ARBITER for a multiple-choice question.\n"
            "You will receive the question/options and upstream analyses.\n"
            "You MUST output exactly ONE line:\n"
            "DECISION: <A|B|C|D>\n"
            "No other text."
        ),
        user_template=(
            "Task: {task}\n\n"
            "Upstream analyses:\n{inputs}\n\n"
            "No other text. Just output exactly one line: DECISION: <A|B|C|D>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["finalize","verify"]
    ),
    "mmlu_answerer": RoleProfile(
        name="mmlu_answerer",
        description="Strict formatter: extract letter from upstream and output one line.",
        system_template=(
            "You are a strict formatter.\n"
            "You will be given upstream messages that may contain:\n"
            "DECISION: X / RECOMMEND: X / CANDIDATE: X / Helper answer: X / Final answer: X.\n\n"
            "Rules (hard):\n"
            "1) If any 'DECISION: <A|B|C|D>' exists, output that letter.\n"
            "2) Else if any 'Final answer:' exists, output the last one.\n"
            "3) Else prefer last RECOMMEND, then last CANDIDATE, then last Helper answer.\n"
            "4) Output MUST be EXACTLY ONE LINE:\n"
            "Final answer: <A|B|C|D>\n"
            "No other text."
        ),
        user_template=(
            "Upstream messages:\n{inputs}\n"
        ),
        local_handler=_mmlu_final_handler,                              
        temperature=0.0,
        capabilities=["finalize", "extract"],
    ),

                                                                 
    "aqua_reasoner": RoleProfile(
        name="aqua_reasoner",
        description="Solve AQuA multiple-choice question; end with CANDIDATE.",
        system_template=(
            "You solve AQuA-style multiple-choice questions (AQuA-RAT), which typically require step-by-step math reasoning.\n"
            "Use the given options and choose the best one.\n"
            "You MAY write a short reasoning, but MUST end with:\n"
            "CANDIDATE: <A|B|C|D|E>\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream context (may include helper suggestions):\n{inputs}\n\n"
            "Solve the question and pick the best option.\n"
            "Use A/B/C/D/E mapping by the index order in task['choices'].\n"
            "Return optional short reasoning, but MUST end with:\n"
            "CANDIDATE: <A|B|C|D|E>\n"
        ),
        local_handler=None,
        temperature=0.2,
        capabilities=["analyze","compute"]
    ),
    "aqua_critic": RoleProfile(
        name="aqua_critic",
        description="Critique the reasoner's candidate; end with RECOMMEND.",
        system_template=(
            "You critique the proposed solution and choice for an AQuA-style 5-way multiple-choice question.\n"
            "Identify mistakes or confirm correctness.\n"
            "You MAY write a short critique, but MUST end with:\n"
            "RECOMMEND: <A|B|C|D|E>\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream (reasoner + helpers if any):\n{inputs}\n\n"
            "Critique the candidate and provide the best recommendation.\n"
            "MUST end with: RECOMMEND: <A|B|C|D|E>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["verify","critic"]
    ),
    "aqua_refiner": RoleProfile(
        name="aqua_refiner",
        description="Refine final decision using critic feedback; end with CANDIDATE.",
        system_template=(
            "You refine the final choice based on critic feedback.\n"
            "Return a short justification (optional) but MUST end with:\n"
            "CANDIDATE: <A|B|C|D|E>\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream (reasoner + critic):\n{inputs}\n\n"
            "Resolve disagreements and choose the best option.\n"
            "Use A/B/C/D/E mapping by index.\n"
            "Return optional short justification, but MUST end with:\n"
            "CANDIDATE: <A|B|C|D|E>\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["refine"]
    ),
    "aqua_arbiter": RoleProfile(
        name="aqua_arbiter",
        description="Arbitrate among multiple solvers for AQuA; output ONE strict decision line.",
        system_template=(
            "You are the FINAL ARBITER for a 5-way multiple-choice question (AQuA-RAT).\n"
            "You will receive the question/options and upstream analyses.\n"
            "You MUST output exactly ONE line:\n"
            "DECISION: <A|B|C|D|E>\n"
            "No other text."
        ),
        user_template=(
            "Task: {task}\n\n"
            "Upstream analyses:\n{inputs}\n\n"
            "No other text. Just output exactly one line: DECISION: <A|B|C|D|E>"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["finalize","verify"]
    ),
    "aqua_answerer": RoleProfile(
        name="aqua_answerer",
        description="Strict formatter for AQuA: extract letter from upstream and output one line.",
        system_template=(
            "You are a strict formatter.\n"
            "You will be given upstream messages that may contain:\n"
            "DECISION: X / RECOMMEND: X / CANDIDATE: X / Helper answer: X / Final answer: X.\n\n"
            "Rules (hard):\n"
            "1) If any 'DECISION: <A|B|C|D|E>' exists, output that letter.\n"
            "2) Else if any 'Final answer:' exists, output the last one.\n"
            "3) Else prefer last RECOMMEND, then last CANDIDATE, then last Helper answer.\n"
            "4) Output MUST be EXACTLY ONE LINE:\n"
            "Final answer: <A|B|C|D|E>\n"
            "No other text."
        ),
        user_template=(
            "Upstream messages:\n{inputs}\n"
        ),
        local_handler=_aqua_final_handler,
        temperature=0.0,
        capabilities=["finalize", "extract"],
    ),
    "nli_analyst": RoleProfile(
        name="nli_analyst",
        description="Analyze MNLI premise and hypothesis and output both reasoning and a final NLI label line.",
        system_template=(
            "You are an expert in natural language inference (NLI).\n"
            "You will be given a task that includes a 'premise' and a 'hypothesis'.\n"
            "You may also see upstream messages from OTHER helper agents.\n"
            "Use them only as optional hints – YOU are the final judge.\n"
            "Your job is to analyze their relationship and decide whether the\n"
            "hypothesis is ENTAILED by, CONTRADICTS, or is NEUTRAL with respect\n"
            "to the premise.\n\n"
            "You MUST do two things, in this order:\n"
            "1) Write 2–5 sentences of concise reasoning in natural language.\n"
            "2) On the LAST line, output EXACTLY one of the following:\n"
            "   Label: entailment\n"
            "   Label: neutral\n"
            "   Label: contradiction\n\n"
            "Rules:\n"
            "- Do NOT mention option letters A/B/C.\n"
            "- The last line MUST start with 'Label:' exactly, followed by one\n"
            "  of 'entailment', 'neutral', or 'contradiction'."
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream context (may be empty):\n{inputs}\n\n"
            "Analyze the relationship between the premise and the hypothesis.\n"
            "Follow the instructions strictly:\n"
            "1) 2–5 sentences of reasoning.\n"
            "2) Final line: 'Label: entailment' OR 'Label: neutral' OR 'Label: contradiction'."
        ),
        local_handler=None,
        temperature=0.1,
        capabilities=["analyze"],
    ),
    "nli_labeler": RoleProfile(
        name="nli_labeler",
        description="Read the NLI Label line from reasoning and map it to A/B/C.",
        system_template=(
            "You are a strict NLI label mapper.\n"
            "Upstream reasoning ends with a line like:\n"
            "  Label: entailment\n"
            "  Label: neutral\n"
            "  Label: contradiction\n\n"
            "Your ONLY job is to:\n"
            "1) Find the LAST line that starts with 'Label:'.\n"
            "2) Read its value (entailment / neutral / contradiction).\n"
            "3) Map it to a letter:\n"
            "   entailment -> A\n"
            "   neutral    -> B\n"
            "   contradiction -> C\n"
            "4) Output EXACTLY one line:\n"
            "   Final answer: <LETTER>\n"
            "   where <LETTER> is A, B, or C.\n\n"
            "IMPORTANT:\n"
            "- Do NOT re-interpret the premise and hypothesis yourself.\n"
            "- TRUST the label in the reasoning.\n"
            "- Do NOT output explanations or the words 'entailment', 'neutral', etc.\n"
            "- Output only 'Final answer: A/B/C'."
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream reasoning (including a 'Label:' line):\n{inputs}\n\n"
            "Follow the mapping rules and output exactly one line:\n"
            "Final answer: <LETTER>\n"
            "where <LETTER> is A, B, or C."
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["classify"],
    ),
    "mnli_direct_answerer": RoleProfile(
        name="mnli_direct_answerer",
        description="Given MNLI task and prior reasoning, choose entailment, neutral, or contradiction (A/B/C).",
        system_template=(
            "You are an NLI classifier that maps a premise, a hypothesis, and prior\n"
            "reasoning to a discrete label.\n"
            "You will see:\n"
            "- a task dict with 'premise', 'hypothesis', and 'choices', where\n"
            "  choices = ['entailment', 'neutral', 'contradiction']\n"
            "- upstream reasoning about the NLI relation.\n\n"
            "Your job is to choose exactly one label among:\n"
            "A: entailment\n"
            "B: neutral\n"
            "C: contradiction\n\n"
            "Rules:\n"
            "1) Use both the reasoning and the original premise/hypothesis.\n"
            "2) Ignore any 'answer' field in the task; it is the gold label for eval.\n"
            "3) Output exactly one line in the form:\n"
            "   Final answer: <LETTER>\n"
            "   where <LETTER> is A, B, or C.\n"
            "4) Do NOT output explanations or option text, only that single line.\n"
        ),
        user_template=(
            "Task (JSON-like): {task}\n"
            "Upstream context (reasoning):\n{inputs}\n\n"
            "Now decide the label and output exactly one line:\n"
            "Final answer: <LETTER>\n"
            "where <LETTER> is A, B, or C.\n"
        ),
        local_handler=None,
        temperature=0.0,
        capabilities=["classify"],
    ),
}
