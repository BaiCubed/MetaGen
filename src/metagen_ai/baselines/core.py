                                  
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import re, time, math, random
from collections import Counter
from metagen_ai.utils.llm import LLMClient


                          
       
                          

_ANS_RE = re.compile(
    r"####\s*([-+]?\d+(?:\.\d+)?)|Final answer:\s*([-+]?\d+(?:\.\d+)?)",
    re.I,
)


def _extract_num(s: str) -> Optional[str]:
    m = _ANS_RE.search(s)
    if m:
        return (m.group(1) or m.group(2)).strip()
                
    m2 = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return m2[-1] if m2 else None


def _is_code_task(task: Dict[str, Any]) -> bool:
    """
    判断当前 task 是否是 HumanEval 这类“代码问题”，而不是 short-answer numeric 这种数值题。
    特征和 datasets/loader 中 HumanEval 的格式对齐：
      - 通常含有 entry_point / prompt / canonical_solution / tests 字段。
    """
    if not isinstance(task, dict):
        return False
    if "entry_point" in task or "canonical_solution" in task:
        return True
    if "tests" in task:
        return True
    return False


def _is_mc_task(task: Dict[str, Any]) -> bool:
    """
    判断是不是多选题（MMLU / MNLI 这种 choices 列表形式）。
    """
    return (
        isinstance(task, dict)
        and isinstance(task.get("choices"), list)
        and len(task.get("choices")) > 0
    )


def _build_mnli_question(task: Dict[str, Any]) -> str:
    """
    构造“论文里常用风格”的 MNLI 题面：
      Premise / Hypothesis + 三个标签 + 只输出 A/B/C。
    不改别的 dataset。
    """
    premise = task.get("premise", "")
    hypothesis = task.get("hypothesis", "")

    return (
        "You are solving a natural language inference (NLI) problem.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Decide the relationship between the hypothesis and the premise.\n"
        "Options:\n"
        "A. entailment\n"
        "B. neutral\n"
        "C. contradiction\n\n"
        "Answer with exactly one line: Final answer: <LETTER>\n"
        "where <LETTER> is A, B, or C."
    )


                          
                      
                          

def _prompt_cot_math(question: str) -> List[Dict[str, str]]:
    sys = (
        "You are a careful math tutor. Think step by step, then give the final "
        "numeric answer as: Final answer: <number>."
    )
    usr = (
        "Solve the problem step by step, then end with one line "
        "'Final answer: <number>'.\n"
        f"Problem: {question}"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


def _prompt_cot_mc(question: str) -> str:
    """
    多选题（包括 MNLI / MMLU）的 CoT prompt。
    这里假设 question 里已经包含了题干、选项和
    “Final answer: <LETTER>” 的格式要求。
    """
    return (
        "You are an expert at multiple-choice reasoning (including NLI tasks).\n"
        "Carefully analyze the problem and think step by step before deciding.\n\n"
        f"{question}\n\n"
        "Think step by step and then give the final answer in the form:\n"
        "Final answer: <LETTER>\n"
    )


                          
                               
                          

def _prompt_code_direct(task: Dict[str, Any], with_cot: bool = False) -> List[Dict[str, str]]:
    """
    HumanEval / 代码任务 baseline 的统一 prompt。

    with_cot = False: 直接“写函数”版（类似社区默认 prompt）
    with_cot = True : CoT 风格，要求在代码里体现思考（注释），但仍然只输出代码块。
    """
    prompt = str(task.get("prompt") or task.get("docstring") or task)
    entry = task.get("entry_point", "solution")

    if with_cot:
        sys = (
            "You are a senior Python engineer.\n"
            "Think carefully about the problem and possible corner cases.\n"
            "Then output ONLY a Python code block with the final implementation.\n"
            "You may include comments inside the code to reflect your reasoning,\n"
            "but do not write any text outside the code block."
        )
    else:
        sys = (
            "You are a senior Python engineer.\n"
            "Return ONLY a Python code block implementing the requested function.\n"
            "Do not include any explanations or reasoning outside the code block."
        )

    usr = (
        "Implement the function described below.\n\n"
        f"Function name: {entry}\n\n"
        "Problem description:\n"
        f"{prompt}\n\n"
        "Constraints:\n"
        "- Keep the SAME function name and parameters as given.\n"
        "- Use pure Python; do not import external packages.\n"
        "- Include any necessary helper functions in the same file.\n"
        "- Make sure the implementation is correct and handles edge cases.\n\n"
        "Your reply MUST be a single Python code block starting with ```python.\n"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]


                                                       
                                      
                                                       

def run_cot(
    llm: LLMClient,
    task: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """
    Chain-of-Thought baseline.
    - 代码任务（HumanEval）：写代码，允许在代码里体现思考。
    - 数值任务（short-answer numeric）：逐步推理，最后给数字答案。
    - 多选题（MNLI / MMLU）：逐步推理，最后给字母答案。
    """
    t0 = time.time()
    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)

    question = task.get("question") or task.get("prompt") or ""
    usage = 0

                                   
    if is_code:
        sys_msg = (
            "You are an expert Python programmer. Write correct, efficient, and readable code.\n"
            "Follow the function signature and pass all tests."
        )
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": question},
        ]
        resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        usage += resp.get("usage", {}).get("total_tokens", 0)
        final = resp.get("text", "")
        dt = time.time() - t0
        return final, {"total_tokens": usage}, dt

                                           
    if is_mc:
                                                     
        dataset_name = str(task.get("dataset") or "").lower()
        if dataset_name == "mnli":
            q_text = _build_mnli_question(task)
        else:
            q_text = question

        prompt = _prompt_cot_mc(q_text)
        msgs = [{"role": "user", "content": prompt}]
        resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        usage += resp.get("usage", {}).get("total_tokens", 0)
        txt = resp.get("text", "")

                                   
        letter = _extract_letter(txt)
        if letter is None:
                                     
            m = re.findall(r"[A-Z]", txt)
            letter = m[-1] if m else txt.strip()

        final = letter                        
        dt = time.time() - t0
        return final, {"total_tokens": usage}, dt

                                  
    msgs = _prompt_cot_math(question)
    resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
    usage += resp.get("usage", {}).get("total_tokens", 0)
    final = resp.get("text", "")
    dt = time.time() - t0
    return final, {"total_tokens": usage}, dt


                                                       
                                      
                                                       

def _extract_letter(ans: str) -> Optional[str]:
    m = re.search(r"Final answer\s*:\s*([A-Z])", ans)
    if not m:
        return None
    return m.group(1).strip().upper()


def run_self_consistency(
    llm: LLMClient,
    task: Dict[str, Any],
    k: int = 5,
    temperature: float = 0.5,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """
    Self-consistency CoT:
    - 代码任务：采样多份代码，简单返回第一份。
    - 数值任务：对 numeric 答案投票。
    - 多选题：对 LETTER 投票。
    """
    t0 = time.time()
    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)

    question = task.get("question") or task.get("prompt") or ""
    usage = 0

                                   
    if is_code:
        sys_msg = (
            "You are an expert Python programmer. Write correct, efficient, and readable code."
        )
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": question},
        ]
        samples = []
        for _ in range(k):
            resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            usage += resp.get("usage", {}).get("total_tokens", 0)
            samples.append(resp.get("text", ""))
        final = samples[0] if samples else ""
        dt = time.time() - t0
        return final, {"total_tokens": usage}, dt

                                               
    if is_mc:
        dataset_name = str(task.get("dataset") or "").lower()
        if dataset_name == "mnli":
            q_text = _build_mnli_question(task)
        else:
            q_text = question

        prompt = _prompt_cot_mc(q_text)
        msgs = [{"role": "user", "content": prompt}]
        letters: List[str] = []
        raw_answers: List[str] = []

        for _ in range(k):
            resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            usage += resp.get("usage", {}).get("total_tokens", 0)
            txt = resp.get("text", "")
            raw_answers.append(txt)
            letter = _extract_letter(txt)
            if letter:
                letters.append(letter)

        if letters:
            cnt = Counter(letters)
            best_letter, _ = cnt.most_common(1)[0]
            final = best_letter                    
        else:
                               
            fallback = raw_answers[-1] if raw_answers else ""
            final = _extract_letter(fallback) or fallback.strip()

        dt = time.time() - t0
        return final, {"total_tokens": usage}, dt

                                  
    msgs = _prompt_cot_math(question)
    answers: List[str] = []
    raw_answers: List[str] = []

    for _ in range(k):
        resp = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        usage += resp.get("usage", {}).get("total_tokens", 0)
        txt = resp.get("text", "")
        raw_answers.append(txt)
        num = _extract_num(txt)
        if num is not None:
            answers.append(num)

    if answers:
        cnt = Counter(answers)
        best_num, _ = cnt.most_common(1)[0]
        final = f"Final answer: {best_num}"
    else:
        final = raw_answers[-1] if raw_answers else ""

    dt = time.time() - t0
    return final, {"total_tokens": usage}, dt


                                                       
                            
                                                       

def run_debate(
    llm: LLMClient,
    task: Dict[str, Any],
    rounds: int = 2,
    temperature: float = 0.5,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """
    - 代码任务（HumanEval）：两位“程序员”多轮互看代码并改进，最后由评审挑选。
    - 多选题（MNLI / MMLU）：两位 debater 输出带 CoT 的 LETTER 答案，最后由裁判选一个字母。
    - 数值任务：数值版 A/B 辩论 + 裁判。
    """
    t0 = time.time()
    rounds = max(1, int(rounds))
    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)
    usage = 0

                                   
    if is_code:
        prompt = str(task.get("prompt") or task.get("docstring") or task)
        entry = task.get("entry_point", "solution")

        def _debater_prompt(side: str, opponent_code: Optional[str]) -> List[Dict[str, str]]:
            if opponent_code is None:
                sys = (
                    f"You are Debater {side}, a strong Python engineer.\n"
                    "Write a complete implementation as a Python code block.\n"
                    "No text outside the code block."
                )
                usr = (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    "Return ONLY one Python code block with your best implementation."
                )
            else:
                sys = (
                    f"You are Debater {side}, a strong Python engineer.\n"
                    "You will see the problem and your opponent's current code.\n"
                    "Critically think about possible bugs or weaknesses and then "
                    "write an improved implementation.\n"
                    "Return ONLY one Python code block."
                )
                usr = (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"Opponent's current code:\n{opponent_code}\n\n"
                    "Now produce your improved implementation as a single Python code block."
                )
            return [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]

                 
        rA = llm.chat(_debater_prompt("A", None), temperature=temperature, max_tokens=max_tokens)
        rB = llm.chat(_debater_prompt("B", None), temperature=temperature, max_tokens=max_tokens)
        codeA, codeB = rA.get("text", ""), rB.get("text", "")
        usage += rA.get("usage", {}).get("total_tokens", 0)
        usage += rB.get("usage", {}).get("total_tokens", 0)

                       
        for _ in range(max(0, rounds - 1)):
            rA = llm.chat(
                _debater_prompt("A", opponent_code=codeB),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            rB = llm.chat(
                _debater_prompt("B", opponent_code=codeA),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            codeA, codeB = rA.get("text", ""), rB.get("text", "")
            usage += rA.get("usage", {}).get("total_tokens", 0)
            usage += rB.get("usage", {}).get("total_tokens", 0)

              
        judge_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a strict Python code reviewer.\n"
                    "You will see a problem statement and two candidate implementations "
                    "A and B. Choose the candidate that is more likely to be correct, "
                    "robust, and efficient.\n"
                    "Respond with exactly one line: CHOSEN: A or CHOSEN: B."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"Candidate A:\n{codeA}\n\n"
                    f"Candidate B:\n{codeB}\n"
                ),
            },
        ]
        jr = llm.chat(judge_msgs, temperature=0.0, max_tokens=32)
        usage += jr.get("usage", {}).get("total_tokens", 0)

        chosen = "A"
        m = re.search(r"CHOSEN\s*:\s*([AB])", jr.get("text", ""))
        if m:
            chosen = m.group(1).upper()
        final_code = codeA if chosen == "A" else codeB

        dt = time.time() - t0
        return final_code, {"total_tokens": usage}, dt

                                               
    if is_mc:
        dataset_name = str(task.get("dataset") or "").lower()
        if dataset_name == "mnli":
            q = _build_mnli_question(task)
        else:
            q = task.get("question") or ""

        debaters: List[str] = []

            
        for side in ("A", "B"):
            sys = (
                f"You are Debater {side}. Reason step by step about this multiple-choice question.\n"
                "End with exactly one line: 'Final answer: <LETTER>'."
            )
            usr = q
            r = llm.chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage += r.get("usage", {}).get("total_tokens", 0)
            debaters.append(r.get("text", ""))

                       
        for _ in range(max(0, rounds - 1)):
            new_debaters = []
            for side_idx, side in enumerate(("A", "B")):
                opponent = debaters[1 - side_idx]
                sys = (
                    "You critique briefly, then refine your multiple-choice reasoning.\n"
                    "End with exactly one line: 'Final answer: <LETTER>'."
                )
                usr = (
                    f"Question:\n{q}\n\n"
                    f"Opponent's reasoning:\n{opponent}\n\n"
                    "Now produce your improved reasoning and final answer."
                )
                r = llm.chat(
                    [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage += r.get("usage", {}).get("total_tokens", 0)
                new_debaters.append(r.get("text", ""))
            debaters = new_debaters

                  
        judge = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a fair judge. Read both solutions and choose ONE option.\n"
                        "Output exactly one line: 'Final answer: <LETTER>'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{q}\n\n"
                        f"Solution A:\n{debaters[0]}\n\n"
                        f"Solution B:\n{debaters[1]}\n"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=128,
        )
        usage += judge.get("usage", {}).get("total_tokens", 0)
        letter = (
            _extract_letter(judge.get("text", ""))
            or _extract_letter(debaters[0])
            or _extract_letter(debaters[1])
            or "A"
        )
        final = letter
        dt = time.time() - t0
        return final, {"total_tokens": usage}, dt

                                   
    q = task["question"]
    debaters: List[str] = []

    for side in ("A", "B"):
        sys = (
            f"You are Debater {side}. Reason step by step and propose a numeric "
            "answer for the problem."
        )
        usr = f"Problem: {q}\nEnd with 'Final answer: <number>'."
        r = llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        debaters.append(r.get("text", ""))
        usage += r.get("usage", {}).get("total_tokens", 0)

    for _ in range(max(0, rounds - 1)):
        na = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You critique briefly, then refine. "
                        "End with 'Final answer: <number>'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Problem: {q}\nOpponent's reasoning:\n{debaters[1]}\n"
                        "Now produce your improved reasoning and final answer."
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        nb = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You critique briefly, then refine. "
                        "End with 'Final answer: <number>'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Problem: {q}\nOpponent's reasoning:\n{debaters[0]}\n"
                        "Now produce your improved reasoning and final answer."
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        debaters = [na.get("text", ""), nb.get("text", "")]
        usage += na.get("usage", {}).get("total_tokens", 0)
        usage += nb.get("usage", {}).get("total_tokens", 0)

    judge = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a fair judge. Read both solutions and choose ONE "
                    "numeric answer. Output exactly one line 'Final answer: <number>'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {q}\nSolution A:\n{debaters[0]}\n"
                    f"Solution B:\n{debaters[1]}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )
    usage += judge.get("usage", {}).get("total_tokens", 0)
    dt = time.time() - t0
    ans = (
        _extract_num(judge.get("text", ""))
        or _extract_num(debaters[0])
        or _extract_num(debaters[1])
        or ""
    )
    return f"Final answer: {ans}", {"total_tokens": usage}, dt


                                                       
                         
                                                       

def run_tot(
    llm: LLMClient,
    task: Dict[str, Any],
    breadth: int = 3,
    depth: int = 2,
    temperature: float = 0.5,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, int], float]:
    """
    - 代码任务：在“算法计划空间”上做 Tree-of-Thought，最后根据最佳 plan 写代码。
    - 多选题：在“推理 step”空间上做小 ToT，最后从最 promising path 中提取 LETTER。
    - 数值任务：简化版 ToT，最后从最 promising path 中提取数字。
    """
    t0 = time.time()
    breadth = max(1, int(breadth))
    depth = max(1, int(depth))

    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)
    usage = 0

                                   
    if is_code:
        prompt = str(task.get("prompt") or task.get("docstring") or task)
        entry = task.get("entry_point", "solution")

        def _plan_prompt(plan_prefix: Optional[str]) -> List[Dict[str, str]]:
            if not plan_prefix:
                sys = (
                    "You are an algorithm designer. Propose a high-level plan "
                    "for implementing a Python function to solve the problem."
                )
                usr = (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    "Write a concise step-by-step plan (no code). "
                    "Focus on correctness and clarity."
                )
            else:
                sys = (
                    "You are an algorithm designer. Improve or extend the given "
                    "high-level plan for implementing a Python function."
                )
                usr = (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"Current plan:\n{plan_prefix}\n\n"
                    "Refine or extend this plan to make it more robust and clear. "
                    "Do NOT write any code, only a step-by-step plan."
                )
            return [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]

        def _score_plan(plan: str) -> Tuple[float, int]:
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict reviewer of algorithmic plans.\n"
                        "Score from 0 to 10 how promising this plan is for producing "
                        "a correct and robust Python implementation.\n"
                        "Output only a number."
                    ),
                },
                {"role": "user", "content": plan},
            ]
            r = llm.chat(msgs, temperature=0.0, max_tokens=16)
            txt = r.get("text", "")
            tok = r.get("usage", {}).get("total_tokens", 0)
            try:
                score = float(re.findall(r"[-+]?\d+(?:\.\d+)?", txt)[0])
            except Exception:
                score = 5.0
            return score, tok

        frontier: List[str] = [""]

        best_plan = ""
        best_score = -1e9

        for _d in range(depth):
            new_frontier: List[Tuple[str, float]] = []
            for plan in frontier:
                for _ in range(breadth):
                    pr = llm.chat(_plan_prompt(plan), temperature=temperature, max_tokens=max_tokens)
                    plan_text = pr.get("text", "")
                    usage += pr.get("usage", {}).get("total_tokens", 0)
                    score, tok = _score_plan(plan_text)
                    usage += tok
                    new_frontier.append((plan_text, score))
                    if score > best_score:
                        best_score = score
                        best_plan = plan_text
            if not new_frontier:
                break
            new_frontier.sort(key=lambda x: x[1], reverse=True)
            frontier = [p for (p, s) in new_frontier[:1]]

        if not best_plan and frontier:
            best_plan = frontier[0]

        impl_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a senior Python engineer.\n"
                    "Given a problem and a high-level algorithmic plan, "
                    "write the final implementation as a single Python code block.\n"
                    "Do not output any text outside the code block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"High-level plan:\n{best_plan}\n\n"
                    "Now implement this plan as a Python function. "
                    "Your reply MUST be a single Python code block."
                ),
            },
        ]
        rc = llm.chat(impl_msgs, temperature=temperature, max_tokens=max_tokens)
        usage += rc.get("usage", {}).get("total_tokens", 0)
        dt = time.time() - t0
        return rc.get("text", ""), {"total_tokens": usage}, dt

                                  
    if is_mc:
        dataset_name = str(task.get("dataset") or "").lower()
        if dataset_name == "mnli":
            q = _build_mnli_question(task)
        else:
            q = task.get("question") or ""

        frontier = [""]

        best_path, best_score, best_letter = "", -1e9, ""

        for _d in range(depth):
            new_frontier: List[Tuple[str, float, str]] = []
            for path in frontier:
                for _ in range(breadth):
                    r = llm.chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Generate one brief reasoning step continuing the "
                                    "solution for this multiple-choice question.\n"
                                    "Then optionally propose 'Final answer: <LETTER>' if you are confident."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Question:\n{q}\n\n"
                                    f"Reasoning so far:\n{path}\n"
                                    "Continue with one short step."
                                ),
                            },
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    usage += r.get("usage", {}).get("total_tokens", 0)
                    step = r.get("text", "")

                    s = llm.chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Score 0-10: how promising is the above step "
                                    "toward a correct LETTER answer? Output only a number."
                                ),
                            },
                            {"role": "user", "content": step},
                        ],
                        temperature=0.0,
                        max_tokens=8,
                    )
                    usage += s.get("usage", {}).get("total_tokens", 0)
                    try:
                        score = float(re.findall(r"[-+]?\d+(?:\.\d+)?", s.get("text", ""))[0])
                    except Exception:
                        score = 5.0

                    cand_letter = _extract_letter(step) or ""
                    new_frontier.append((path + "\n" + step, score, cand_letter))

            new_frontier.sort(key=lambda x: x[1], reverse=True)
            if new_frontier:
                path, score, cand = new_frontier[0]
                frontier = [path]
                if score > best_score and cand:
                    best_score, best_path, best_letter = score, path, cand

        dt = time.time() - t0
        final_letter = best_letter or "A"
        final = final_letter
        return final, {"total_tokens": usage}, dt

                                   
    q = task["question"]
    usage = 0

    frontier = [""]

    best_path, best_score, best_ans = "", -1e9, ""

    for _d in range(depth):
        new_frontier: List[Tuple[str, float, str]] = []
        for path in frontier:
            for _ in range(breadth):
                r = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Generate one brief reasoning step continuing the "
                                "solution. Then propose 'Final answer: <number>' "
                                "if you are confident."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Problem: {q}\nContext so far:\n{path}\n"
                                "Continue with one short step."
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage += r.get("usage", {}).get("total_tokens", 0)
                step = r.get("text", "")

                s = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Score 0-10: how promising is the above step "
                                "toward a correct numeric answer? Output only a number."
                            ),
                        },
                        {"role": "user", "content": step},
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                usage += s.get("usage", {}).get("total_tokens", 0)
                try:
                    score = float(re.findall(r"[-+]?\d+(?:\.\d+)?", s.get("text", ""))[0])
                except Exception:
                    score = 5.0

                cand = _extract_num(step) or ""
                new_frontier.append((path + "\n" + step, score, cand))

        new_frontier.sort(key=lambda x: x[1], reverse=True)
        if new_frontier:
            path, score, cand = new_frontier[0]
            frontier = [path]
            if score > best_score and cand:
                best_score, best_path, best_ans = score, path, cand

    dt = time.time() - t0
    final = best_ans or ""
    return f"Final answer: {final}", {"total_tokens": usage}, dt


                                                       
                               
                                                       

def run_star_lite(
    llm: LLMClient,
    task: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int], float]:
    """
    - 代码任务：Student-Teacher-Student 的代码版。
    - 多选题：Student-Teacher-Student 的 LETTER 版。
    - 数值任务：原数值版 STAR-lite。
    """
    t0 = time.time()
    is_code = _is_code_task(task)
    is_mc = _is_mc_task(task)
    usage = 0

                                   
    if is_code:
        prompt = str(task.get("prompt") or task.get("docstring") or task)
        entry = task.get("entry_point", "solution")

             
        s1_msgs = [
            {
                "role": "system",
                "content": (
                    "You are Student 1, a junior Python programmer.\n"
                    "Write a first attempt implementation as a Python code block.\n"
                    "It may contain mistakes, but try your best."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    "Return ONLY a Python code block with your implementation."
                ),
            },
        ]
        r1 = llm.chat(s1_msgs, temperature=max(temperature, 0.4), max_tokens=max_tokens)
        code1 = r1.get("text", "")
        usage += r1.get("usage", {}).get("total_tokens", 0)

            
        t_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a teacher and expert Python engineer.\n"
                    "Review the student's code, point out possible bugs, missing "
                    "corner cases, and style issues.\n"
                    "Suggest how to fix them, but do NOT write full code."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"Student 1's code:\n{code1}\n\n"
                    "Please write a review and precise suggestions for improvement."
                ),
            },
        ]
        rt = llm.chat(t_msgs, temperature=0.2, max_tokens=max_tokens)
        review = rt.get("text", "")
        usage += rt.get("usage", {}).get("total_tokens", 0)

             
        s2_msgs = [
            {
                "role": "system",
                "content": (
                    "You are Student 2, a strong Python programmer.\n"
                    "You will see the problem, the original code, and the teacher's "
                    "review. Write an improved implementation as a single Python "
                    "code block.\n"
                    "Do not output any text outside the code block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem description:\n{prompt}\n\n"
                    f"Function name: {entry}\n\n"
                    f"Student 1's code:\n{code1}\n\n"
                    f"Teacher's review:\n{review}\n\n"
                    "Now write the improved implementation as a Python code block."
                ),
            },
        ]
        r2 = llm.chat(s2_msgs, temperature=temperature, max_tokens=max_tokens)
        code2 = r2.get("text", "")
        usage += r2.get("usage", {}).get("total_tokens", 0)

        dt = time.time() - t0
        return code2, {"total_tokens": usage}, dt

    q_raw = task.get("question") or ""

                                  
    if is_mc:
        dataset_name = str(task.get("dataset") or "").lower()
        if dataset_name == "mnli":
            q = _build_mnli_question(task)
        else:
            q = q_raw

        student1 = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a student. Solve the multiple-choice problem step by step, "
                        "and end with exactly one line 'Final answer: <LETTER>'."
                    ),
                },
                {"role": "user", "content": q},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage += student1.get("usage", {}).get("total_tokens", 0)

        teacher = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a teacher. Briefly point out mistakes or improvements "
                        "in the student's reasoning. Suggest the correct LETTER answer if possible."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{q}\n\nStudent solution:\n{student1.get('text', '')}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=384,
        )
        usage += teacher.get("usage", {}).get("total_tokens", 0)

        student2 = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite a clean, corrected solution, and end with exactly one "
                        "line 'Final answer: <LETTER>'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{q}\n\nTeacher feedback:\n{teacher.get('text', '')}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        usage += student2.get("usage", {}).get("total_tokens", 0)

        dt = time.time() - t0
        letter = (
            _extract_letter(student2.get("text", ""))
            or _extract_letter(teacher.get("text", ""))
            or _extract_letter(student1.get("text", ""))
            or "A"
        )
        final = letter
        return final, {"total_tokens": usage}, dt

                                   
    q = q_raw

    student1 = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a student. Solve step by step, end with "
                    "'Final answer: <number>'."
                ),
            },
            {"role": "user", "content": q},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage += student1.get("usage", {}).get("total_tokens", 0)

    teacher = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a teacher. Briefly point out mistakes or improvements. "
                    "Suggest the correct final numeric answer if possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {q}\nStudent solution:\n{student1.get('text', '')}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=384,
    )
    usage += teacher.get("usage", {}).get("total_tokens", 0)

    student2 = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite a clean, corrected solution, and end with exactly one "
                    "line 'Final answer: <number>'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {q}\nTeacher feedback:\n{teacher.get('text', '')}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    usage += student2.get("usage", {}).get("total_tokens", 0)

    dt = time.time() - t0
    ans = (
        _extract_num(student2.get("text", ""))
        or _extract_num(teacher.get("text", ""))
        or _extract_num(student1.get("text", ""))
        or ""
    )
    final = f"Final answer: {ans}"
    return final, {"total_tokens": usage}, dt
