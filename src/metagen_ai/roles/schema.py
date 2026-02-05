from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import re

                          
        
                           
                             
                        
                  
   
LocalHandler = Callable[[Dict[str, Any]], str]


@dataclass
class RoleProfile:
    name: str
    description: str
    system_template: str
    user_template: str
    local_handler: Optional[LocalHandler] = None
    temperature: Optional[float] = None
                            
                                                               
    capabilities: List[str] = field(default_factory=list)


@dataclass
class NodeOutput:
    text: str
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }
    )


@dataclass
class RunTraceItem:
    node_id: str
    role: str
    prompt_preview: str
    output_preview: str
    usage: Dict[str, int]


@dataclass
class Hooks:
                                                    
    before_node: Optional[Callable[[str, Dict[str, Any]], None]] = None
    after_node: Optional[Callable[[str, NodeOutput], None]] = None
    textual_gradient_hook: Optional[Callable[[Dict[str, Any]], None]] = None


                  
_BRACKET_Q = re.compile(r"\{task\[['\"]?question['\"]?\]\}")
_BRACKET_A = re.compile(r"\{task\[['\"]?answer['\"]?\]\}")
_BRACKET_QUERY = re.compile(r"\{task\[['\"]?query['\"]?\]\}")
_BRACKET_PROMPT = re.compile(r"\{task\[['\"]?prompt['\"]?\]\}")


class _SafeDict(dict):
    """format_map 兜底：缺键给空串，避免 KeyError。"""
    def __missing__(self, key):
        return ""


def _sanitize_placeholders(tmpl: str) -> str:
    """
    把 {task[question]} / {task['question']} / {task["question"]} 等替换成 {question}，
    同理处理 answer / query / prompt。
    """
    if not isinstance(tmpl, str):
        return ""
    s = tmpl
    s = _BRACKET_Q.sub("{question}", s)
    s = _BRACKET_A.sub("{answer}", s)
    s = _BRACKET_QUERY.sub("{query}", s)
    s = _BRACKET_PROMPT.sub("{prompt}", s)
    return s


REDACT_KEYS = {
           
    "answer", "label", "gold", "target", "y", "output",

           
    "answers", "answer_text", "gold_answers", "references",

             
    "canonical_solution", "solution", "reference", "tests", "test",

                
    "ground_truth", "gt", "expected", "expected_output",
}

def redact_task_for_prompt(task: Any) -> Any:
    """给 LLM 的 task：去掉所有 gold/参考答案字段。"""
    if not isinstance(task, dict):
        return task
    t = dict(task)
    for k in list(t.keys()):
        if k in REDACT_KEYS:
            t.pop(k, None)
    return t


def render_messages(role: RoleProfile,
                    task: Dict[str, Any],
                    inputs: Dict[str, Any],
                    prev_summary: str):
    """
    将角色模板渲染为 OpenAI/DeepSeek 兼容的 messages 列表。
    支持占位符：
      {task} / {inputs} / {prev_summary}
      {question} / {answer} / {query} / {prompt}
    并自动把 {task[...]} 写法替换成扁平字段以避免 KeyError。
    """
    task_raw = task
    task_prompt = redact_task_for_prompt(task_raw)

           
    question = ""
    answer = ""
    query = ""
    prompt = ""
    choices = None
    choices_str = ""
    subject = ""

    premise = ""
    hypothesis = ""
    passage = ""

    if isinstance(task_raw, dict):
        question = str(task_raw.get("question", ""))
        prompt = str(task_raw.get("prompt", ""))
        answer = ""
        query = str(task_raw.get("query", ""))

        subject = str(task_raw.get("subject", ""))
        premise = str(task_raw.get("premise", ""))
        hypothesis = str(task_raw.get("hypothesis", ""))

        passage = str(task_raw.get("passage", task_raw.get("context", "")) or "")

        choices = task_raw.get("choices", None)
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            lines = []
            for i, c in enumerate(choices):
                if i >= len(letters):
                    break
                lines.append(f"{letters[i]}) {c}")
            choices_str = "\n".join(lines)
    else:
        question = str(task)

    ctx = _SafeDict({
        "task": task_prompt,                          
        "inputs": inputs,
        "prev_summary": prev_summary,

        "question": question,
        "answer": answer,               
        "query": query,
        "prompt": prompt,

        "choices": choices,
        "choices_str": choices_str,
        "subject": subject,

        "premise": premise,
        "hypothesis": hypothesis,
        "passage": passage,
    })

    sys_t = _sanitize_placeholders(getattr(role, "system_template", "") or "")
    usr_t = _sanitize_placeholders(getattr(role, "user_template", "") or "")

    try:
        sys = sys_t.format_map(ctx)
    except Exception:
        sys = sys_t

    try:
        usr = usr_t.format_map(ctx)
    except Exception:
        usr = usr_t

                                 
    if not sys and not usr:
        usr = f"Task: {question or prompt or task}\nInputs: {inputs}\nPrev: {prev_summary}"

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]
