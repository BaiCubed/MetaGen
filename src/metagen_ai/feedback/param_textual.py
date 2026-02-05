                                          
from __future__ import annotations
from typing import Dict, Any, Optional
import re

from metagen_ai.roles.parametric import ParamManager

def _success(summary: str) -> bool:
    s = (summary or "").lower()
    return ("verified" in s) or (re.search(r"\bfinal answer:\b", s) is not None)

def build_parametric_hooks(program, pm: ParamManager):
    """
    Returns a pair of hooks:
      - before_node: applies ParamManager to mutate RoleProfile in-place
      - textual_gradient_hook: updates temperatures & prompt hints from success/failure signal
    """
    default_temp = program.default_temperature

    def before_node(node_id: str, ctx: Dict[str, Any]) -> None:
        role_name = program.G.nodes[node_id].get("role", "")
        role = program.role_library.get(role_name)
        if role is None:
            return
        pm.apply_before_node(role, role_name, default_temperature=default_temp)

    def textual_gradient_hook(payload: Dict[str, Any]) -> None:
                                                                
        summary = payload.get("summary", "")
        ok = _success(summary)

                            
        judge = "evaluator" if "evaluator" in program.role_library else "judge"
        solvers = [r for r in ("math_simplifier", "calculator") if r in program.role_library]

        temp_grads: Dict[str, float] = {}

        if ok:
                                                                                    
            temp_grads[judge] = +1.0                                                             
            for s in solvers:
                temp_grads[s] = +0.5
                                                                         
            pm.add_hint(judge, "concise")
        else:
                                                                                                    
            temp_grads[judge] = -0.5
            for s in solvers:
                temp_grads[s] = -1.0
                                                                      
            for s in solvers:
                pm.add_hint(s, "cot_short")
                pm.add_hint(s, "verify")
            pm.add_hint(judge, "concise")

        pm.step_temperature(temp_grads)

                                                     
    class _Hooks:
        pass
    h = _Hooks()
    h.before_node = before_node
    h.textual_gradient_hook = textual_gradient_hook
    return h
