from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx

from metagen_ai.roles.schema import RoleProfile, NodeOutput, RunTraceItem, Hooks, render_messages, redact_task_for_prompt
from metagen_ai.utils.llm import LLMClient

@dataclass
class GraphProgram:
    """
    Executable multi-agent program represented as a DAG.
    Node attributes expected:
        - role: str (must exist in role_library)
        - active: bool (optional, default True)    # for pruning/gating
    Edge attributes (u, v):
        - active: bool (optional, default True)    # for pruning/gating
        - kind: "space" | "time" (default "space")
    """
    G: nx.DiGraph
    role_library: Dict[str, RoleProfile]
    llm: Optional[LLMClient] = None
    default_temperature: float = 0.2

    def __post_init__(self):
        if not nx.is_directed_acyclic_graph(self.G):
            raise ValueError("GraphProgram requires a DAG.")
                             
        for n, data in self.G.nodes(data=True):
            data.setdefault("active", True)
        for u, v, data in self.G.edges(data=True):
            data.setdefault("active", True)
            data.setdefault("kind", "space")

    def run(self,
            task: Dict[str, Any],
            rounds: int = 1,
            hooks: Optional[Hooks] = None,
            early_exit: bool = False) -> Dict[str, Any]:
        """
        Execute the DAG for a number of rounds. Space edges pass same-round messages;
        time edges bring summaries from the previous round.

        Returns:
            {
              "final": str,
              "node_outputs": Dict[str, str],
              "traces": List[RunTraceItem],
              "usage": Dict[str, int],
              "rounds": int
            }
        """
                                                     
        hooks = hooks or type("Hooks", (), {})()

        order = list(nx.topological_sort(self.G))
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        traces: List[RunTraceItem] = []
        last_round_summary = ""

        node_text: Dict[str, str] = {}

        for r in range(rounds):
            round_outputs: Dict[str, NodeOutput] = {}
            for node_id in order:
                nattr = self.G.nodes[node_id]
                if not nattr.get("active", True):
                    continue
                role_name = nattr.get("role", "")
                if role_name not in self.role_library:
                    raise KeyError(f"Role '{role_name}' not found in role_library.")
                role = self.role_library[role_name]

                                                                        
                inputs = {}
                for u in self.G.predecessors(node_id):
                    if not self.G.nodes[u].get("active", True):
                        continue
                    eattr = self.G.get_edge_data(u, node_id) or {}
                    if not eattr.get("active", True) or eattr.get("kind", "space") != "space":
                        continue
                    if u in round_outputs:
                        inputs[u] = round_outputs[u].text
                    else:
                                                                   
                        pass

                                               
                prev_summary = last_round_summary

                                                
                before = getattr(hooks, "before_node", None)
                if callable(before):
                    before(node_id, {"task": task, "inputs": inputs, "prev_summary": prev_summary})

                                                            
                if role.local_handler is not None:
                    text = role.local_handler({
                        "task": task, "inputs": inputs, "prev_summary": prev_summary, "node_id": node_id
                    })
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                else:
                    if self.llm is None:
                        raise RuntimeError(f"No LLM client configured and role '{role.name}' has no local handler.")
                    messages = render_messages(role, task, inputs, prev_summary)
                    temp = role.temperature if role.temperature is not None else self.default_temperature
                    resp = self.llm.chat(messages, temperature=temp)
                    text = resp["text"]
                    usage = resp.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

                out = NodeOutput(text=text, usage=usage)
                round_outputs[node_id] = out

                                  
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)

                                 
                prompt_preview = ""                              
                traces.append(RunTraceItem(
                    node_id=node_id,
                    role=role.name,
                    prompt_preview=prompt_preview,
                    output_preview=(text[:160] + ("..." if len(text) > 160 else "")),
                    usage=usage,
                ))

                                               
                after = getattr(hooks, "after_node", None)
                if callable(after):
                    after(node_id, out)

                                                                               
                if early_exit and nattr.get("is_exit", False):
                    break

                                                                           
            sinks = [n for n in order if self.G.out_degree(n) == 0 and self.G.nodes[n].get("active", True)]
            last_round_summary = "\n".join(round_outputs[s].text for s in sinks if s in round_outputs)
                                                
            node_text = {nid: ro.text for nid, ro in round_outputs.items()}

                                                                
            tgh = getattr(hooks, "textual_gradient_hook", None)
            if callable(tgh):
                tgh({
                    "task": redact_task_for_prompt(task),           
                    "round": r,
                    "node_outputs": node_text,
                    "summary": last_round_summary,
                })

                                                                            
        final_text = self._select_final(node_text, prefer="judge")
        return {
            "final": final_text,
            "node_outputs": node_text,
            "traces": traces,
            "usage": total_usage,
            "rounds": rounds,
        }

    def _select_final(self, node_text: Dict[str, str], prefer: str = "judge") -> str:
                       
        if prefer in self.G.nodes and prefer in node_text:
            return node_text[prefer]

                                    
        if "evaluator" in self.G.nodes and "evaluator" in node_text:
            return node_text["evaluator"]

                                       
        exits = [n for n, d in self.G.nodes(data=True) if d.get("is_exit")]
        for e in exits:
            if e in node_text:
                return node_text[e]

                              
        sinks = [n for n in self.G.nodes if self.G.out_degree(n) == 0]
        for s in sinks:
            if s in node_text:
                return node_text[s]

                          
        for n in reversed(list(self.G.nodes)):
            if n in node_text:
                return node_text[n]
        return ""

