                                     
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import math, random, copy

from metagen_ai.graph_ops.softmask import SoftMaskManager, SoftMaskConfig
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.feedback.param_textual import build_parametric_hooks
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.utils.metrics import accuracy_from_final, cost_from_usage

@dataclass
class ESConfig:
    sigma: float = 0.5                  
    lr: float = 0.2                       
    lam_cost: float = 1e-3                             
    population: int = 8                                                                 
    seed: int = 42

def _pack_theta(smm: SoftMaskManager, pm: ParamManager) -> List[float]:
    vec = []
                                          
    for n in sorted(smm.state.node_alpha.keys()):
        vec.append(smm.state.node_alpha[n])
                 
    for e in sorted(smm.state.edge_alpha.keys()):
        vec.append(smm.state.edge_alpha[e])
                                              
    for r in sorted(pm.temp_delta.keys()):
        vec.append(pm.temp_delta[r])
    return vec

def _unpack_theta(theta: List[float], smm: SoftMaskManager, pm: ParamManager) -> None:
    i = 0
    for n in sorted(smm.state.node_alpha.keys()):
        smm.state.node_alpha[n] = theta[i]; i+=1
    for e in sorted(smm.state.edge_alpha.keys()):
        smm.state.edge_alpha[e] = theta[i]; i+=1
    for r in sorted(pm.temp_delta.keys()):
        pm.temp_delta[r] = theta[i]; i+=1

def _eval_once(task: Dict[str, Any], cfg: Dict[str, Any], smm: SoftMaskManager, pm: ParamManager) -> Tuple[float, Dict[str, Any]]:
                                                                
    program = smm.program
    smm.forward(write_active=True)
    hooks = build_parametric_hooks(program, pm)
    res = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
    acc = accuracy_from_final(res["final"], task.get("answer", ""))
    tokens = float(res["usage"].get("total_tokens", 0))
    reward = float(acc) - ESConfig().lam_cost * tokens
    return reward, res

def es_step(task: Dict[str, Any], cfg: Dict[str, Any], es: ESConfig) -> Dict[str, Any]:
    """
    One ES update step on a single task:
      - Build program and initial soft parameters (smm, pm),
      - Evaluate antithetic noise samples,
      - Update parameters, return diagnostics.
    """
    rnd = random.Random(es.seed)

                      
    program = build_task_graph(task, cfg)
    program = sample_architecture_costaware(program, task, cfg)
    if (cfg.get("pruning", {}) or {}).get("enable", True):
        program = prune_once(program, task, cfg)

                    
    smm = SoftMaskManager(program, SoftMaskConfig(tau_node=0.8, tau_edge=0.8, lr_node=0.0, lr_edge=0.0))
    pm = ParamManager()
                                                     
    for _, d in program.G.nodes(data=True):
        r = d.get("role", "")
        if r and r not in pm.temp_delta:
            pm.temp_delta[r] = 0.0

                             
    theta0 = _pack_theta(smm, pm)

                            
    grads = [0.0] * len(theta0)
    m = es.population // 2
    for k in range(m):
        eps = [rnd.gauss(0.0, es.sigma) for _ in theta0]

                  
        theta_pos = [t + e for t, e in zip(theta0, eps)]
        _unpack_theta(theta_pos, smm, pm)
        r_pos, _ = _eval_once(task, cfg, smm, pm)

                  
        theta_neg = [t - e for t, e in zip(theta0, eps)]
        _unpack_theta(theta_neg, smm, pm)
        r_neg, _ = _eval_once(task, cfg, smm, pm)

                           
        scale = (r_pos - r_neg) / (2.0 * (es.sigma**2))
        for i in range(len(grads)):
            grads[i] += scale * eps[i]

    grads = [g / max(1, m) for g in grads]

                          
    theta_new = [t + es.lr * g for t, g in zip(theta0, grads)]
    _unpack_theta(theta_new, smm, pm)

                                   
    r_final, res = _eval_once(task, cfg, smm, pm)
    return {
        "reward": r_final,
        "final": res["final"],
        "usage": res["usage"],
        "theta_dim": len(theta0),
        "active_counts": smm.counts(),
        "temp_deltas": dict(pm.temp_delta),
    }
