                                           
from __future__ import annotations
import os, json, re, math, random
from typing import Dict, List, Tuple, Optional, Any, Iterable
from dataclasses import dataclass, field

                                      
     
                        
                                        
                                                                 

_NUM_RE = re.compile(r"[a-z0-9]+")

def _toks(s: str) -> List[str]:
    s = (s or "").lower()
    return _NUM_RE.findall(s)

def _bow_feats(prefix: str, text: str, limit: int = 64) -> List[str]:
    toks = _toks(text)
    if not toks:
        return []
            
    toks = toks[:limit]
    return [f"{prefix}:{t}" for t in toks]

@dataclass
class _Lin:
    w: Dict[str, float] = field(default_factory=dict)

    def score(self, feats: Iterable[str]) -> float:
        s = 0.0
        for f in feats:
            s += self.w.get(f, 0.0)
        return s

    def update(self, feats_pos: Iterable[str], feats_neg: Iterable[str], lr: float) -> None:
        for f in feats_pos:
            self.w[f] = self.w.get(f, 0.0) + lr
        for f in feats_neg:
            self.w[f] = self.w.get(f, 0.0) - lr

    def l2_clip(self, max_norm: float = 10.0) -> None:
                     
        import math
        n2 = sum(v*v for v in self.w.values())
        if n2 > max_norm * max_norm:
            scale = max_norm / math.sqrt(n2)
            for k in list(self.w.keys()):
                self.w[k] *= scale

class RoleEdgePolicy:
    """
    - select_programmers: 从候选 programmer_v* 中选 top-k
    - decide_edges: 决定哪些 programmer→code_evaluator 的兜底边 active（默认允许，被学习关停）
    - update_after_run: 根据 reward 在线更新
    持久化：cfg['policy']['path']（默认 logs/policy_humaneval.json）
    """
    def __init__(self, path: str, lr: float = 0.05, seed: int = 42, topk_programmers: int = 2,
                 edge_keep_ratio: float = 1.0):
        self.path = path
        self.lr = float(lr)
        self.topk_programmers = int(topk_programmers)
        self.edge_keep_ratio = float(edge_keep_ratio)                                       
        random.seed(seed)

        self.role_lin = _Lin()
        self.edge_lin = _Lin()
        self._last_ctx: Dict[str, Any] = {}                

        self._load()

                                       
    def _load(self):
        if not self.path or (not os.path.isfile(self.path)):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.role_lin.w = dict(obj.get("role_w", {}))
            self.edge_lin.w = dict(obj.get("edge_w", {}))
        except Exception:
            pass

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"role_w": self.role_lin.w, "edge_w": self.edge_lin.w}, f, ensure_ascii=False, indent=2)

                                    
    @staticmethod
    def _role_feats(task: Dict[str, Any], role_name: str, rp) -> List[str]:
        txt_task = str(task.get("prompt") or task.get("question") or task)
        txt_role = " ".join([
            rp.description or "",
            rp.system_template or "",
            rp.user_template or "",
            " ".join(list(rp.capabilities or []))
        ])
        feats: List[str] = []
        feats += _bow_feats("t", txt_task, limit=64)
        feats += _bow_feats("r", txt_role, limit=64)
        feats.append(f"name:{role_name}")
        for c in (rp.capabilities or []):
            feats.append(f"cap:{c}")
                             
        if role_name.startswith("programmer_"):
            feats.append("family:programmer")
        return feats

    @staticmethod
    def _edge_feats(u: str, v: str, u_rp, v_rp) -> List[str]:
        feats = [f"edge:{u}->{v}", f"src:{u}", f"dst:{v}"]
        for c in (u_rp.capabilities or []):
            feats.append(f"src_cap:{c}")
        for c in (v_rp.capabilities or []):
            feats.append(f"dst_cap:{c}")
        return feats

                                     
    def select_programmers(self, task: Dict[str, Any], role_library: Dict[str, Any],
                           prog_names: List[str], topk: Optional[int] = None) -> List[str]:
        if topk is None:
            topk = self.topk_programmers
        scored: List[Tuple[float, str]] = []
        for pn in prog_names:
            rp = role_library.get(pn)
            if not rp:
                continue
            feats = self._role_feats(task, pn, rp)
            s = self.role_lin.score(feats)
                  
            s += random.uniform(-0.01, 0.01)
            scored.append((s, pn))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [pn for _, pn in scored[:max(1, topk)]]
                  
        self._last_ctx["roles_all"] = list(prog_names)
        self._last_ctx["roles_chosen"] = list(chosen)
        self._last_ctx["task_snip"] = str(task.get("prompt") or task.get("question") or "")[:160]
        return chosen

    def decide_edges(self, role_library: Dict[str, Any], chosen_programmers: List[str]) -> Dict[Tuple[str, str], bool]:
        """
        学习是否保留 programmer→code_evaluator 的兜底直连。
        返回字典：{(u,v): active_bool}
        """
        edges = {}
        keep_budget = max(0, int(math.ceil(self.edge_keep_ratio * len(chosen_programmers))))
                   
        scored: List[Tuple[float, Tuple[str, str]]] = []
        for pn in chosen_programmers:
            u_rp = role_library.get(pn); v_rp = role_library.get("code_evaluator")
            feats = self._edge_feats(pn, "code_evaluator", u_rp, v_rp)
            s = self.edge_lin.score(feats) + random.uniform(-0.01, 0.01)
            scored.append((s, (pn, "code_evaluator")))
        scored.sort(key=lambda x: x[0], reverse=True)
        active_set = set([e for _, e in scored[:keep_budget]])
        for _, e in scored:
            edges[e] = (e in active_set)
        self._last_ctx["edges_considered"] = [e for _, e in scored]
        self._last_ctx["edges_active"] = [e for e in active_set]
        return edges

                                                             
    def update_after_run(self, task: Dict[str, Any], program, out: Dict[str, Any],
                         ok: bool, tokens: int = 0, latency: float = 0.0,
                         lambda_cost: float = 1e-5, lambda_latency: float = 0.0) -> None:
                                      
        reward = (1.0 if ok else 0.0) - lambda_cost * float(tokens) - lambda_latency * float(latency)
                  
        all_names = self._last_ctx.get("roles_all", []) or []
        chosen = self._last_ctx.get("roles_chosen", []) or []
        neg = [n for n in all_names if n not in chosen]

        for pn in chosen:
            rp = program.role_library.get(pn)
            feats_pos = self._role_feats(task, pn, rp)
            self.role_lin.update(feats_pos, [], self.lr * reward)
        for pn in neg:
            rp = program.role_library.get(pn)
            feats_neg = self._role_feats(task, pn, rp)
            self.role_lin.update([], feats_neg, self.lr * reward * 0.5)

                                                   
        for (u, v) in self._last_ctx.get("edges_considered", []):
            u_rp = program.role_library.get(u); v_rp = program.role_library.get(v)
            feats = self._edge_feats(u, v, u_rp, v_rp)
            is_active = ((u, v) in set(self._last_ctx.get("edges_active", [])))
            if is_active:
                self.edge_lin.update(feats, [], self.lr * reward)
            else:
                self.edge_lin.update([], feats, self.lr * reward * 0.5)

        self.role_lin.l2_clip()
        self.edge_lin.l2_clip()
        self.save()


                                
_POLICY: Optional[RoleEdgePolicy] = None

def get_policy(cfg: Dict[str, Any]) -> RoleEdgePolicy:
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    pol_cfg = (cfg.get("policy") or {}) if isinstance(cfg, dict) else {}
    path = pol_cfg.get("path", "logs/policy_humaneval.json")
    lr = float(pol_cfg.get("lr", 0.05))
    topk = int(pol_cfg.get("topk_programmers", 2))
    edge_keep_ratio = float(pol_cfg.get("edge_keep_ratio", 1.0))
    _POLICY = RoleEdgePolicy(path=path, lr=lr, topk_programmers=topk, edge_keep_ratio=edge_keep_ratio)
    return _POLICY
