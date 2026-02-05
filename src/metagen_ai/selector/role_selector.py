                                          
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import os, json, math, random, re
from collections import defaultdict

from metagen_ai.roles.schema import RoleProfile
from metagen_ai.roles.builtin import BUILTIN_ROLES

                                 
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = None
except Exception:
    SentenceTransformer = None
    _SBERT = None

def _lazy_embedder(cfg: Optional[Dict[str, Any]] = None):
    global _SBERT
    sbert_path = None
    local_only = False
    if isinstance(cfg, dict):
        emb = (cfg.get("embeddings") or {})
        sbert_path = emb.get("sbert_path")
        local_only = bool(emb.get("local_only", False))
    sbert_path = os.environ.get("SBERT_LOCAL_PATH", sbert_path)

    def _offline():
        import hashlib
        def enc(text: str, dim: int = 256):
            vec = [0.0] * dim
            for tok in str(text).lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            s = math.sqrt(sum(x*x for x in vec)) or 1.0
            return [x/s for x in vec]
        return enc

    if local_only:
        if sbert_path and os.path.isdir(sbert_path):
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                if SentenceTransformer is not None and _SBERT is None:
                    _SBERT = SentenceTransformer(sbert_path)
                if _SBERT is not None:
                    def enc(text: str):
                        v = _SBERT.encode([text], normalize_embeddings=True)[0]
                        return v.tolist()
                    return enc
            except Exception:
                pass
        return _offline()

    if SentenceTransformer is not None and _SBERT is None:
        try:
            if sbert_path and os.path.isdir(sbert_path):
                _SBERT = SentenceTransformer(sbert_path)
            else:
                _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT = None

    if _SBERT is not None:
        def enc(text: str):
            v = _SBERT.encode([text], normalize_embeddings=True)[0]
            return v.tolist()
        return enc

    return _offline()

def _cos(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x, y in zip(a, b)))

                                  
@dataclass
class RoleStats:
    attempts: int = 0
    successes: int = 0
    contribs: int = 0                                  
    ema_acc: float = 0.0
    ema_tokens: float = 0.0
    last_ok: int = 0

class RoleStatsDB:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, RoleStats] = {}

    def load(self):
        if os.path.isfile(self.path):
            try:
                obj = json.load(open(self.path, "r", encoding="utf-8"))
                for k, v in obj.items():
                    rs = RoleStats(**v)
                    self.data[k] = rs
            except Exception:
                self.data = {}
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        out = {k: vars(v) for k, v in self.data.items()}
        json.dump(out, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def update_from_run(self,
                        program,                                    
                        out: Dict[str, Any],                          
                        ok: bool,                                       
                        dataset: str = "unknown"):
                          
        active_nodes = [n for n, d in program.G.nodes(data=True) if d.get("active", True) and n != "task_hub"]
        builtin_names = set(BUILTIN_ROLES.keys()) | {"task_hub"}

        kept = program.G.graph.get("kept_edges", []) or []
        contrib_nodes = set()
        for u, v in kept:
            contrib_nodes.add(u); contrib_nodes.add(v)
        contrib_nodes.discard("task_hub")

        usage = out.get("usage", {}) or {}
        total_tokens = float(usage.get("total_tokens", 0.0))

        for nid in active_nodes:
            if nid in builtin_names:
                continue
            rs = self.data.get(nid, RoleStats())
            rs.attempts += 1
            rs.successes += int(ok)
            rs.contribs += int(nid in contrib_nodes)
                 
            rs.ema_acc = 0.8*rs.ema_acc + 0.2*float(ok)
            rs.ema_tokens = 0.9*rs.ema_tokens + 0.1*total_tokens
            rs.last_ok = int(ok)
            self.data[nid] = rs

                                
def dedup_roles(cands: List[RoleProfile],
                threshold: float = 0.92,
                cfg: Optional[Dict[str, Any]] = None) -> List[RoleProfile]:
    if not cands:
        return []
    emb = _lazy_embedder(cfg)
    vecs = []
    for r in cands:
        text = f"{r.name}\n{r.description}\n{r.system_template}\n{r.user_template}"
        vecs.append(emb(text))
    keep: List[int] = []
    for i in range(len(cands)):
        dup = False
        for j in keep:
            if _cos(vecs[i], vecs[j]) >= threshold:
                dup = True
                break
        if not dup:
            keep.append(i)
    return [cands[i] for i in keep]

                                                 
@dataclass
class SelectorConfig:
    committee_size: int = 3
    explore_weight: float = 0.5             
    relevance_weight: float = 0.5
    cost_weight: float = 0.1
    dedup_threshold: float = 0.92
    stats_path: str = "logs/role_stats.json"

def _role_text(r: RoleProfile) -> str:
    return f"{r.name}. {r.description}. {r.system_template}. {r.user_template}"

def _task_text(task: Dict[str, Any]) -> str:
    return str(task.get("question") or task.get("prompt") or task)

def select_committee(task: Dict[str, Any],
                     role_library: Dict[str, RoleProfile],
                     cfg: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, float]]:
    scfg = SelectorConfig(
        committee_size=int(((cfg or {}).get("selector", {}) or {}).get("committee_size", 3)),
        explore_weight=float(((cfg or {}).get("selector", {}) or {}).get("explore_weight", 0.5)),
        relevance_weight=float(((cfg or {}).get("selector", {}) or {}).get("relevance_weight", 0.5)),
        cost_weight=float(((cfg or {}).get("selector", {}) or {}).get("cost_weight", 0.1)),
        dedup_threshold=float(((cfg or {}).get("selector", {}) or {}).get("dedup_threshold", 0.92)),
        stats_path=str(((cfg or {}).get("paths", {}) or {}).get("role_stats_path", "logs/role_stats.json")),
    )

    builtin = set(BUILTIN_ROLES.keys()) | {"task_hub"}
    candidates: List[RoleProfile] = [rp for name, rp in role_library.items() if name not in builtin]

        
    candidates = dedup_roles(candidates, threshold=scfg.dedup_threshold, cfg=cfg)

           
    emb = _lazy_embedder(cfg)
    tvec = emb(_task_text(task))
    rel: Dict[str, float] = {}
    for r in candidates:
        rel[r.name] = _cos(emb(_role_text(r)), tvec)

                             
    stats = RoleStatsDB(scfg.stats_path).load().data
    score_items = []
    for r in candidates:
        rs = stats.get(r.name, RoleStats())
        alpha = 1 + rs.successes
        beta = 1 + max(0, rs.attempts - rs.successes)
        ts = random.betavariate(alpha, beta)               
                    
        cost_pen = min(1.0, rs.ema_tokens / 2000.0) if rs.ema_tokens > 0 else 0.0
        score = scfg.explore_weight*ts + scfg.relevance_weight*rel.get(r.name, 0.0) - scfg.cost_weight*cost_pen
        score_items.append((r.name, float(score)))

                  
    score_items.sort(key=lambda x: x[1], reverse=True)
    picked = [name for name, _ in score_items[: max(1, scfg.committee_size)]]
    weights = {name: (i+1) for i, name in enumerate(reversed(picked))}             
    return picked, weights
