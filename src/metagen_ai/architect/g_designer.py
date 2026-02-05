                                        
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import os, random, math
import networkx as nx

from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.roles.schema import RoleProfile
from metagen_ai.roles.generative import generate_and_register_roles
from metagen_ai.graph_ops.runner import GraphProgram
from metagen_ai.utils.llm import build_llm_from_cfg
from metagen_ai.selector.role_selector import select_committee
from metagen_ai.selector.learned_policy import get_policy

                                                                  
_TG_OK = False
try:
    import torch
    from torch import nn
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
    _TG_OK = True
except Exception:
    _TG_OK = False

                                                     
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = None
except Exception:
    SentenceTransformer = None
    _SBERT = None


def _lazy_embedder(cfg: Optional[Dict[str, Any]] = None):
    """
    enc(text) -> List[float]; prefer SBERT; if not available or offline, fallback to BoW hash.
    """
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
            s = math.sqrt(sum(x * x for x in vec)) or 1.0
            return [x / s for x in vec]
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
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class GDesignerConfig:
                
    max_new_roles: int = 3                                       
    min_semantic_distance: float = 0.25                       

                         
    k_top: int = 6
    sparsity: float = 0.5
    use_vgae: bool = False
    vgae_hidden: int = 16
    vgae_epochs: int = 60
    vgae_lr: float = 1e-2

                 
    seed: int = 42


                                                              
if _TG_OK:
    class VGAE(nn.Module):
        def __init__(self, in_dim: int, hid: int):
            super().__init__()
            self.enc1 = GCNConv(in_dim, hid)
            self.mu = GCNConv(hid, hid)
            self.logvar = GCNConv(hid, hid)

        def forward(self, x, edge_index):
            h = torch.relu(self.enc1(x, edge_index))
            mu = self.mu(h, edge_index)
            logvar = self.logvar(h, edge_index)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            adj_hat = torch.sigmoid(torch.matmul(z, z.t()))
            return adj_hat, mu, logvar

    def _vgae_refine(adj_init, x_init, hid, epochs, lr) -> List[List[float]]:
        device = torch.device("cpu")
        x = torch.tensor(x_init, dtype=torch.float32, device=device)
        adj = torch.tensor(adj_init, dtype=torch.float32, device=device)
        edge_index = dense_to_sparse((adj > 0).to(torch.float32))[0]

        model = VGAE(in_dim=x.shape[1], hid=hid).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(max(1, int(epochs))):
            opt.zero_grad()
            adj_hat, mu, logvar = model(x, edge_index)
            recon = torch.nn.functional.binary_cross_entropy(adj_hat, adj)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 1e-3 * kl
            loss.backward()
            opt.step()

        return adj_hat.detach().cpu().tolist()
else:
    def _vgae_refine(*args, **kwargs):
        raise RuntimeError("torch-geometric not available")


                                                                                  

def build_task_graph(task: Dict[str, Any], cfg: Dict[str, Any],
                     role_library: Optional[Dict[str, RoleProfile]] = None) -> GraphProgram:
    """
    HumanEval: fixed minimal chain with learned role selection / edges:
        task_hub → spec_parser → programmer_v* → static_checker → repairer → code_evaluator (exit)

    MMLU / 多选题：
        task_hub → mmlu_direct_answerer (exit)

    DROP-style reading comprehension:
        task_hub → drop_reader → drop_span_proposer → drop_span_aggregator → drop_answerer (exit)
        Optionally: task_hub → [generated_read_i]* → drop_span_aggregator

    Main chain: task_hub → qa_answerer (exit)
        Parallel:   task_hub → [numeric_only_i]* → committee_aggregator
        Strict whitelist: only numeric-related roles kept in the graph.
    """
    rcfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    seed = int(rcfg.get("seed", 42))
    random.seed(seed)

         
    try:
        llm = build_llm_from_cfg(cfg)
    except Exception:
        llm = None

                                 
    role_lib = dict(role_library or BUILTIN_ROLES)

                      
    task_hub_profile = RoleProfile(
        name="task_hub",
        description="A virtual node that distributes the task context.",
        system_template="You are a router; do not solve the task.",
        user_template="Task: {task}\nPrev: {prev_summary}\nUpstream: {inputs}\nRewrite the task briefly for downstream agents.",
        local_handler=lambda ctx: f"Context: {ctx['task']}",
        capabilities=[]
    )
    role_lib_aug = dict(role_lib)
    role_lib_aug["task_hub"] = task_hub_profile

                                           
    is_humaneval = isinstance(task, dict) and (("entry_point" in task) or ("tests" in task))
    if is_humaneval:
                               
        for n in ["spec_parser", "programmer", "static_checker", "repairer", "code_evaluator"]:
            if n not in role_lib_aug:
                role_lib_aug[n] = BUILTIN_ROLES[n]

        he_cfg = (cfg.get("humaneval") or {}) if isinstance(cfg, dict) else {}
        he_topology = str(he_cfg.get("topology", "full")).lower()

                                                          
                                                                                          
        if he_topology == "simple":
            G = nx.DiGraph()

            def _n(nid: str, is_exit: bool = False):
                if nid not in G.nodes:
                    G.add_node(nid, role=nid, active=True)
                if is_exit:
                    G.nodes[nid]["is_exit"] = True

            def _e(u: str, v: str, active: bool = True, **attrs):
                if not G.has_edge(u, v):
                    ea = {"kind": "space", "active": bool(active)}
                    ea.update(attrs or {})
                    G.add_edge(u, v, **ea)
                else:
                    G[u][v]["active"] = bool(active)
                    for k, val in (attrs or {}).items():
                        G[u][v][k] = val if val is not None else G[u][v].get(k)

                  
            for n in ["task_hub", "spec_parser", "programmer", "static_checker", "repairer", "code_evaluator"]:
                _n(n, is_exit=(n == "code_evaluator"))

                  
            _e("task_hub", "spec_parser", active=True, required=True, keep_alive=True)
            _e("spec_parser", "programmer", active=True, required=True, keep_alive=True)
            _e("programmer", "static_checker", active=True, required=True, keep_alive=True)
            _e("static_checker", "repairer", active=True, required=True, keep_alive=True)
            _e("repairer", "code_evaluator", active=True, required=True, keep_alive=True)

                                        
            G.nodes["code_evaluator"]["is_exit"] = True
            kept = G.graph.setdefault("kept_edges", [])
            for u, v in G.in_edges("code_evaluator"):
                if (u, v) not in kept:
                    kept.append((u, v))
                G[u][v]["keep_alive"] = True

                                          
            G.graph["he_selected_programmers"] = ["programmer"]

            return GraphProgram(
                G=nx.DiGraph(G),
                role_library=role_lib_aug,
                llm=llm,
                default_temperature=cfg.get("llm", {}).get("temperature", 0.2) if isinstance(cfg, dict) else 0.2,
            )
                                                          
        elif he_topology == "chain_aux":
            G = nx.DiGraph()

            def _n(nid: str, is_exit: bool = False):
                if nid not in G.nodes:
                    G.add_node(nid, role=nid, active=True)
                if is_exit:
                    G.nodes[nid]["is_exit"] = True

            def _e(u: str, v: str, active: bool = True, **attrs):
                if not G.has_edge(u, v):
                    ea = {"kind": "space", "active": bool(active)}
                    ea.update(attrs or {})
                    G.add_edge(u, v, **ea)
                else:
                    G[u][v]["active"] = bool(active)
                    for k, val in (attrs or {}).items():
                        G[u][v][k] = val if val is not None else G[u][v].get(k)

                                            
            if "aux_code_analyzer" not in role_lib_aug:
                base = role_lib_aug["static_checker"]
                role_lib_aug["aux_code_analyzer"] = RoleProfile(
                    name="aux_code_analyzer",
                    description=(base.description or "") + " (acts as an auxiliary analyzer before repair.)",
                    system_template=(base.system_template or "")
                                    + "\nYou are an auxiliary code analyst. "
                                      "Read the current candidate implementation and produce a short, actionable critique "
                                      "for the repairer. Focus only on the most likely bugs or missing corner cases.",
                    user_template=base.user_template,
                    local_handler=base.local_handler,
                    temperature=base.temperature,
                    capabilities=list(base.capabilities or []),
                )

                                 
            base_nodes = [
                "task_hub",
                "spec_parser",
                "programmer",
                "static_checker",
                "repairer",
                "code_evaluator",
                "aux_code_analyzer",
            ]
            for n in base_nodes:
                _n(n, is_exit=(n == "code_evaluator"))

                                                                                                 
            _e("task_hub", "spec_parser", active=True, required=True, keep_alive=True)
            _e("spec_parser", "programmer", active=True, required=True, keep_alive=True)
            _e("programmer", "static_checker", active=True, required=True, keep_alive=True)
            _e("static_checker", "repairer", active=True, required=True, keep_alive=True)
            _e("repairer", "code_evaluator", active=True, required=True, keep_alive=True)

                                                            
            _e("programmer", "aux_code_analyzer",
               active=True, required=False, keep_alive=False)
            _e("aux_code_analyzer", "repairer",
               active=True, required=False, keep_alive=True)

                                  
            G.nodes["code_evaluator"]["is_exit"] = True
            kept = G.graph.setdefault("kept_edges", [])
            for u, v in G.in_edges("code_evaluator"):
                if (u, v) not in kept:
                    kept.append((u, v))
                G[u][v]["keep_alive"] = True

                                 
            G.graph["he_selected_programmers"] = ["programmer"]

            return GraphProgram(
                G=nx.DiGraph(G),
                role_library=role_lib_aug,
                llm=llm,
                default_temperature=cfg.get("llm", {}).get("temperature", 0.2)
                if isinstance(cfg, dict) else 0.2,
            )

                                                                         
        K = int(he_cfg.get("n_programmers", 3))
        style_bank = he_cfg.get("programmer_styles") or [
            "concise recursion & handle corner cases",
            "iterative with early returns",
            "comprehensive docstring & examples",
            "prefer pythonic builtins and comprehensions",
            "avoid recursion; emphasize clarity and tests hints",
        ]

        def _clone_role(base: RoleProfile, new_name: str, style_hint: str) -> RoleProfile:
            sys_t = (base.system_template or "") + ("\n(Style preference: " + style_hint + ")")
            return RoleProfile(
                name=new_name,
                description=(base.description or "") + " (style=" + style_hint + ")",
                system_template=sys_t,
                user_template=base.user_template,
                local_handler=base.local_handler,
                temperature=base.temperature,
                capabilities=list(base.capabilities or []),
            )

        prog_all: List[str] = []
        for i in range(K):
            style = style_bank[i % len(style_bank)]
            new_name = f"programmer_v{i+1}"
            if new_name not in role_lib_aug:
                role_lib_aug[new_name] = _clone_role(role_lib_aug["programmer"], new_name, style)
            prog_all.append(new_name)

                                                             
        policy = get_policy(cfg)
        chosen = policy.select_programmers(task, role_lib_aug, prog_all, topk=None)

        G = nx.DiGraph()

        def _n(nid: str, is_exit: bool = False):
            if nid not in G.nodes:
                G.add_node(nid, role=nid, active=True)
            if is_exit:
                G.nodes[nid]["is_exit"] = True

        def _e(u: str, v: str, active: bool = True, **attrs):
            if not G.has_edge(u, v):
                ea = {"kind": "space", "active": bool(active)}
                ea.update(attrs or {})
                G.add_edge(u, v, **ea)
            else:
                G[u][v]["active"] = bool(active)
                for k, val in (attrs or {}).items():
                    G[u][v][k] = val if val is not None else G[u][v].get(k)

                    
        for n in ["task_hub", "spec_parser", "static_checker", "repairer", "code_evaluator"]:
            _n(n, is_exit=(n == "code_evaluator"))
        for pn in prog_all:
            _n(pn)

                                
        _e("task_hub", "spec_parser", active=True, required=True, keep_alive=True)

                                              
        for pn in prog_all:
            act = (pn in chosen)
            _e("task_hub", pn, active=act, required=act, keep_alive=act)
            _e("spec_parser", pn, active=act, required=act, keep_alive=act)

                                             
        for pn in chosen:
            _e(pn, "static_checker", active=True, required=True, keep_alive=True)
            _e("static_checker", "repairer", active=True, required=True, keep_alive=True)
            _e("repairer", "code_evaluator", active=True, required=True, keep_alive=True)

                                                                                 
        edge_actives = policy.decide_edges(role_lib_aug, chosen)
        for (u, v), act in edge_actives.items():
            _e(u, v, active=act, required=False, keep_alive=True)

                                    
        G.nodes["code_evaluator"]["is_exit"] = True
        kept = G.graph.setdefault("kept_edges", [])
        for u, v in G.in_edges("code_evaluator"):
            if (u, v) not in kept:
                kept.append((u, v))
            G[u][v]["keep_alive"] = True

        G.graph["he_selected_programmers"] = list(chosen)

        return GraphProgram(
            G=nx.DiGraph(G),
            role_library=role_lib_aug,
            llm=llm,
            default_temperature=cfg.get("llm", {}).get("temperature", 0.2) if isinstance(cfg, dict) else 0.2,
        )
                                                                 
    is_mmlu = (
        isinstance(task, dict)
        and isinstance(task.get("choices"), list)
        and len(task.get("choices")) > 0
    )

    if is_mmlu:
        dataset_name = str(task.get("dataset") or "").lower()
                                        
        print("[MMLU DEBUG] enter is_mmlu branch:",
              "dataset_name=", dataset_name,
              "task_id=", task.get("id"),
              "subject=", task.get("subject"),
              "choices=", task.get("choices"))
        if dataset_name == "mnli":
                                                                              
            for rname in ["nli_analyst", "nli_labeler"]:
                if rname not in role_lib_aug:
                    role_lib_aug[rname] = BUILTIN_ROLES[rname]

            G = nx.DiGraph()

            def _safe_add_node_mc(nid: str, is_exit: bool = False):
                if nid not in G.nodes:
                    G.add_node(nid, role=nid, active=True)
                if is_exit:
                    G.nodes[nid]["is_exit"] = True

            def _safe_add_edge_mc(u: str, v: str, **attrs):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **attrs)
                else:
                    G[u][v].update(attrs)

            _safe_add_node_mc("task_hub", is_exit=False)
            _safe_add_node_mc("nli_analyst", is_exit=False)
            _safe_add_node_mc("nli_labeler", is_exit=True)

            _safe_add_edge_mc(
                "task_hub",
                "nli_analyst",
                required=True,
                keep_alive=True,
                active=True,
                kind="space",
            )
            _safe_add_edge_mc(
                "nli_analyst",
                "nli_labeler",
                required=True,
                keep_alive=True,
                active=True,
                kind="space",
            )
                                                                  

            def _is_nli_candidate_role(rp: Optional[RoleProfile]) -> bool:
                if not rp:
                    return False

                                                                            
                                                               
                name_l = (rp.name or "").lower()
                if "nli_analyst" in name_l:
                    return True

                desc_l = (rp.description or "").lower()
                sys_l = (rp.system_template or "").lower()
                usr_l = (rp.user_template or "").lower()
                text = " ".join([name_l, desc_l, sys_l, usr_l])

                                                                
                ban_kw = [
                    "equation", "numeric", "arithmetic", "calculator",
                    "python", "code", "function signature", "test case",
                    "compile", "debug", "human-eval", "humaneval"
                ]
                if any(bk in text for bk in ban_kw):
                    return False

                                                               
                pos_kw = [
                    "nli", "natural language inference",
                    "entailment", "contradiction", "neutral",
                    "premise", "hypothesis", "textual entailment",
                    "sentence pair", "text pair", "relationship between sentences",
                    "infer the relation", "determine the relation"
                ]
                if any(pk in text for pk in pos_kw):
                    return True

                                                                    
                soft_kw = [
                    "classify", "classification", "label the", "logical relation",
                    "judge the relation", "decide the relation"
                ]
                if any(sk in text for sk in soft_kw):
                    return True

                                       
                return False

            role_gen_cfg = (cfg.get("role_gen") or {})
            if bool(role_gen_cfg.get("enable", True)):
                                        
                role_lib_aug, _ = generate_and_register_roles(task, cfg, role_library=role_lib_aug)
                                        
                picked, _weights = select_committee(task, role_lib_aug, cfg=cfg)

                picked_before_filter = list(picked)

                               
                picked = [r for r in picked if _is_nli_candidate_role(role_lib_aug.get(r))]
                mnli_cfg = (cfg.get("mnli") or {}) if isinstance(cfg, dict) else {}
                max_helpers = int(mnli_cfg.get("max_helpers", 2))
                picked = picked[:max_helpers]
                print(f"[MNLI role_gen] picked_before_filter={picked_before_filter}, picked_after_filter={picked}")
                                                                      
                for rname in picked:
                    _safe_add_node_mc(rname, is_exit=False)
                    _safe_add_edge_mc(
                        "task_hub",
                        rname,
                        required=False,
                        keep_alive=False,
                        active=True,
                        kind="space",
                    )
                    _safe_add_edge_mc(
                        rname,
                        "nli_analyst",
                        required=False,
                        keep_alive=False,
                        active=True,
                        kind="space",
                    )

                                                        
            G.nodes["nli_labeler"]["is_exit"] = True
            kept = G.graph.setdefault("kept_edges", [])
            for u, v in G.in_edges("nli_labeler"):
                if (u, v) not in kept:
                    kept.append((u, v))
                G[u][v]["keep_alive"] = True

            G.graph["task_type"] = "mnli"

            return GraphProgram(
                G=nx.DiGraph(G),
                role_library=role_lib_aug,
                llm=llm,
                default_temperature=cfg.get("llm", {}).get("temperature", 0.2)
                if isinstance(cfg, dict) else 0.2,
            )

                                                   
        else:

                                                                   
            if dataset_name == "aqua":
                print("[AQUA DEBUG] enter AQuA-specific branch:",
                      "dataset_name=", dataset_name,
                      "task_id=", task.get("id"))

                ans_role_name = "aqua_answerer"
                for r in ["aqua_reasoner", "aqua_critic", "aqua_refiner", "aqua_arbiter", "aqua_answerer"]:
                    if r not in role_lib_aug:
                        role_lib_aug[r] = BUILTIN_ROLES[r]

                G = nx.DiGraph()

                def _safe_add_node_mc(nid: str, is_exit: bool = False):
                    if nid not in G.nodes:
                        G.add_node(nid, role=nid, active=True)
                    if is_exit:
                        G.nodes[nid]["is_exit"] = True

                def _safe_add_edge_mc(u: str, v: str, **attrs):
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, **attrs)
                    else:
                        G[u][v].update(attrs)

                                                                                
                _safe_add_node_mc("task_hub", is_exit=False)
                _safe_add_node_mc("aqua_reasoner", is_exit=False)
                _safe_add_node_mc("aqua_critic", is_exit=False)
                _safe_add_node_mc("aqua_refiner", is_exit=False)
                _safe_add_node_mc("aqua_arbiter", is_exit=False)
                _safe_add_node_mc("aqua_answerer", is_exit=True)

                _safe_add_edge_mc("task_hub", "aqua_reasoner", required=True, keep_alive=True, active=True, kind="space")
                _safe_add_edge_mc("aqua_reasoner", "aqua_critic", required=True, keep_alive=True, active=True, kind="space")
                _safe_add_edge_mc("aqua_critic", "aqua_refiner", required=True, keep_alive=True, active=True, kind="space")
                _safe_add_edge_mc("aqua_refiner", "aqua_arbiter", required=True, keep_alive=True, active=True, kind="space")
                _safe_add_edge_mc("aqua_arbiter", "aqua_answerer", required=True, keep_alive=True, active=True, kind="space")

                                              
                aqua_cfg = ((cfg.get("aqua") or cfg.get("mmlu") or {}) if isinstance(cfg, dict) else {})
                helper_mode = str(aqua_cfg.get("helper_mode", "clone")).lower()
                num_helpers = int(aqua_cfg.get("num_helpers", 2))
                max_helpers = int(aqua_cfg.get("max_helpers", num_helpers))
                num_helpers = max(0, min(num_helpers, max_helpers))

                print("[AQUA DEBUG] helper_mode=", helper_mode,
                      "num_helpers=", num_helpers,
                      "max_helpers=", max_helpers)

                                                     
                if helper_mode == "clone" and num_helpers > 0:
                    for k in range(num_helpers):
                        rname = f"aqua_helper_v{k+1}"
                        if rname not in role_lib_aug:
                            role_lib_aug[rname] = RoleProfile(
                                name=rname,
                                description="A helper analyst for AQuA multiple-choice questions. Provide analysis + a suggested letter.",
                                system_template=(
                                    "You are a HELPER analyst for AQuA-style (AQuA-RAT) multiple-choice questions.\n"
                                    "You are NOT the final decision maker.\n"
                                    "Solve ONLY the given task.\n"
                                    "Do NOT invent or switch to unrelated problems.\n\n"
                                    "Output format:\n"
                                    "- Write a short analysis (can be step-by-step).\n"
                                    "- On the LAST line, output exactly: Helper answer: <A|B|C|D|E>\n"
                                    "Do NOT output 'Final answer'."
                                ),
                                user_template=(
                                    "Task: {task}\n"
                                    "Upstream (if any): {inputs}\n\n"
                                    "Give your analysis, then end with one line: Helper answer: <A|B|C|D|E>."
                                ),
                                local_handler=None,
                                temperature=0.2,
                                capabilities=["analyze", "compute", "verify"],
                            )

                        _safe_add_node_mc(rname, is_exit=False)
                        _safe_add_edge_mc("task_hub", rname, required=False, keep_alive=False, active=True, kind="space")
                        _safe_add_edge_mc(rname, "aqua_reasoner", required=False, keep_alive=False, active=True, kind="space")
                        _safe_add_edge_mc(rname, "aqua_critic", required=False, keep_alive=False, active=True, kind="space")

                    print("[AQUA DEBUG] clone-mode helpers:", [n for n in G.nodes if n.startswith("aqua_helper_v")])

                                               
                elif helper_mode == "llm":
                    role_gen_cfg = (cfg.get("role_gen") or {}) if isinstance(cfg, dict) else {}
                    print("[AQUA DEBUG] LLM-mode role_gen_cfg =", role_gen_cfg)

                    if bool(role_gen_cfg.get("enable", True)):
                        before_roles = set(role_lib_aug.keys())
                        role_lib_aug, _ = generate_and_register_roles(task, cfg, role_library=role_lib_aug)
                        after_roles = set(role_lib_aug.keys())
                        new_roles = sorted(after_roles - before_roles)
                        print("[AQUA DEBUG] new_roles=", new_roles)

                        picked, _weights = select_committee(task, role_lib_aug, cfg=cfg)
                        print("[AQUA DEBUG] committee picked (before llm filter):", picked)

                        def _llm_filter_aqua_helpers(cand_names: List[str], max_keep: int = 2) -> List[str]:
                            if not cand_names:
                                return []
                            import json as _json
                            lines = []
                            for rn in cand_names:
                                rp = role_lib_aug.get(rn)
                                if not rp:
                                    continue
                                desc = (rp.description or "").strip()
                                sys_t = (rp.system_template or "").strip()
                                lines.append(f"- {rn}: {desc} | system: {sys_t[:200]}")

                            q = (task.get("question") or "").strip()
                            choices = task.get("choices") or []
                            choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                            task_text = q + "\n\nChoices:\n" + choices_str

                            prompt = (
                                "You are selecting helpful assistant roles for solving an AQuA (AQuA-RAT) multiple-choice question.\n"
                                "Question:\n" + task_text + "\n\n"
                                "Candidate roles:\n" + "\n".join(lines) + "\n\n"
                                "Pick at most " + str(max_keep) + " roles that are most helpful.\n"
                                "Return ONLY a JSON array of role names, e.g. [\"role_a\", \"role_b\"]."
                            )
                            try:
                                resp = llm.chat(
                                    messages=[
                                        {"role": "system", "content": "You are a precise JSON-only selector."},
                                        {"role": "user", "content": prompt},
                                    ],
                                    temperature=0.0,
                                )
                                content = resp["choices"][0]["message"]["content"]
                                kept = _json.loads(content)
                                kept = [str(x) for x in kept if str(x) in cand_names]
                                return kept[:max_keep]
                            except Exception as e:
                                print("[AQUA DEBUG] llm filter failed, fallback. error=", e)
                                return cand_names[:max_keep]

                        helpers = _llm_filter_aqua_helpers(picked, max_keep=max_helpers)
                        print("[AQUA DEBUG] helpers after llm filter:", helpers)

                        for rname in helpers:
                            _safe_add_node_mc(rname, is_exit=False)
                            _safe_add_edge_mc("task_hub", rname, required=False, keep_alive=False, active=True, kind="space")
                            _safe_add_edge_mc(rname, "aqua_reasoner", required=False, keep_alive=False, active=True, kind="space")
                            _safe_add_edge_mc(rname, "aqua_critic", required=False, keep_alive=False, active=True, kind="space")

                                                                
                G.nodes[ans_role_name]["is_exit"] = True
                kept = G.graph.setdefault("kept_edges", [])

                backbone = [
                    ("task_hub", "aqua_reasoner"),
                    ("aqua_reasoner", "aqua_critic"),
                    ("aqua_critic", "aqua_refiner"),
                    ("aqua_refiner", "aqua_arbiter"),
                    ("aqua_arbiter", ans_role_name),
                ]
                for (u, v) in backbone:
                    if (u, v) not in kept:
                        kept.append((u, v))
                    if G.has_edge(u, v):
                        G[u][v]["keep_alive"] = True

                for (u, v) in list(G.in_edges(ans_role_name)):
                    if (u, v) not in kept:
                        kept.append((u, v))
                    G[u][v]["keep_alive"] = True

                G.graph["task_type"] = "aqua"

                print("[AQUA DEBUG] final G nodes:", list(G.nodes))
                print("[AQUA DEBUG] final G edges:", list(G.edges))

                return GraphProgram(
                    G=nx.DiGraph(G),
                    role_library=role_lib_aug,
                    llm=llm,
                    default_temperature=cfg.get("llm", {}).get("temperature", 0.2)
                    if isinstance(cfg, dict) else 0.2,
                )

                                                                                        
            print("[MMLU DEBUG] enter MMLU-specific branch:",
                  "dataset_name=", dataset_name,
                  "task_id=", task.get("id"),
                  "subject=", task.get("subject"))

            ans_role_name = "mmlu_answerer"
            for r in ["mmlu_reasoner", "mmlu_critic", "mmlu_refiner", "mmlu_arbiter", "mmlu_answerer"]:
                if r not in role_lib_aug:
                    role_lib_aug[r] = BUILTIN_ROLES[r]

            G = nx.DiGraph()

            def _safe_add_node_mc(nid: str, is_exit: bool = False):
                if nid not in G.nodes:
                    G.add_node(nid, role=nid, active=True)
                if is_exit:
                    G.nodes[nid]["is_exit"] = True

            def _safe_add_edge_mc(u: str, v: str, **attrs):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **attrs)
                else:
                    G[u][v].update(attrs)

                                                  
            _safe_add_node_mc("task_hub", is_exit=False)
            _safe_add_node_mc("mmlu_reasoner", is_exit=False)
            _safe_add_node_mc("mmlu_critic", is_exit=False)
            _safe_add_node_mc("mmlu_refiner", is_exit=False)
            _safe_add_node_mc("mmlu_arbiter", is_exit=False)       
            _safe_add_node_mc("mmlu_answerer", is_exit=True)

            _safe_add_edge_mc("task_hub", "mmlu_reasoner", required=True, keep_alive=True, active=True, kind="space")
            _safe_add_edge_mc("mmlu_reasoner", "mmlu_critic", required=True, keep_alive=True, active=True, kind="space")
            _safe_add_edge_mc("mmlu_critic", "mmlu_refiner", required=True, keep_alive=True, active=True, kind="space")

                                                  
            _safe_add_edge_mc("mmlu_refiner", "mmlu_arbiter", required=True, keep_alive=True, active=True, kind="space")
            _safe_add_edge_mc("mmlu_arbiter", "mmlu_answerer", required=True, keep_alive=True, active=True, kind="space")

                                      
            mmlu_cfg = (cfg.get("mmlu") or {}) if isinstance(cfg, dict) else {}
            helper_mode = str(mmlu_cfg.get("helper_mode", "clone")).lower()
            num_helpers = int(mmlu_cfg.get("num_helpers", 2))
            max_helpers = int(mmlu_cfg.get("max_helpers", num_helpers))

            print("[MMLU DEBUG] helper_mode=", helper_mode,
                  "num_helpers=", num_helpers,
                  "max_helpers=", max_helpers)

                                                                          
            if helper_mode == "clone":
                                   
                num_helpers = int(mmlu_cfg.get("num_helpers", 1))
                max_helpers = int(mmlu_cfg.get("max_helpers", num_helpers))
                num_helpers = max(0, min(num_helpers, max_helpers))

                for k in range(num_helpers):
                    rname = f"mmlu_helper_v{k+1}"
                    if rname not in role_lib_aug:
                        role_lib_aug[rname] = RoleProfile(
                            name=rname,
                            description="A helper analyst for MMLU multiple-choice questions. Provide analysis + a suggested letter.",
                            system_template=(
                                "You are a HELPER analyst for MMLU-style multiple-choice questions.\n"
                                "You are NOT the final decision maker.\n"
                                "Solve ONLY the given task.\n"
                                "Do NOT invent or switch to unrelated problems.\n\n"
                                "Output format:\n"
                                "- Write a short analysis (can be step-by-step).\n"
                                "- On the LAST line, output exactly: Helper answer: <A|B|C|D>\n"
                                "Do NOT output 'Final answer'."
                            ),
                                                                                  
                            user_template=(
                                "Task: {task}\n"
                                "Upstream (if any): {inputs}\n\n"
                                "Give your analysis, then end with one line: Helper answer: <A|B|C|D>."
                            ),
                            local_handler=None,
                            temperature=0.2,
                            capabilities=["analyze", "compute", "verify"],
                        )

                    _safe_add_node_mc(rname, is_exit=False)

                                            
                    _safe_add_edge_mc(
                        "task_hub", rname,
                        required=False, keep_alive=False, active=True, kind="space",
                    )

                                                                     
                    _safe_add_edge_mc(
                        rname, "mmlu_reasoner",
                        required=False, keep_alive=False, active=True, kind="space",
                    )
                    _safe_add_edge_mc(
                        rname, "mmlu_critic",
                        required=False, keep_alive=False, active=True, kind="space",
                    )

                print("[MMLU DEBUG] clone-mode helpers:", [n for n in G.nodes if n.startswith("mmlu_helper_v")])


                                                                     
            elif helper_mode == "llm":
                role_gen_cfg = (cfg.get("role_gen") or {})
                print("[MMLU DEBUG] LLM-mode role_gen_cfg =", role_gen_cfg)

                if bool(role_gen_cfg.get("enable", True)):
                              
                    before_roles = set(role_lib_aug.keys())
                    print("[MMLU DEBUG] before generate_and_register_roles:",
                          "num_roles=", len(before_roles))

                    role_lib_aug, _ = generate_and_register_roles(
                        task, cfg, role_library=role_lib_aug
                    )

                    after_roles = set(role_lib_aug.keys())
                    new_roles = sorted(after_roles - before_roles)
                    print("[MMLU DEBUG] after generate_and_register_roles:",
                          "num_roles=", len(after_roles),
                          "new_roles=", new_roles)

                                                      
                    picked, _weights = select_committee(task, role_lib_aug, cfg=cfg)
                    print("[MMLU DEBUG] committee picked (before llm filter):", picked)

                                     
                    def _llm_filter_mmlu_helpers(
                        cand_names: List[str],
                        max_keep: int = 2,
                    ) -> List[str]:
                        if not cand_names:
                            return []

                                                                       
                        import json as _json

                        lines = []
                        for rn in cand_names:
                            rp = role_lib_aug.get(rn)
                            if not rp:
                                continue
                            desc = (rp.description or "").strip()
                            sys_t = (rp.system_template or "").strip()
                            lines.append(
                                f"- {rn}: {desc} | system: {sys_t[:200]}"
                            )

                        q = (task.get("question") or "").strip()
                        choices = task.get("choices") or []
                        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                        task_text = q + "\n\nChoices:\n" + choices_str
                        subj = (task.get("subject") or "").strip()
                        prompt = (
                            "You are selecting helpful assistant roles for solving an MMLU multiple-choice question.\n"
                            "Question (subject="
                            + subj
                            + "):\n"
                            + task_text
                            + "\n\n"
                              "Candidate roles:\n"
                            + "\n".join(lines)
                            + "\n\n"
                              "From the candidate roles above, pick at most "
                            + str(max_keep)
                            + " roles that are most helpful for answering this question.\n"
                              "Return ONLY a JSON array of role names, e.g. [\"role_a\", \"role_b\"]. No extra text."
                        )

                        try:
                            resp = llm.chat(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a precise JSON-only selector.",
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.0,
                            )
                            content = resp["choices"][0]["message"]["content"]
                            print("[MMLU DEBUG] llm filter raw content:", content)
                            kept = _json.loads(content)
                            kept = [str(x) for x in kept if str(x) in cand_names]
                            return kept[:max_keep]
                        except Exception as e:
                            print("[MMLU DEBUG] llm filter failed, fallback to original picked. error=", e)
                            return cand_names[:max_keep]

                    helpers = _llm_filter_mmlu_helpers(picked, max_keep=max_helpers)
                    print("[MMLU DEBUG] helpers after llm filter:", helpers)

                    for rname in helpers:
                        _safe_add_node_mc(rname, is_exit=False)
                        _safe_add_edge_mc(
                            "task_hub", rname,
                            required=False, keep_alive=False, active=True, kind="space",
                        )
                        _safe_add_edge_mc(
                            rname, "mmlu_reasoner",
                            required=False, keep_alive=False, active=True, kind="space",
                        )
                        _safe_add_edge_mc(
                            rname, "mmlu_critic",
                            required=False, keep_alive=False, active=True, kind="space",
                        )

            else:
                print("[MMLU DEBUG] helper_mode unknown, fallback to simple main chain only.")

                                                                           
            G.nodes[ans_role_name]["is_exit"] = True

            kept = G.graph.setdefault("kept_edges", [])

                                                      
            backbone = [
                ("task_hub", "mmlu_reasoner"),
                ("mmlu_reasoner", "mmlu_critic"),
                ("mmlu_critic", "mmlu_refiner"),
                ("mmlu_refiner", "mmlu_arbiter"),
                ("mmlu_arbiter", ans_role_name),
            ]

            for (u, v) in backbone:
                if (u, v) not in kept:
                    kept.append((u, v))
                if G.has_edge(u, v):
                    G[u][v]["keep_alive"] = True

                                                            
            for (u, v) in list(G.in_edges(ans_role_name)):
                if (u, v) not in kept:
                    kept.append((u, v))
                G[u][v]["keep_alive"] = True

            if dataset_name == "mmlu":
                G.graph["task_type"] = "mmlu"

            print("[MMLU DEBUG] final G nodes:", list(G.nodes))
            print("[MMLU DEBUG] final G edges:", list(G.edges))

            return GraphProgram(
                G=nx.DiGraph(G),
                role_library=role_lib_aug,
                llm=llm,
                default_temperature=cfg.get("llm", {}).get("temperature", 0.2)
                if isinstance(cfg, dict) else 0.2,
            )
                                            
    passage = task.get("passage") or task.get("context")
    answers = task.get("answers")
    answer = task.get("answer")
    is_drop_like = bool(passage) and (answers is not None or answer is not None)

    if is_drop_like:
                                 
        for n in ["drop_reader", "drop_span_proposer", "drop_span_aggregator", "drop_answerer"]:
            if n not in role_lib_aug:
                role_lib_aug[n] = BUILTIN_ROLES[n]

        G = nx.DiGraph()

        def _safe_add_node(nid: str, is_exit: bool = False):
            if nid not in G.nodes:
                G.add_node(nid, role=nid, active=True)
            if is_exit:
                G.nodes[nid]["is_exit"] = True

        def _safe_add_edge(u: str, v: str, **attrs):
            if not G.has_edge(u, v):
                edge_attr = {"kind": "space", "active": True}
                edge_attr.update(attrs or {})
                G.add_edge(u, v, **edge_attr)
            else:
                for k, val in (attrs or {}).items():
                    G[u][v][k] = val if val is not None else G[u][v].get(k)

                         
        for n in ["task_hub", "drop_reader", "drop_span_proposer", "drop_span_aggregator", "drop_answerer"]:
            _safe_add_node(n, is_exit=(n == "drop_answerer"))

        backbone = [
            ("task_hub", "drop_reader"),
            ("drop_reader", "drop_span_proposer"),
            ("drop_span_proposer", "drop_span_aggregator"),
            ("drop_span_aggregator", "drop_answerer"),
        ]
        for (u, v) in backbone:
            _safe_add_edge(u, v, required=True, keep_alive=True, active=True, kind="space")

                                                                           
        def _is_drop_candidate_role(rp: Optional[RoleProfile]) -> bool:
            if not rp:
                return False
            name_l = (rp.name or "").lower()
            desc_l = (rp.description or "").lower()
            sys_l = (rp.system_template or "").lower()
            usr_l = (rp.user_template or "").lower()
            text = " ".join([desc_l, sys_l, usr_l])

                                                        
            pos = any(kw in text for kw in [
                "reading comprehension", "span", "answer span", "extract", "qa", "question answering"
            ])
            if not pos:
                return False

                                                        
            if any(kw in text for kw in ["calculator", "numeric", "arithmetic", "equation"]):
                return False

            caps = set(rp.capabilities or [])
            if not ({"read", "reason", "parse", "aggregate", "verify"} & caps):
                return False

            return True

        role_gen_cfg = (cfg.get("role_gen") or {})
        if bool(role_gen_cfg.get("enable", True)):
            role_lib_aug, _ = generate_and_register_roles(task, cfg, role_library=role_lib_aug)
            picked, _weights = select_committee(task, role_lib_aug, cfg=cfg)
            picked = [r for r in picked if _is_drop_candidate_role(role_lib_aug.get(r))]
            for rname in picked:
                _safe_add_node(rname)
                _safe_add_edge("task_hub", rname, active=True, kind="space")
                _safe_add_edge(rname, "drop_span_aggregator", active=True, kind="space")

                           
        G.nodes["drop_answerer"]["is_exit"] = True
        kept = G.graph.setdefault("kept_edges", [])
        for e in backbone:
            if e not in kept:
                kept.append(e)
        for u, v in G.in_edges("drop_answerer"):
            if (u, v) not in kept:
                kept.append((u, v))
            G[u][v]["keep_alive"] = True

        return GraphProgram(
            G=nx.DiGraph(G),
            role_library=role_lib_aug,
            llm=llm,
            default_temperature=cfg.get("llm", {}).get("temperature", 0.2) if isinstance(cfg, dict) else 0.2,
        )

                                                 
    qa_cfg = (cfg.get("qa") or {}) if isinstance(cfg, dict) else {}
    qa_topology = str(qa_cfg.get("topology", "simple")).lower()

    for n in ["qa_answerer"]:
        if n not in role_lib_aug:
            role_lib_aug[n] = BUILTIN_ROLES[n]

    G = nx.DiGraph()

    def _safe_add_node_qa(nid: str, is_exit: bool = False):
        if nid not in G.nodes:
            G.add_node(nid, role=nid, active=True)
        if is_exit:
            G.nodes[nid]["is_exit"] = True

    def _safe_add_edge_qa(u: str, v: str, **attrs):
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attrs)
        else:
            G[u][v].update(attrs)

    _safe_add_node_qa("task_hub", is_exit=False)
    _safe_add_node_qa("qa_answerer", is_exit=True)
    _safe_add_edge_qa("task_hub", "qa_answerer", required=True, keep_alive=True, active=True, kind="space")

    kept = G.graph.setdefault("kept_edges", [])
    if ("task_hub", "qa_answerer") not in kept:
        kept.append(("task_hub", "qa_answerer"))
    G["task_hub"]["qa_answerer"]["keep_alive"] = True

    G.graph["task_type"] = "qa"

    return GraphProgram(
        G=nx.DiGraph(G),
        role_library=role_lib_aug,
        llm=llm,
        default_temperature=cfg.get("llm", {}).get("temperature", 0.2)
        if isinstance(cfg, dict) else 0.2,
    )
