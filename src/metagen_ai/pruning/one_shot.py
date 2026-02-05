                                    
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set
import networkx as nx
import math
import random

def _active_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        if d.get("active", True):
            H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if d.get("active", True) and H.has_node(u) and H.has_node(v):
            H.add_edge(u, v, **d)
    return H

def _find_exits(G: nx.DiGraph) -> List[str]:
    exits = [n for n, d in G.nodes(data=True) if d.get("is_exit", False) and d.get("active", True)]
    if exits:
        return exits
               
    for name in ("evaluator", "judge"):
        if name in G.nodes and G.nodes[name].get("active", True):
            return [name]
                                                                     
    return [n for n in G.nodes if G.out_degree(n) == 0 and G.nodes[n].get("active", True)]

def _find_sources(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes if G.in_degree(n) == 0 and G.nodes[n].get("active", True)]

def _shortest_keep_edges(G: nx.DiGraph, sources: List[str], exits: List[str]) -> Set[Tuple[str, str]]:
    """Keep at least one path from any source to any exit (shortest path heuristic)."""
    keep: Set[Tuple[str, str]] = set()
    for s in sources:
        for t in exits:
            if s not in G.nodes or t not in G.nodes:
                continue
            try:
                path = nx.shortest_path(G, s, t)
                for i in range(len(path) - 1):
                    keep.add((path[i], path[i+1]))
            except nx.NetworkXNoPath:
                continue
    return keep

def _edge_betweenness_scores(H: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Low betweenness ≈ more prune-able."""
    if H.number_of_edges() == 0:
        return {}
    U = H.to_undirected(as_view=False)
    try:
        eb = nx.edge_betweenness_centrality(U, normalized=True)
    except Exception:
        eb = {}
        for u, v in H.edges:
            eb[(u, v)] = (H.degree(u) * H.degree(v)) / max(1.0, (2.0 * H.number_of_edges()))
    out: Dict[Tuple[str, str], float] = {}
    for u, v in H.edges:
        key = (u, v) if (u, v) in eb else (v, u)
        out[(u, v)] = float(eb.get(key, 0.0))
    return out

def prune_once(program, task: Dict[str, Any], cfg: Dict[str, Any]):
    """
    One-shot 剪枝（安全版）：
      - 永远保活：
          * 图中标记了 required=True 或 keep_alive=True 的边
          * 所有指向出口节点(is_exit=True)的入边
      - 出口节点永远 active
      - 其余边/点：当前版本不做激进裁剪（保守起见，避免误杀主链）
    说明：
      - 这版优先“正确性稳定”，再考虑成本优化。等正确性抬稳后，可以在此基础上恢复贡献度裁剪。
    """
    G: nx.DiGraph = program.G

               
    exit_nodes = [n for n, d in G.nodes(data=True) if d.get("is_exit", False)]
    if not exit_nodes:
                 
        return program

                     
    for n in G.nodes:
        G.nodes[n]["active"] = True
    for u, v in G.edges:
        G[u][v]["active"] = True

                     
    kept_edges: List[Tuple[str, str]] = list(G.graph.get("kept_edges", [])) if isinstance(G.graph.get("kept_edges", []), list) else []

                                        
    for ex in exit_nodes:
        for u, _ in G.in_edges(ex):
            if (u, ex) not in kept_edges:
                kept_edges.append((u, ex))
            G[u][ex]["keep_alive"] = True

                                                           
    kept_set = set(kept_edges)
    for u, v, d in G.edges(data=True):
        if d.get("required") or d.get("keep_alive") or ((u, v) in kept_set):
            G[u][v]["active"] = True

                      
    for ex in exit_nodes:
        G.nodes[ex]["active"] = True

                                 
    G.graph["kept_edges"] = kept_edges
    return program