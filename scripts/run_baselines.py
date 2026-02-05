                          
from __future__ import annotations
import argparse, csv, os, time, random
from typing import Dict, Any, Callable

from tqdm import tqdm

from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.utils.llm import build_llm_from_cfg
from metagen_ai.datasets.loader import load_dataset
from metagen_ai.eval.metrics import judge_correct

from metagen_ai.baselines.core import (
    run_cot, run_self_consistency, run_debate, run_tot, run_star_lite
)
from metagen_ai.baselines.fewshot import run_fewshot_cot
from metagen_ai.baselines.graph_structures import (
    run_chain_graph,
    run_complete_graph,
    run_random_graph,
)


def _normalize_task_for_mc(task: Dict[str, Any]) -> Dict[str, Any]:
    """MMLU JSONL produced by prepare_mmlu.py stores choices separately.

    Existing baselines often expect the question field to already contain option lines.
    To avoid touching already-implemented baselines, we normalize ONLY when needed.
    """
    try:
        choices = task.get("choices")
        q = (task.get("question") or "").strip()
        if not isinstance(choices, list) or not choices:
            return task
                                                               
        if "Options:" in q or "OPTIONS:" in q:
            return task
        import re
        if re.search(r"(?m)^\s*[A-E][\).]\s+", q):
            return task

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opt_lines = []
        for i, c in enumerate(choices):
            if i >= len(letters):
                break
            opt_lines.append(f"{letters[i]}. {c}")
        q_full = (
            f"{q}\n\nOptions:\n" + "\n".join(opt_lines) +
            "\n\nAnswer with exactly one line: Final answer: <LETTER>"
        )
        t2 = dict(task)
        t2["question"] = q_full
        return t2
    except Exception:
        return task

BASELINES: Dict[str, Callable] = {
    "cot": run_cot,
    "fewshot_cot": run_fewshot_cot,
    "chain": run_chain_graph,
    "tree": run_tot,                                  
    "selfcons": run_self_consistency,
    "debate": run_debate,
    "tot": run_tot,
    "star": run_star_lite,
    "complete_graph": run_complete_graph,
    "random_graph": run_random_graph,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="mnli_dev_matched")
    ap.add_argument("--baseline", default="cot", choices=list(BASELINES.keys()))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=-1)
          
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=512)
         
    ap.add_argument("--sc_k", type=int, default=5)
            
    ap.add_argument("--debate_rounds", type=int, default=2)
         
    ap.add_argument("--tot_breadth", type=int, default=3)
    ap.add_argument("--tot_depth", type=int, default=2)
                  
    ap.add_argument("--n_shots", type=int, default=5)
    ap.add_argument(
        "--shots_dataset",
        default=None,
        help="Few-shot pool dataset name (e.g., mmlu_val). If omitted, uses sensible defaults.",
    )
                     
    ap.add_argument("--chain_steps", type=int, default=3)
    ap.add_argument("--graph_agents", type=int, default=3)
    ap.add_argument("--rand_edge_prob", type=float, default=0.5)
    ap.add_argument("--judge_temperature", type=float, default=0.2)
        
    ap.add_argument("--out", default=None)
              
    ap.add_argument(
        "--print_failures",
        action="store_true",
        help="Print wrong predictions (gold vs model output) for inspection.",
    )
    args = ap.parse_args()

    cfg = bootstrap(args.config)
    os.makedirs("logs/metrics", exist_ok=True)
    out_csv = args.out or f"logs/metrics/baseline_{args.baseline}.csv"

    tasks_all = list(load_dataset(args.dataset))

                                            
    if str(args.dataset).lower() == "mmlu_test":
        tasks_all = tasks_all[:153]

    tasks = tasks_all if args.max_examples <= 0 else tasks_all[: args.max_examples]

    if not tasks:
        print(f"[WARN] dataset empty: {args.dataset}")
        return

    rows = []
    for seed in range(args.seeds):
        random.seed(seed)
        llm = build_llm_from_cfg(cfg)

                                                  
        shots_pool = None
        if args.baseline == "fewshot_cot":
                                               
                                                                                         
            pool_name = args.shots_dataset
            if pool_name:
                try:
                    shots_pool = list(load_dataset(pool_name))
                except Exception:
                    shots_pool = None

                            
        def run_one(i: int, task: Dict[str, Any]):
            task = _normalize_task_for_mc(task)
            if args.baseline == "cot":
                return run_cot(
                    llm,
                    task,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "fewshot_cot":
                return run_fewshot_cot(
                    llm,
                    task,
                    shots_pool=shots_pool,
                    n_shots=args.n_shots,
                    seed=seed * 100000 + i,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "chain":
                return run_chain_graph(
                    llm,
                    task,
                    steps=args.chain_steps,
                    temperature=max(args.temperature, 0.3),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "tree":
                return run_tot(
                    llm,
                    task,
                    breadth=args.tot_breadth,
                    depth=args.tot_depth,
                    temperature=max(args.temperature, 0.4),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "selfcons":
                return run_self_consistency(
                    llm,
                    task,
                    k=args.sc_k,
                    temperature=max(args.temperature, 0.5),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "debate":
                return run_debate(
                    llm,
                    task,
                    rounds=args.debate_rounds,
                    temperature=max(args.temperature, 0.5),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "tot":
                return run_tot(
                    llm,
                    task,
                    breadth=args.tot_breadth,
                    depth=args.tot_depth,
                    temperature=max(args.temperature, 0.4),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "star":
                return run_star_lite(
                    llm,
                    task,
                    temperature=max(args.temperature, 0.3),
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "complete_graph":
                return run_complete_graph(
                    llm,
                    task,
                    n_agents=args.graph_agents,
                    temperature=max(args.temperature, 0.5),
                    judge_temperature=args.judge_temperature,
                    max_tokens=args.max_tokens,
                )
            if args.baseline == "random_graph":
                return run_random_graph(
                    llm,
                    task,
                    n_agents=args.graph_agents,
                    edge_prob=args.rand_edge_prob,
                    seed=seed * 100000 + i,
                    temperature=max(args.temperature, 0.5),
                    judge_temperature=args.judge_temperature,
                    max_tokens=args.max_tokens,
                )
            raise ValueError("unknown baseline")

        n_ok, tot_tok, lat_sum = 0, 0, 0.0
        bar = tqdm(
            total=len(tasks),
            desc=f"{args.dataset} {args.baseline} seed={seed}",
            unit="ex",
        )

        for i, task in enumerate(tasks, 1):
            out, usage, dt = run_one(i, task)                      
            ok = bool(judge_correct(task, out))
            n_ok += int(ok)
            tot_tok += usage.get("total_tokens", 0)
            lat_sum += dt

                                 
            if args.print_failures and not ok:
                print("\n[BASELINE-FAIL]")
                print(f"  dataset   = {args.dataset}")
                print(f"  baseline  = {args.baseline}")
                print(f"  seed      = {seed}")
                print(f"  idx       = {i}")
                print(f"  task_id   = {task.get('id', '')}")
                print(f"  gold      = {task.get('answer', '')}")
                print(f"  prediction= {out}")
                q = task.get("question") or task.get("prompt") or ""
                if q:
                    print("  question:")
                    print(q)
                print("-" * 80)
                                      

            bar.set_postfix(
                {
                    "acc": f"{(n_ok / i):.3f}",
                    "avg_tok": f"{(tot_tok / max(1, i)):.1f}",
                    "avg_lat": f"{(lat_sum / max(1, i)):.2f}s",
                }
            )
            bar.update(1)
        bar.close()

        rows.append(
            {
                "baseline": args.baseline,
                "dataset": args.dataset,
                "seed": seed,
                "accuracy": n_ok / max(1, len(tasks)),
                "avg_tokens": tot_tok / max(1, len(tasks)),
                "avg_latency_s": lat_sum / max(1, len(tasks)),
                "count": len(tasks),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved -> {out_csv}")


if __name__ == "__main__":
    main()
