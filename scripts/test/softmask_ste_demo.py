                              
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.graph_ops.softmask import SoftMaskManager, SoftMaskConfig
from metagen_ai.feedback.soft_textual import build_softmask_textual_hook
from metagen_ai.pruning.one_shot import prune_once

def toy_task():
    return {"question": "What is 19 + 24? Provide only the number.", "answer": "43"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

                                                                       
    program = build_task_graph(task, cfg)
    program = sample_architecture_costaware(program, task, cfg)

    print("[Init] Active counts before soft masks:",
          sum(1 for n in program.G.nodes if program.G.nodes[n].get("active", True)),
          "nodes,",
          sum(1 for e in program.G.edges if program.G.edges[e[0], e[1]].get("active", True)),
          "edges")

                                                         
    smm = SoftMaskManager(
        program,
        cfg=SoftMaskConfig(
            tau_node=0.8, tau_edge=0.8,
            thresh_node=0.5, thresh_edge=0.5,
            lr_node=0.25, lr_edge=0.25, wd=1e-3
        )
    )
    smm.forward(write_active=True)

                                                                                       
    tg_hook = build_softmask_textual_hook(smm)

                                                                                  
    ROUNDS = 6
    for r in range(ROUNDS):
        res = program.run(task=task, rounds=1, early_exit=True,
                          hooks=type("H", (), {"textual_gradient_hook": tg_hook})())
        counts = smm.counts()
        print(f"[Round {r}] Final={res['final']} | nodes={counts['active_nodes']} edges={counts['active_edges']}")

                                                                      
    program = prune_once(program, task, cfg)
    kept = program.G.graph.get("kept_edges")
    print("[After Pruning] kept edges:", kept)

if __name__ == "__main__":
    main()
