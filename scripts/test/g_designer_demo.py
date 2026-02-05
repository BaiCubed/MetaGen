                            
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.sampler import sample_architecture
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook

def toy_task():
                                                                                       
    return {"question": "Add 12 and 21. Provide the final result.", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    program = build_task_graph(task, cfg)                             
                                                                     
    program = sample_architecture(program, task, cfg)

                                             
    tg_hook = build_textual_gradient_hook(program, cfg)

                        
    res1 = program.run(task=task, rounds=1, early_exit=True, hooks=None)
    print("[G-Designer] Final (before pruning):", res1["final"])

                    
    program = prune_once(program, task, cfg)

                                                     
                                                                                        
    hooks = type("H", (), {
        "textual_gradient_hook": staticmethod(tg_hook),
                                                              
                                                                 
                                                                
    })()

    res2 = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
    print("[G-Designer] Final (after pruning):", res2["final"])
    print("Kept edges:", program.G.graph.get("kept_edges"))

if __name__ == "__main__":
    main()
