                            
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.feedback.param_textual import build_parametric_hooks

def toy_task():
    return {"question": "Compute 12 + 21. Output only the final number.", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

                                      
    program = build_task_graph(task, cfg)
                                                     
    program = sample_architecture_costaware(program, task, cfg)

                                                        
    pm = ParamManager()
    hooks = build_parametric_hooks(program, pm)

                                                     
    for r in range(5):
        res = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
        print(f"[Round {r}] Final={res['final']}")
                                                       
        for role_name in ["math_simplifier", "calculator", "evaluator"]:
            if role_name in program.role_library:
                rp = program.role_library[role_name]
                print(f"  - {role_name}: T={rp.temperature} hints={pm.prompt_hints.get(role_name, [])}")

if __name__ == "__main__":
    main()
