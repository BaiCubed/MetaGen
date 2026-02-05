                          
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.eval.suites import basic_arith_n, run_suite

import matplotlib.pyplot as plt
import csv, os

def main():
    cfg = bootstrap("configs/default.yaml")

    out_dir = "logs/metrics"
    os.makedirs(out_dir, exist_ok=True)

                                     
    csv_sanity = os.path.join(out_dir, "arith_sanity.csv")
    m1 = run_suite(basic_arith_n(), cfg, csv_sanity)

                              
    csv_arith = os.path.join(out_dir, "arith_grid.csv")
    m2 = run_suite(basic_arith_n(10, 20), cfg, csv_arith)

                                        
    labels = ["arith_sanity", "arith_grid"]
    accs = [m1["accuracy"], m2["accuracy"]]
    plt.figure()
    plt.bar(labels, accs)
    plt.title("Accuracy by Suite")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    fig_path = os.path.join(out_dir, "accuracy_bar.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print("Saved:", csv_sanity, csv_arith, fig_path)

if __name__ == "__main__":
    main()
