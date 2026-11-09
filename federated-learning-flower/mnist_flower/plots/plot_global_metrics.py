import json 
from pathlib import Path

import matplotlib.pyplot as plt

# retrieve the saved experiments
results_root = Path("results")

metric_files = list(results_root.glob("*/global_metrics.json"))

if not metric_files:
    raise FileNotFoundError(
        "No global_metrics.json found. Check your results folder."
    )

# Use the most recently modified metrics file
metrics_file = max(metric_files, key=lambda p: p.stat().st_mtime)
results_dir = metrics_file.parent

with open(metrics_file, "r") as file:
    metrics = json.load(file)

if not metrics:
    raise ValueError("The metrics file contains no recorded rounds.")

metrics.sort(key=lambda item: item["round"])

rounds = [item["round"] for item in metrics]
accuracy = [item["accuracy"] for item in metrics]
loss = [item["loss"] for item in metrics]

# Create a folder for plots
plots_dir = results_dir / "plots"
plots_dir.mkdir(exist_ok=True)

# Plot 1: Global accuracy
plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracy, marker="o")
plt.xlabel("Server round")
plt.ylabel("Global test accuracy")
plt.title("Global Model Accuracy Across Rounds")
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "global_accuracy.png", dpi=300)
plt.show()

# Plot 2: Global loss
plt.figure(figsize=(8, 5))
plt.plot(rounds, loss, marker="o")
plt.xlabel("Server round")
plt.ylabel("Global test loss")
plt.title("Global Model Loss Across Rounds")
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "global_loss.png", dpi=300)
plt.show()

# Print changes in accuracy between rounds
print("\nGlobal accuracy changes:")

for i in range(1, len(metrics)):
    change = accuracy[i] - accuracy[i - 1]
    print(
        f"Round {rounds[i - 1]} -> {rounds[i]}: "
        f"{change:+.4f}"
    )

print(f"\nPlots saved in: {plots_dir.resolve()}")
