"""pytorchexample: A Flower / PyTorch app."""

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from mnist_flower.task import MLP, load_centralized_dataset, test

from pathlib import Path
from datetime import datetime
import json

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = MLP()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    
    global_metrics = []

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn= lambda server_rounds, arrays: global_evaluate(server_rounds, arrays, global_metrics),
    )
    
    dev_mode = False
    
    # Save the final results into a csv file 
    run_id = context.run_id
                
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if dev_mode:
        results_dir = Path("results") / "dev"
    else:
        results_dir = (
            Path("results") / f"run_{run_id}_{timestamp}"
        )
    
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving experiment results to: {results_dir.resolve()}") 
    
    print("\nSaving model metrics to results folder...")
    with open(results_dir / "global_metrics.json", "w", newline="") as file:
        json.dump(global_metrics, file, indent=4)
    
    print("\nSaving model config to results folder...")
    config = {
        "flower_run_id": run_id,
        "timestamp": timestamp,
        "dataset": "MNIST",
        "model": "MLP",
        "partition_method": "random IID",
        "learning_rate": lr,
        "fraction_evaluate": fraction_evaluate,
        "num_rounds": num_rounds,
        "seed": 42,
        "num_supernodes": 5,
        "differential_privacy": False
    }
    
    with open(results_dir / "config.json", "w" ) as file:
        json.dump(config, file, indent=4)
    

    if context.run_config["save-model"]:
        # Save final model
        print("\nSaving final model to results folder...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, results_dir / "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord, global_metrics_log: list | None = None) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = MLP()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc, test_y_true, test_y_pred = test(model, test_dataloader, device)
    
    if global_metrics_log is not None:
        global_metrics_log.append({
            "round": server_round,
            "accuracy": float(test_acc),
            "loss": float(test_loss)
        })
    
    # classification report 
    cr  = classification_report(test_y_true, test_y_pred)
    print(f"Classification report for global model")
    print("\n",cr) 
    
    # Confusion matrix
    # cm = confusion_matrix(test_y_true, test_y_pred)
    # display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    # display_cm.plot()

    # plt.show()

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
