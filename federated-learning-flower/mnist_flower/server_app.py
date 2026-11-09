import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping

from mnist_flower.task import MLP, load_centralized_dataset, test
from mnist_flower.evaluation.metrics import calculate_metrics

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
    num_of_clients = context.run_config["num-of-clients"]

    # Differential privacy parameters
    clipping_norm = float(context.run_config["clipping-norm"])
    noise_multiplier = float(context.run_config["noise-multiplier"])
    # sensitivity = float(context.run_config["sensitivity"])
    # epsilon = float(context.run_config["epsilon"])
    # delta = float(context.run_config["delta"])

    # Load global model
    global_model = MLP()
    arrays = ArrayRecord(global_model.state_dict()) # initial model parameters

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        min_available_nodes=num_of_clients
        
    )
    
    dp_strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy=strategy, 
        noise_multiplier=noise_multiplier, 
        clipping_norm=clipping_norm, 
        num_sampled_clients=num_of_clients
    )
    
    global_metrics = []

    # Start strategy, run FedAvg for `num_rounds`
    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays, # initial model parameters
        train_config=ConfigRecord({"lr": lr}), # config to be sent during the training rounds
        num_rounds=num_rounds,
        evaluate_fn= lambda server_rounds, arrays: global_evaluate(server_rounds, arrays, global_metrics),
    )
    
    dev_mode = True
    
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

    # Calculate Gaussian noise standard deviation
    # noise_std = (
    #     sensitivity * math.sqrt(2 * math.log(1.25 / delta))
    # ) / epsilon
    
    # noise_multiplier = noise_std / clipping_norm
    
    dp = True

    print(f"Saving experiment results to: {results_dir.resolve()}") 
    
    with open(results_dir / "global_metrics.json", "w", newline="") as file:
        json.dump(global_metrics, file, indent=4)

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
        "num_supernodes": context.run_config["num-of-clients"],
        # Differential privacy configuration
        "differential_privacy": dp,
        "clipping_norm": clipping_norm if dp else "N/A",
        "noise_multiplier": noise_multiplier if dp else "N/A"
        # "sensitivity": sensitivity if dp else "N/A",
        # "epsilon": epsilon if dp else "N/A",
        # "delta": delta if dp else "N/A",
        # "noise_std": noise_std if dp else "N/A",
        # "noise_multiplier": noise_multiplier if dp else "N/A",
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
        
    global_metrics = calculate_metrics(test_y_true,test_y_pred)
    global_metrics["accuracy"] = test_acc
    global_metrics["loss"] = test_loss
    
    if global_metrics_log is not None:
        global_metrics_log.append({
            "round": server_round,
            **global_metrics
        })
    
    # Return the evaluation metrics
    flower_metrics = {
        key: value
        for key, value in global_metrics.items()
        if isinstance(value, (int, float))
    }
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
