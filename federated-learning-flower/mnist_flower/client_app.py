"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.clientapp.mod import LocalDpMod

from mnist_flower.task import MLP, load_data
from mnist_flower.task import test as test_fn
from mnist_flower.task import train as train_fn

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Flower ClientApp
app = ClientApp()

# instance of loaclDPmod
local_dp_obj = LocalDpMod(app.clipping_norm, app.sensitivity, app.epsilon, app.delta)

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # save the updated models parameters    
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    # dictionary of the models parameters and training metrics
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    # return a reply to flower, so these can be used in model aggregation
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc, eval_y_pred, eval_y_true = test_fn(
        model,
        valloader,
        device,
    )
    
    # classification report 
    cr  = classification_report(eval_y_true, eval_y_pred)
    print(f"Classification report for {partition_id}")
    print("\n",cr)
    
    # Confusion matrix
    # cm = confusion_matrix(eval_y_true, eval_y_pred)
    # display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    # display_cm.plot()

    # plt.show()

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
