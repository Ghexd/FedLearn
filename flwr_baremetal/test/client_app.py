"""test: A Flower / NumPy app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from test.task import Net, load_data_from_disk
from test.task import test as test_fn
from test.task import train as train_fn
import subprocess
import os

app = ClientApp()

absolute_path = "./"

@app.train()
def train(msg: Message, context: Context):

    print("Start Training")

    # Extract the configuration from the message
    config = msg.content["config"]

    # Check for a "start_train" instruction (before the first round)
    if config.get("task") == "start_train":
        # start monitoring script
        monitor = subprocess.run(['./start_monitor.sh'])
        print(f"Monitoring script script started (exit code: {monitor.returncode})")
        try:
            train_file_path = os.path.join(absolute_path, "Train_time.txt")
            test_file_path = os.path.join(absolute_path, "Test_time.txt")

            with open(train_file_path, "w") as file:
                file.write("")
            with open(test_file_path, "w") as file:
                file.write("")
        except IOError as e:
            print(f"Error writing file: {e}")

    # Read from run config
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data_from_disk(dataset_path, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        learning_rate,
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):

    print("Start Evaluating")

    # Extract the configuration from the message
    config = msg.content["config"]

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data_from_disk(dataset_path, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Check for a "stop_eval" instruction
    if config.get("task") == "stop_eval":
        # stop monitoring script
        monitor = subprocess.run(['./stop_monitor.sh'])
        print(f"perf and pidstat script stopped (exit code: {monitor.returncode})")

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
