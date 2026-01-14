import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from test_client_balancing.task_apnea import Net, load_data_from_disk
from test_client_balancing.task_apnea import test as test_fn
from test_client_balancing.task_apnea import train as train_fn
import subprocess
import os

app = ClientApp()
absolute_path = "/home"

# Change this in server_app too (in main)
dataset_threshold = 15000

def get_data_partition(dataset_path, batch_size, partition_type="full"):
    trainloader, valloader = load_data_from_disk(dataset_path, batch_size)
    dataset = trainloader.dataset
    total_len = len(dataset)
    
    if partition_type == "full":
        return trainloader, valloader
    
    if partition_type == "standard":
        # Ensure we don't exceed total_len if the dataset is smaller than the threshold
        end_idx = min(dataset_threshold, total_len)
        indices = list(range(0, end_idx))
        
    elif partition_type == "excess":
        # If dataset is smaller than threshold, this will result in an empty list
        start_idx = dataset_threshold
        if start_idx >= total_len:
            indices = []
        else:
            indices = list(range(start_idx, total_len))
            
    else:
        indices = list(range(total_len))
        
    subset = torch.utils.data.Subset(dataset, indices)
    
    if len(subset) == 0:
        print(f"Warning: Partition '{partition_type}' is empty for dataset of size {total_len}")
    else:
        print(f"Subset of length {len(subset)} created for '{partition_type}' partition")
        
    new_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return new_loader, valloader


def start_monitor():
    print("Starting monitoring scripts...")
    # subprocess.run(['/home/start_monitor.sh'])
    try:
        with open(os.path.join(absolute_path, "Train_info.txt"), "w") as f:
            f.write("")
        with open(os.path.join(absolute_path, "Test_info.txt"), "w") as f:
            f.write("")
    except IOError as e:
        print(f"Error writing file: {e}")


@app.train()
def train(msg: Message, context: Context):
    
    config = msg.content["config"]
    task = config.get("task", "standard")
    
    # Check "start_monitoring" instruction
    if config.get("start_monitoring") == "true":
        start_monitor()

    # Round 1: dataset size
    if task == "report_size":
        dataset_path = context.node_config["dataset-path"]
        # Load a single batch only to read the dataset size
        trainloader, _ = load_data_from_disk(dataset_path, 32)
        num_examples = len(trainloader.dataset)
        
        print(f"Reporting dataset size: {num_examples}")

        # Dummy model
        model = Net()
        
        metrics = {"num-examples": num_examples}
        
        content = RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics)
        })
        return Message(content=content, reply_to=msg)

    print(f"Start Training. Task: {task}")

    # Setup
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    dataset_path = context.node_config["dataset-path"]
    
    if task == "train_excess":
        # Rich Client: pre-training phase
        trainloader, _ = get_data_partition(dataset_path, batch_size, "excess")
    elif task == "train_standard":
        # Rich Client: aggregation phase
        trainloader, _ = get_data_partition(dataset_path, batch_size, "standard")
    elif task == "fine_tune":
        # Poor Client: fine-tuning
        trainloader, _ = get_data_partition(dataset_path, batch_size, "full")
    else:
        # Default fallback
        trainloader, _ = get_data_partition(dataset_path, batch_size, "full")

    # Load model
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if task == "fine_tune":
        try:
            round_id = msg.metadata.group_id if msg.metadata.group_id else "unknown"
            
            filename = f"received_model_round_{round_id}.pt"
            save_path = os.path.join(absolute_path, filename)
            
            print(f"Saving received model for privacy analysis: {save_path}")
            torch.save(model.state_dict(), save_path)
            
        except Exception as e:
            print(f"Error saving privacy analysis model: {e}")

    # Train
    train_loss = train_fn(model, trainloader, local_epochs, learning_rate, device)

    # Reply
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
    """Evaluate the model on local data."""

    print("Start Evaluating")
    config = msg.content["config"]

    # Load model
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data_from_disk(dataset_path, batch_size)

    # Test
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Check "stop_eval" instruction
    # if config.get("task") == "stop_eval":
    #     print("Stopping monitor scripts...")
    #     monitor = subprocess.run(['/home/stop_monitor.sh'])
    #     print(f"Monitor stopped (exit code: {monitor.returncode})")

    # Reply
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)