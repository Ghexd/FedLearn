import logging
import grpc
import torch
import onnx
import io

from onnx2torch import convert
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from . import test_pb2, test_pb2_grpc
from .task_onnx import Net

app = ClientApp()

ip_address = "REAL_CLIENT_IP"

logging.basicConfig()

@app.train()
def train(msg: Message, context: Context):

    print("Start Training")

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    model.eval() 

    # Fake input to ONNX export
    dummy_input = torch.randn(1, 1, 28, 28)

    print("Exporting model in ONNX format...")
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )

    onnx_model_bytes = buffer.getvalue()
    print(f"Export completed. Model size: {len(onnx_model_bytes)} bytes.")

    with grpc.insecure_channel(f"{ip_address}:50051") as channel:
        stub = test_pb2_grpc.TrainingStub(channel)
        response = stub.Train(
            test_pb2.TrainRequest(
                model=onnx_model_bytes,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
        )

        model_size = len(response.model)
        print(f"Received model from server. Size: {model_size} bytes.")

        try:
            onnx_model = onnx.load_model_from_string(response.model)
            onnx.checker.check_model(onnx_model)
            print("ONNX model valid and verified.")

            print("Starting conversion from ONNX to PyTorch with 'onnx2torch'...")
            pytorch_model = convert(onnx_model)
            print("Model successfully converted to PyTorch.")

            # Remap state_dict keys to adapt to Net()
            print("Remapping layer names for PyTorch compatibility...")
            state_dict = pytorch_model.state_dict()
            new_state_dict = {}

            for k, v in state_dict.items():
                new_k = k.replace("/Conv/Conv", "").replace("/Gemm/Gemm", "")
                new_k = new_k.replace("initializers.onnx_initializer_0", "")
                new_state_dict[new_k] = v

            # Load remapped weights into a "clean" Net instance
            fixed_model = Net()
            missing, unexpected = fixed_model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {missing}, Unexpected keys: {unexpected}")

        except Exception as e:
            error_message = f"Error during model conversion or verification: {e}"
            print(error_message)
            fixed_model = model  # fallback

    print("Server reply: Train")

    model_record = ArrayRecord(fixed_model.state_dict())
    metrics = {
        "train_loss": response.avg_trainloss,
        "num-examples": response.dataset_len,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):

    print("Start Evaluating")

    batch_size = context.run_config["batch-size"]

    model = Net()

    # Extract and remap received weights
    state_dict = msg.content["arrays"].to_torch_state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k.replace("/Conv/Conv", "").replace("/Gemm/Gemm", "")
        new_k = new_k.replace("initializers.onnx_initializer_0", "")
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing}, Unexpected keys: {unexpected}")

    print("Exporting PyTorch model to ONNX format...")
    dummy_input = torch.randn(1, 1, 28, 28)
    buffer = io.BytesIO()

    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )

    onnx_model_bytes = buffer.getvalue()
    print(f"Export completed. Model size: {len(onnx_model_bytes)} bytes.")

    with grpc.insecure_channel(f"{ip_address}:50051") as channel:
        stub = test_pb2_grpc.TrainingStub(channel)
        response = stub.Test(
            test_pb2.TestRequest(model=onnx_model_bytes, batch_size=batch_size)
        )

        print(f"Received response from server: loss={response.eval_loss}, acc={response.eval_acc}")

    metrics = {
        "eval_loss": response.eval_loss,
        "eval_acc": response.eval_acc,
        "num-examples": response.dataset_len,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    print("Evaluation successfully completed.")
    return Message(content=content, reply_to=msg)