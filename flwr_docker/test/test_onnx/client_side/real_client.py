from concurrent import futures
import io
import logging
import threading

import grpc
import test_pb2
import test_pb2_grpc

import task
import torch
import onnx
from onnx2torch import convert

dataset_path = "./fashionmnist_part_1.npz"

class Training(test_pb2_grpc.TrainingServicer):
    def Train(self, request, context):
        onnx_model, loss, len_dataset = train(request.model, request.local_epochs, request.learning_rate, request.batch_size)
        return test_pb2.TrainReply(model = onnx_model, avg_trainloss = loss, dataset_len = len_dataset)
    def Test(self, request, context):
        loss, acc, len_dataset = evaluate(request.model, request.batch_size)
        return test_pb2.TestReply(eval_loss = loss, eval_acc = acc, dataset_len = len_dataset)

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_TrainingServicer_to_server(Training(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

def train(model_bytes, local_epochs, learning_rate, batch_size):

    print("Start Training")

    model_size = len(model_bytes)
    print(f"Received model from client. Size: {model_size} bytes.")

    try:
        onnx_model = onnx.load_model_from_string(model_bytes)
        onnx.checker.check_model(onnx_model)
        print("ONNX model valid and verified.")

        print("Starting conversion from ONNX to PyTorch with 'onnx2torch'...")
        pytorch_model = convert(onnx_model)
        print("Model successfully converted to PyTorch.")

    except Exception as e:
        error_message = f"Error during model conversion or verification: {e}"
        print(error_message)
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.to(device)

    trainloader, _ = task.load_data_from_disk(dataset_path, batch_size)
    train_loss = task.train(pytorch_model, trainloader, local_epochs, learning_rate, device)

    print("Exporting PyTorch model to ONNX format...")
    
    onnx_export_lock = threading.Lock()

    with onnx_export_lock:
        buffer = io.BytesIO()

        # Move to CPU to avoid type errors
        pytorch_model_cpu = pytorch_model.to("cpu").eval()
        dummy_input = torch.randn(1, 1, 28, 28, device="cpu")
        pytorch_model.eval()
        
        torch.onnx.export(
            pytorch_model_cpu,
            dummy_input,
            buffer,
            input_names=['input'],
            output_names=['output'],
            opset_version=12
        )

    onnx_model_bytes = buffer.getvalue()
    print("Export successfully completed.")

    return onnx_model_bytes, train_loss, len(trainloader.dataset)


def evaluate(model_bytes, batch_size):
    
    print("Start Evaluating")

    try:
        onnx_model = onnx.load_model_from_string(model_bytes)
        onnx.checker.check_model(onnx_model)
        print("ONNX model valid and verified.")

        print("Starting conversion from ONNX to PyTorch with 'onnx2torch'...")
        pytorch_model = convert(onnx_model)
        print("Model successfully converted to PyTorch.")

    except Exception as e:
        error_message = f"Error during model conversion or verification: {e}"
        print(error_message)
        raise e
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.to(device)

    _, valloader = task.load_data_from_disk(dataset_path, batch_size)
    eval_loss, eval_acc = task.test(pytorch_model, valloader, device)

    return eval_loss, eval_acc, len(valloader.dataset)

if __name__ == "__main__":
    logging.basicConfig()
    serve()