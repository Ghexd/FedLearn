"""test: A Flower / NumPy app."""

import torch
import subprocess
from typing import Iterable
from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from test.task import Net
import socket

# Create a custom strategy to send instructions to clients
class CustomFedAvg(FedAvg):
    def __init__(self, *args, num_rounds: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        custom_config = config.copy()

        if server_round == 1:
            print(f"\nRound {server_round}: Sending 'start_train' instruction to clients.\n")
            custom_config["task"] = "start_train"
            
        # Call the parent class method with the potentially modified config
        return super().configure_train(server_round, arrays, custom_config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:

        custom_config = config.copy()

        if server_round == self.num_rounds:
            print(f"\nRound {server_round}: Sending 'stop_eval' instruction to clients.\n")
            custom_config["task"] = "stop_eval"

        # Call the parent class method with the potentially modified config
        return super().configure_evaluate(server_round, arrays, custom_config, grid)


def get_host_ip():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(('8.8.8.8', 1))
    IP = s.getsockname()[0]
  except Exception:
    IP = '127.0.0.1'
  finally:
    s.close()
  return IP

IPAddr = get_host_ip()

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    print("Starting Server...")

    net = subprocess.run(['sudo', './start_net.sh', f'{IPAddr}'])
    print(f"Net script started (exit code: {net.returncode})")
    monitor = subprocess.run(['sudo', './start_monitor.sh'])
    print(f"Monitoring script started (exit code: {monitor.returncode})")

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize the custom FedAvg strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        num_rounds=num_rounds,
    )

    # Start strategy, running for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "./final_model.pt")

    net = subprocess.run(['sudo', './stop_net.sh'])
    print(f"Net script stopped (exit code: {net.returncode})")
    monitor = subprocess.run(['sudo', './stop_monitor.sh'])
    print(f"Monitoring script stopped (exit code: {monitor.returncode})")