import torch
from typing import Iterable
from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task_onnx import Net

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


# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    print("Starting Server...")

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize the custom FedAvg strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        num_rounds=num_rounds, # Pass total rounds to the strategy
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
    torch.save(state_dict, "/home/final_model.pt")