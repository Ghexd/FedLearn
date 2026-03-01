import torch
import subprocess
from typing import Iterable, Dict, List, Tuple, Optional, Union
from flwr.common import Scalar
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import socket

from test_client_balancing.task_apnea import Net


class RichPoorStrategy(FedAvg):
    def __init__(self, *args, num_rounds: int, dataset_threshold: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.dataset_threshold = dataset_threshold
        
        self.client_sizes: Dict[str, int] = {}
        self.rich_clients: List[str] = []
        self.poor_clients: List[str] = []
        
        # Mapping receivining_client to sending_client
        self.pairs: Dict[str, str] = {}
        
        # List of poor clients left alone (not paired)
        self.solo_poor_clients: List[str] = []
        
        self.rich_models_cache: Dict[str, ArrayRecord] = {}

        # Store the final aggregated weights from the last round
        self.final_arrays: Optional[ArrayRecord] = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        # Round 1: discovery
        if server_round == 1:
            print(f"\n[Round {server_round}] Discovery Phase: Asking clients for dataset size.")
            cfg = config.copy()
            cfg["task"] = "report_size"
            cfg["start_monitoring"] = "true"
            return super().configure_train(server_round, arrays, cfg, grid)

        messages = []
        
        # Round 2: transfer (Even Rounds)
        # In this phase training is done by all rich clients (train_excess) 
        # and poor clients that act as "mentors" for other poor clients (train_standard)
        if server_round % 2 == 0:
            print(f"\n[Round {server_round}] Transfer Phase: Preparing models for transfer.")
            
            # Rich clients train on excess data
            for rich_id in self.rich_clients:
                print(f" -> Rich Client {rich_id} training on excess data.")
                cfg = config.copy()
                cfg["task"] = "train_excess"
                record = RecordDict({"arrays": arrays, "config": cfg})
                msg = Message(
                    content=record,
                    dst_node_id=int(rich_id),
                    message_type="train",
                    group_id=str(server_round),
                )
                messages.append(msg)
            
            # Poor clients paired with other poor clients must generate a model for their partner
            poor_sources = [src for src in self.pairs.values() if src in self.poor_clients]
            # Remove duplicates
            poor_sources = list(set(poor_sources))

            for poor_src_id in poor_sources:
                print(f" -> Poor Client {poor_src_id} training as Source for Peer.")
                cfg = config.copy()
                # A poor client has no "excess", so it uses its entire dataset
                # Therefore we use "fine_tune", which uses the full dataset
                cfg["task"] = "fine_tune"
                
                record = RecordDict({"arrays": arrays, "config": cfg})
                msg = Message(
                    content=record,
                    dst_node_id=int(poor_src_id),
                    message_type="train",
                    group_id=str(server_round),
                )
                messages.append(msg)
            
            return messages

        # Round 3: aggregation (Odd Rounds > 1)
        else:
            print(f"\n[Round {server_round}] Aggregation Phase.")
            
            # Rich clients (standard training)
            for rich_id in self.rich_clients:
                cfg = config.copy()
                cfg["task"] = "train_standard"
                record = RecordDict({"arrays": arrays, "config": cfg})
                msg = Message(
                    content=record,
                    dst_node_id=int(rich_id),
                    message_type="train",
                    group_id=str(server_round),
                )
                messages.append(msg)
                
            # Paired Poor clients (fine-tuning)
            for poor_id, partner_id in self.pairs.items():
                if partner_id in self.rich_models_cache:
                    print(f" -> Poor {poor_id} fine-tuning on model from {partner_id}")
                    cfg = config.copy()
                    cfg["task"] = "fine_tune"
                    rich_model_arrays = self.rich_models_cache[partner_id]
                    record = RecordDict({"arrays": rich_model_arrays, "config": cfg})
                    msg = Message(
                        content=record,
                        dst_node_id=int(poor_id),
                        message_type="train",
                        group_id=str(server_round),
                    )
                    messages.append(msg)
                else:
                    print(f"WARNING: Model from partner {partner_id} not found for {poor_id}")

            # Solo Poor clients (standard training)
            for solo_id in self.solo_poor_clients:
                print(f" -> Solo Poor {solo_id} training standard (no partner).")
                cfg = config.copy()
                # A solo poor client uses all its data
                cfg["task"] = "fine_tune"  # "fine_tune" uses the FULL dataset
                record = RecordDict({"arrays": arrays, "config": cfg})
                msg = Message(
                    content=record,
                    dst_node_id=int(solo_id),
                    message_type="train",
                    group_id=str(server_round),
                )
                messages.append(msg)

            return messages

    def aggregate_train(
        self,
        server_round: int,
        results: List[Message],
        failures: List[Union[Tuple[Message, BaseException], BaseException]] = None,
    ) -> Tuple[Optional[ArrayRecord], Dict[str, Scalar]]:
        
        if failures is None:
            failures = []

        # Round 1: pairing logic
        if server_round == 1:
            print(f"Aggregating Round 1: Analyzing Dataset Sizes...")
            
            for response in results:
                node_id = response.metadata.src_node_id
                metrics = response.content.get("metrics")
                if metrics and metrics.get("num-examples") is not None:
                    self.client_sizes[str(node_id)] = int(metrics.get("num-examples"))

            if not self.client_sizes:
                return None, {}

            self.rich_clients = []
            self.poor_clients = []
            self.solo_poor_clients = []
            self.pairs = {}

            # Split Rich and Poor clients
            for node_id, size in self.client_sizes.items():
                if size >= self.dataset_threshold:
                    self.rich_clients.append(node_id)
                else:
                    self.poor_clients.append(node_id)

            self.rich_clients.sort(key=lambda x: self.client_sizes[x], reverse=True)
            self.poor_clients.sort(key=lambda x: self.client_sizes[x], reverse=True)

            print(f"[INFO] Rich: {len(self.rich_clients)} | Poor: {len(self.poor_clients)}")
            
            available_rich = self.rich_clients.copy()
            available_poor = self.poor_clients.copy()

            # Pair Rich with Poor while Rich clients are available
            while available_rich and available_poor:
                r = available_rich.pop(0)
                p = available_poor.pop(0)
                self.pairs[p] = r
            
            # Pair remaining Poor clients with each other
            while len(available_poor) >= 2:
                p1 = available_poor.pop(0)
                p2 = available_poor.pop(0)
                
                # Pair p1 to p2 but also p2 to p1
                self.pairs[p1] = p2
                self.pairs[p2] = p1
                print(f"   -> Poor-to-Poor Pair: {p1} <-> {p2}")

            # One Poor client left alone
            if available_poor:
                solo = available_poor.pop(0)
                self.solo_poor_clients.append(solo)
                print(f"   -> Solo Poor Client: {solo}")

            print(f"[STATUS] Pairs Map: {self.pairs}")
            return None, {}

        # Round 2: caching
        if server_round % 2 == 0:
            print(f"Aggregating Transfer Round: Caching {len(results)} models.")
            self.rich_models_cache.clear()
            for response in results:
                node_id = str(response.metadata.src_node_id)
                self.rich_models_cache[node_id] = response.content["arrays"]
            return None, {}

        # Round 3: standard aggregation
        else:
            print("Aggregating Standard Round: Updating Global Model.")
            
            aggregated_arrays, metrics = super().aggregate_train(server_round, results)
            
            if aggregated_arrays is not None:
                self.final_arrays = aggregated_arrays
                
            return aggregated_arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        # Only evaluate in non-preparation rounds (round 1 and even rounds)
        if (server_round == 1 or server_round % 2 == 0) and (server_round != self.num_rounds):
            return []

        custom_config = config.copy()
        if server_round == self.num_rounds:
            custom_config["task"] = "stop_eval"

        return super().configure_evaluate(server_round, arrays, custom_config, grid)



def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
    except Exception:
        ip_addr = "127.0.0.1"
    finally:
        s.close()
    return ip_addr

IPAddr = get_local_ip()

app = ServerApp()

print(f"IP ADDRESS: {IPAddr}")

@app.main()
def main(grid: Grid, context: Context) -> None:

    print("Starting Server...")
    subprocess.run([f'./start_net.sh', f'{IPAddr}'])
    subprocess.run(['./start_monitor.sh'])

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = (context.run_config["num-server-rounds"] * 2) + 1

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = RichPoorStrategy(
        fraction_evaluate=fraction_evaluate,
        num_rounds=num_rounds,
        
        # Change this in client_app too
        dataset_threshold=20000,
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Update global_model with the weights trained by the strategy
    if strategy.final_arrays is not None:
        print("\nUpdating global_model with trained weights...")
        try:
            # Convert ArrayRecord back to PyTorch Tensors
            state_dict = {}
            for k, v in strategy.final_arrays.items():
                
                state_dict[k] = torch.from_numpy(v.numpy())
            
            global_model.load_state_dict(state_dict)
            print("Model weights updated successfully.")
        except Exception as e:
            print(f"Error updating global_model weights: {e}")
    else:
        print("\n[WARNING] No trained weights found in strategy. Saving initial/random model.")

    print("\nSaving final model to disk...")
    try:
        torch.save(global_model.state_dict(), "./final_model.pt")
    except Exception as e:
        print(f"Error saving model: {e}")

    subprocess.run(['./stop_net.sh'])
    subprocess.run(['./stop_monitor.sh'])