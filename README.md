## Overview

This repository is based on this [Flower](https://github.com/adap/flower/tree/main/examples/embedded-devices) example and extends it to run a Federated Learning demo on embedded devices.

The main goal of this work is to use a well-established Federated Learning framework and execute **performance measurements** across a variety of devices. The demo has been tested on laptops (Linux Mint 22.1), a Raspberry Pi 400, and a smartphone running Ubuntu Touch.

The main additions include performance-monitoring scripts and various fixes.

There are two main folders:

- **flwr_baremetal** - intended for environments where Flower is installed directly on the target machines.
- **flwr_docker** - intended for environments using Docker containers.

Inside the Docker folder, there is also a variant of the demo (in `test_onnx`) that attempts to extend the Flower framework to support scenarios where clients cannot run Flower directly. The idea is that the Flower server communicates with *fake clients* running on the server, while these fake clients communicate with the real devices (on other machines) using the gRPC protocol.

![Extended Framework Schema](assets/Architecture_schema.drawio.png "Extended Framework Schema")

## How to Run the Demo

**NOTE**: To run the measurement scripts correctly, make sure you run the Flower commands as **administrator** as some tools require elevated privileges.

For detailed instructions, refer to the `instructions.txt` file in the **baremetal** folder.

### Baremetal Version

1. Copy the performance script to the server. Open a terminal in the same directory and run the Flower **SuperLink** command.

2. Copy the performance script to each client. Open a terminal in the same directory and run the Flower **SuperNode** command *(repeat for each client you want to include)*.

3. Navigate to the folder containing the `pyproject.toml` file and execute Flower **run** command.

### Docker Version

1. Run the following script: `compose_with_privileges.sh`.

2. Navigate to the folder containing the `pyproject.toml` file and execute Flower **run** command.

### ONNX Version

1. Change project.toml as follow:

```toml
[tool.flwr.app.components]
serverapp = "test_onnx.server_side.server_app:app"
clientapp = "test_onnx.server_side.client_app:app"
```

2. Start `real_client.py` in `client_side` folder.

3. In `server_side/client_app` edit `ip_address` with real client address. 

4. Run `compose_with_privileges.sh`.




