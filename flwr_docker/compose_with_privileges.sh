#!/bin/bash

# Temporarily Adjust perf_event_paranoid
sudo sysctl -w kernel.perf_event_paranoid=-1

# If you want to preserve the state replace "compose" with "compose_statefull"
sudo docker compose up -d
