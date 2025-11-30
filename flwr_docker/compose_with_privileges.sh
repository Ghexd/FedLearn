#!/bin/bash

# Temporarily Adjust perf_event_paranoid
sudo sysctl -w kernel.perf_event_paranoid=-1

sudo docker compose up -d

