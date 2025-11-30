#!/bin/bash

PID_TRACK_FILE="simpleperf.pid"

if [ ! -f "$PID_TRACK_FILE" ]; then
    echo "Error: file $PID_TRACK_FILE not found"
    exit 1
fi

SIMPLEPERF_PID=$(cat "$PID_TRACK_FILE")

if ps -p "$SIMPLEPERF_PID" > /dev/null 2>&1; then
    sudo kill "$SIMPLEPERF_PID"
    rm -f "$PID_TRACK_FILE"
    echo "simpleperf killed."
else
    echo "No active process with pids: $SIMPLEPERF_PID."
    rm -f "$PID_TRACK_FILE"
fi
