#!/bin/bash
PID_FILE="../target_pid.txt"                   
OUTPUT_FILE="simpleperf_output.txt"  
SIMPLEPERF_PATH="./simpleperf"  
PID_TRACK_FILE="simpleperf.pid"  

if [ ! -f "$PID_FILE" ]; then
    echo "Error: file $PID_FILE does not exist!"
    exit 1
fi

if [ ! -x "$SIMPLEPERF_PATH" ]; then
    echo "Error: simpleperf not found or not executable in $SIMPLEPERF_PATH"
    exit 1
fi

PIDS=$(tr '\n' ',' < "$PID_FILE" | sed 's/,$//')

if [ -z "$PIDS" ]; then
    echo "Error: no PID found in $PID_FILE"
    exit 1
fi

echo "Start simpleperf for PID: $PIDS"

sudo "$SIMPLEPERF_PATH" stat -e instructions -p "$PIDS" > "$OUTPUT_FILE" 2>&1 &
SIMPLEPERF_PID=$!

echo "$SIMPLEPERF_PID" > "$PID_TRACK_FILE"

echo "simpleperf started (PID: $SIMPLEPERF_PID)"

