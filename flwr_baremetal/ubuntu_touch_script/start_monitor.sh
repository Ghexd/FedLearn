#!/bin/bash

# Start nmon to capture data to a file, sampling every 3 seconds.
nmon -f -s 3 -t

echo "nmon started"

# perf and pidstat startup
PROCESS_NAME="flower"
MONITORING_PIDS_FILE="monitoring_pids.txt"
TEMP_DIR="monitoring_temp_logs"
TARGET_PIDS="../Documents/target_pid.txt"

# Find PIDs of the process to monitor.
PIDS=$(pgrep -f "$PROCESS_NAME" | grep -v $$)

if [ -z "$PIDS" ]; then
  echo "No process found matching name: $PROCESS_NAME"
  exit 1
fi

# Clean up previous monitoring files.
rm -f "$MONITORING_PIDS_FILE"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

for PID in $PIDS; do
  PIDSTAT_LOG="$TEMP_DIR/pidstat_${PID}.log"

  COMMAND_LINE=$(ps -p "$PID" -o cmd=)
  echo "$PID:$COMMAND_LINE" >> "$TEMP_DIR/pid_commands.txt"

  # Start pidstat monitoring in the background.
  pidstat -d -p "$PID" 1 > "$PIDSTAT_LOG" &
  echo "$!" >> "$MONITORING_PIDS_FILE"
done

echo "$PIDS" > "$TARGET_PIDS"

echo "pidstat started."
