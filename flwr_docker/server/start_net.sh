#!/bin/bash

PID_FILE="/tmp/tshark_capture.info"

# IP check
TARGET_IP=$1
if [ -z "$TARGET_IP" ]; then
  echo "Error: No IP address provided."
  exit 1
fi

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(head -n 1 "$PID_FILE")
    # Check if the process still exists
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Error: A capture session is already active with PID ${OLD_PID}."
        exit 1
    else
        # The file exists but the process does not, so clean up
        rm -f "$PID_FILE"
    fi
fi


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FILE_NAME="capture_${TIMESTAMP}.pcap"
OUTPUT_PATH="/tmp/${FILE_NAME}"

tshark -i any -w "${OUTPUT_PATH}" -f "host ${TARGET_IP} and (port 9091 or port 9092 or port 9093)" & 
echo "$!" >> "${PID_FILE}"

echo "${OUTPUT_PATH}" >> "${PID_FILE}"

echo "Network analysis started"

