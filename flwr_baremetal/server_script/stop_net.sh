#!/bin/bash

REAL_USER=$(logname)

PID_FILE="/tmp/tshark_capture.info"

DESTINATION_DIR="./tshark_captures"


if [ ! -f "$PID_FILE" ]; then
    echo "No active capture session found (file ${PID_FILE} does not exist)."
    exit 1
fi

TSHARK_PID=$(head -n 1 "$PID_FILE")
CAPTURE_FILE_PATH=$(tail -n 1 "$PID_FILE")

if [ -z "$TSHARK_PID" ] || [ -z "$CAPTURE_FILE_PATH" ]; then
    echo "Error: The status file ${PID_FILE} is corrupt or empty."
    rm -f "$PID_FILE"
    exit 1
fi

if ps -p "$TSHARK_PID" > /dev/null; then
    sudo kill "${TSHARK_PID}"
    sleep 2
    echo "tshark process terminated."
else
    echo "Warning: Process with PID ${TSHARK_PID} was not found. It might have already been terminated."
fi

if [ -f "$CAPTURE_FILE_PATH" ]; then

    mkdir -p "$DESTINATION_DIR"

    sudo cp "$CAPTURE_FILE_PATH" "$DESTINATION_DIR/"
    
    BASENAME_CAPTURE_FILE=$(basename "$CAPTURE_FILE_PATH")
    FINAL_PCAP_PATH="${DESTINATION_DIR}/${BASENAME_CAPTURE_FILE}"
    FINAL_CSV_PATH="${FINAL_PCAP_PATH}.csv"

    sudo tshark -r "$FINAL_PCAP_PATH" -q -z conv,tcp | LC_NUMERIC=C awk '
    BEGIN {
        OFS=",";
        print "source_A,destination_B,frames_A_to_B,bytes_A_to_B,frames_B_to_A,bytes_B_to_A,tot_frames,tot_bytes,relative_start,duration,bytes/s_A_to_B,bytes/s_B_to_A"
    }
    / <-> / {
        # Remove thousand separators for calculations
        bytes_ab = $8;  gsub(/[.,]/, "", bytes_ab);
        bytes_ba = $5;  gsub(/[.,]/, "", bytes_ba);
        bytes_tot = $11; gsub(/[.,]/, "", bytes_tot);

        # Convert decimal notation for awk (comma -> dot)
        start_rel = $13; gsub(/,/, ".", start_rel);
        duration = $14; gsub(/,/, ".", duration);
        
        # Calculate Bytes/second, avoiding division by zero
        bps_ab = 0;
        bps_ba = 0;
        if (duration > 0) {
            bps_ab = bytes_ab / duration;
            bps_ba = bytes_ba / duration;
        }
        
        # Print the formatted fields
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%.6f,%.6f,%.2f,%.2f\n", $1, $3, $4, bytes_ab, $7, bytes_ba, $10, bytes_tot, start_rel, duration, bps_ab, bps_ba
    }' > "$FINAL_CSV_PATH"

    sudo chown -R "$REAL_USER:$REAL_USER" "$DESTINATION_DIR"

else
    echo "Error: Capture file ${CAPTURE_FILE_PATH} not found!"
fi

rm -f "$PID_FILE"

# sudo rm -f "$CAPTURE_FILE_PATH"