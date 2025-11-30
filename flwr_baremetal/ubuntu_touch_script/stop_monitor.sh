#!/bin/bash

# Find the PIDs of nmon processes started with the -f flag.
NMON_PIDS=$(pgrep -f "nmon -f" | grep -v $$)

if [ -z "$NMON_PIDS" ]; then
  echo "No 'nmon' data-capture processes were found."
else
  # Send USR2 signal to gracefully shut down nmon.
  kill -USR2 $NMON_PIDS
  if [ $? -eq 0 ]; then
    echo "Termintating nmon process."
    sleep 1 
  else
    echo "Error: Failed to send signal to nmon."
  fi
fi


# Stop tools and generate report
MONITORING_PIDS_FILE="monitoring_pids.txt"
TEMP_DIR="monitoring_temp_logs"
REPORTS_DIR="monitoring_reports"

if [ ! -f "$MONITORING_PIDS_FILE" ]; then
  echo "Monitoring PIDs file not found: $MONITORING_PIDS_FILE"
  exit 1
fi

# Stop all background monitoring tools by reading their PIDs from the file.
while read -r pid; do
  if ps -p "$pid" > /dev/null; then
    kill -INT "$pid"
  fi
done < "$MONITORING_PIDS_FILE"

# Wait for processes to terminate and write their final logs.
sleep 2

if [ ! -d "$TEMP_DIR" ]; then
  echo "Temporary log directory not found: $TEMP_DIR"
  exit 1
fi

mkdir -p "$REPORTS_DIR"

mv *.nmon $REPORTS_DIR/

TARGET_PIDS="../Documents/target_pid.txt"

TARGET_PIDS=$(cat "$TARGET_PIDS")

for PID in $TARGET_PIDS; do

  SUB_DIR="$REPORTS_DIR/pid_${PID}"
  mkdir -p $SUB_DIR

  # Get the command line associated with the PID.
  COMMAND_LINE=$(grep "^$PID:" "$TEMP_DIR/pid_commands.txt" | cut -d':' -f2-)

  # Create the Pidstat report
  if [ -f "$TEMP_DIR/pidstat_${PID}.log" ]; then
    PIDSTAT_CSV_FILE="$SUB_DIR/pidstat_report_${PID}.csv"
    {
      echo "read_kB/s,write_kB/s"
      
      awk '
      NR > 3 {
        # If we have a previous line stored, print it now.
        # so it doesnt print the last row that contains the avg value
        if (previous_line) {
          print previous_line;
        }
        
        # Now, process the current line and store it for the next iteration.
        gsub(",", ".", $4);
        gsub(",", ".", $5);
        previous_line = $4 "," $5;
      }
      ' "$TEMP_DIR/pidstat_${PID}.log"
    } > "$PIDSTAT_CSV_FILE"
  fi  
done

# Clean up temporary files.
rm -rf "$TEMP_DIR"
rm -f "$MONITORING_PIDS_FILE"

echo "pidstat monitoring stopped and reports generated in '$REPORTS_DIR' directory."

exit 0








