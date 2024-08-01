#!/bin/bash
# Usage: ./Measure_Ram_use.sh <PID>

# Check if a PID is provided
if [ $# -eq 0 ]; then
    echo "Please provide a PID as an argument."
    exit 1
fi

PID=$1  

LOG_FILE="memory_usage_${PID}.log"

echo "Logging memory usage for PID: $PID to $LOG_FILE"

while true; do
    # Append current date, time, RSS and VSZ values to the log file
    echo "$(date '+%Y-%m-%d %H:%M:%S'), $(ps -o rss=,vsz= -p $PID)" >> $LOG_FILE
    sleep 1  # Delay for 1 second before the next record
done
