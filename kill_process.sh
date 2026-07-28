#!/bin/bash

echo "Stopping vLLM and settings processes..."

for pid in $(pgrep -f 'vllm|settings')
do
    echo "Killing PID $pid"
    kill -9 "$pid" 2>/dev/null
done

echo "Cleaning remaining GPU processes..."

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null)
    echo "$cmd" | grep -qi "vllm"
    if [ $? -eq 0 ]; then
        echo "Killing GPU process $pid"
        kill -9 "$pid"
    fi
done

echo "Done."