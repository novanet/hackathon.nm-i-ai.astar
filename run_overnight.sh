#!/bin/bash
cd ~/app
source ~/venv/bin/activate

echo "[$(date)] Overnight orchestrator wrapper started"

# Wait for the comparison job to finish first
while pgrep -f run_comparison.py > /dev/null 2>&1; do
    echo "[$(date)] Waiting for run_comparison.py to finish..."
    sleep 120
done
echo "[$(date)] Comparison done!"

# Kill the big_unet_queue waiter if running (we'll re-launch via overnight.py)
pkill -f start_big_unet.sh 2>/dev/null
pkill -f "big_unet_queue" 2>/dev/null

# Grab comparison results
echo "[$(date)] Comparison results:"
cat comparison_output.log

# Now run the full overnight orchestrator
echo "[$(date)] Starting overnight.py..."
python -u overnight.py

echo "[$(date)] Overnight complete!"
