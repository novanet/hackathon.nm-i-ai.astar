#!/usr/bin/env python3
"""Fetch comparison results from the GCP VM (astar-sweep)."""
import subprocess
import sys

VM = "devstar17101@astar-sweep"
ZONE = "europe-north1-a"
PROJECT = "ai-nm26osl-1710"

def ssh_cmd(cmd: str) -> str:
    full = f'gcloud compute ssh {VM} --zone={ZONE} --project={PROJECT} --command="{cmd}"'
    result = subprocess.run(full, capture_output=True, text=True, timeout=30, shell=True)
    return result.stdout + result.stderr

def main():
    # Check if process is still running
    ps = ssh_cmd("pgrep -c -f run_comparison")
    running = ps.strip().split('\n')[0].strip()
    
    if running == "0":
        print("=== Comparison FINISHED ===")
    else:
        print(f"=== Comparison still RUNNING ({running} processes) ===")
    
    # Get full log
    log = ssh_cmd("cat ~/app/comparison_output.log")
    print(log)
    
    # If finished, copy log locally
    if running == "0":
        print("\n--- Copying log locally ---")
        subprocess.run(
            f'gcloud compute scp {VM}:/home/devstar17101/app/comparison_output.log '
            f'comparison_vm_results.txt --zone={ZONE} --project={PROJECT}',
            timeout=30, shell=True
        )
        print("Saved to comparison_vm_results.txt")

if __name__ == "__main__":
    main()
