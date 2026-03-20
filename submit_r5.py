"""Submit predictions for R5 (active round)."""
import time
from astar.client import get_round_detail, get_budget, simulate_grid
from astar.submit import submit_round

R5_ID = "fd3c92ff-3178-4dc9-8d9b-acf389b3982b"

def main():
    detail = get_round_detail(R5_ID)
    n_seeds = len(detail.get("initial_states", []))
    
    budget = get_budget()
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} used")
    
    # Run 9-viewport grid coverage for each seed (9 queries per seed = 45 total)
    for seed in range(n_seeds):
        print(f"\n--- Querying seed {seed} ---")
        results = simulate_grid(R5_ID, seed, delay=0.3)
        print(f"  Got {len(results)} viewport observations")
        time.sleep(0.5)
    
    budget = get_budget()
    print(f"\nBudget after queries: {budget['queries_used']}/{budget['queries_max']} used")
    
    # Build predictions and submit
    print("\n--- Submitting predictions ---")
    scores = submit_round(R5_ID)
    print(f"\nResults: {scores}")

if __name__ == "__main__":
    main()
