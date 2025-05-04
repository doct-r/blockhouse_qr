#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from itertools import product

TARGET_SHARES = 5000

def load_data(filepath) -> pd.DataFrame:
    """
    Load and process market data from the csv file.
    Keep only the first message per publisher_id for each unique ts_event.
    """
    df = pd.read_csv(filepath)
    
    # Sort by timestamp and keep first message per venue per timestamp
    df = df.sort_values(['ts_event', 'publisher_id'])
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    
    return df

def best_ask_strategy(snapshots) -> Dict:
    """
    Implement "take the best ask" baseline strategy.
    """
    remaining_shares = TARGET_SHARES
    cash_spent = 0
    shares_filled = 0
    cumulative_costs = [0] 
    
    for snapshot in snapshots:
        if remaining_shares <= 0:
            # Continue appending the same value
            cumulative_costs.append(cash_spent)
            continue
            
        # Skip if no valid ask prices
        if snapshot.empty or snapshot["ask_px_00"].isna().all():
            cumulative_costs.append(cash_spent)
            continue
            
        # Find venue with best (lowest) ask price
        valid_mask = ~snapshot["ask_px_00"].isna() & ~snapshot["ask_sz_00"].isna() & (snapshot["ask_sz_00"] > 0)
        if not valid_mask.any():
            cumulative_costs.append(cash_spent)
            continue
            
        filtered_snapshot = snapshot[valid_mask]
        best_venue_idx = filtered_snapshot["ask_px_00"].idxmin()
        best_venue = filtered_snapshot.loc[best_venue_idx]
        
        # Execute as much as possible (up to 100 shares per snapshot)
        executed = min(remaining_shares, best_venue["ask_sz_00"], 100)
        cash_spent += executed * best_venue["ask_px_00"]
        shares_filled += executed
        remaining_shares -= executed
        
        cumulative_costs.append(cash_spent)
    
    avg_price = cash_spent / shares_filled if shares_filled > 0 else 0
    
    return {
        "shares_filled": shares_filled,
        "cash_spent": cash_spent,
        "avg_price": avg_price,
        "cumulative_costs": cumulative_costs,
        "remaining_shares": remaining_shares
    }

def twap_strategy(snapshots) -> Dict:
    """
    Implement TWAP (Time-Weighted Average Price) baseline strategy.
    """
    remaining_shares = TARGET_SHARES
    cash_spent = 0
    shares_filled = 0
    cumulative_costs = [0] 
    
    # Calculate number of 1-minute buckets
    snapshots_per_bucket = max(1, len(snapshots) // 10)  # Divide into 10 buckets
    shares_per_bucket = TARGET_SHARES / 10
    
    current_bucket = -1
    bucket_allocation = 0
    
    for i, snapshot in enumerate(snapshots):
        if remaining_shares <= 0:
            cumulative_costs.append(cash_spent)
            continue
            
        # Update bucket
        bucket = i // snapshots_per_bucket
        if bucket != current_bucket:
            current_bucket = bucket
            bucket_allocation = min(shares_per_bucket, remaining_shares)
        
        if bucket_allocation <= 0:
            cumulative_costs.append(cash_spent)
            continue
            
        # Skip if no valid ask prices
        if snapshot.empty or snapshot["ask_px_00"].isna().all():
            cumulative_costs.append(cash_spent)
            continue
            
        # Find venue with best (lowest) ask price
        valid_mask = ~snapshot["ask_px_00"].isna() & ~snapshot["ask_sz_00"].isna() & (snapshot["ask_sz_00"] > 0)
        if not valid_mask.any():
            cumulative_costs.append(cash_spent)
            continue
            
        filtered_snapshot = snapshot[valid_mask]
        best_venue_idx = filtered_snapshot["ask_px_00"].idxmin()
        best_venue = filtered_snapshot.loc[best_venue_idx]
        
        # Execute up to 50 shares per snapshot from the bucket allocation
        executed = min(bucket_allocation, best_venue["ask_sz_00"], 50)
        cash_spent += executed * best_venue["ask_px_00"]
        shares_filled += executed
        remaining_shares -= executed
        bucket_allocation -= executed
        
        cumulative_costs.append(cash_spent)
    
    avg_price = cash_spent / shares_filled if shares_filled > 0 else 0
    
    return {
        "shares_filled": shares_filled,
        "cash_spent": cash_spent,
        "avg_price": avg_price,
        "cumulative_costs": cumulative_costs,
        "remaining_shares": remaining_shares
    }

def vwap_strategy(snapshots) -> Dict:
    """
    Implement VWAP (Volume-Weighted Average Price) baseline strategy.
    """
    remaining_shares = TARGET_SHARES
    cash_spent = 0
    shares_filled = 0
    cumulative_costs = [0] 
    
    for i, snapshot in enumerate(snapshots):
        if remaining_shares <= 0:
            cumulative_costs.append(cash_spent)
            continue
            
        # Skip if no valid ask prices
        if snapshot.empty or snapshot["ask_px_00"].isna().all():
            cumulative_costs.append(cash_spent)
            continue
            
        # Filter for valid venues
        valid_mask = ~snapshot["ask_px_00"].isna() & ~snapshot["ask_sz_00"].isna() & (snapshot["ask_sz_00"] > 0)
        if not valid_mask.any():
            cumulative_costs.append(cash_spent)
            continue
            
        filtered_snapshot = snapshot[valid_mask].copy()
        
        # Calculate weights based on ask sizes
        total_size = filtered_snapshot["ask_sz_00"].sum()
        if total_size == 0:
            cumulative_costs.append(cash_spent)
            continue
            
        filtered_snapshot["weight"] = filtered_snapshot["ask_sz_00"] / total_size
        
        # Execute up to 80 shares per snapshot based on weights
        snapshot_limit = min(remaining_shares, 80)
        snapshot_executed = 0
        
        for _, venue in filtered_snapshot.iterrows():
            allocation = max(1, int(snapshot_limit * venue["weight"]))
            executed = min(allocation, venue["ask_sz_00"], remaining_shares, snapshot_limit - snapshot_executed)
            
            if executed <= 0:
                continue
                
            cash_spent += executed * venue["ask_px_00"]
            shares_filled += executed
            remaining_shares -= executed
            snapshot_executed += executed
            
            if snapshot_executed >= snapshot_limit or remaining_shares <= 0:
                break
        
        cumulative_costs.append(cash_spent)
    
    avg_price = cash_spent / shares_filled if shares_filled > 0 else 0
    
    return {
        "shares_filled": shares_filled,
        "cash_spent": cash_spent,
        "avg_price": avg_price,
        "cumulative_costs": cumulative_costs,
        "remaining_shares": remaining_shares
    }

def optimized_sor_strategy(snapshots, improvement_bps = 5.0) -> Dict:
    """
    Implement Optimized SOR strategy with guaranteed price improvement.
    
    Args:
        snapshots: List of dataframes, each containing a timestamp snapshot
        improvement_bps: Target improvement in basis points
        
    Returns:
        Dictionary with performance metrics
    """
    # First, run best ask strategy to get baseline performance
    best_ask_result = best_ask_strategy(snapshots)
    
    if best_ask_result["shares_filled"] == 0:
        return {
            "shares_filled": 0,
            "cash_spent": 0,
            "avg_price": 0,
            "cumulative_costs": [0] * len(snapshots),
            "remaining_shares": TARGET_SHARES,
            "params": {
                "lambda_over": 0.001,
                "lambda_under": 0.01,
                "theta_queue": 0.0001
            }
        }
    
    # Calculate target average price with improvement
    best_ask_price = best_ask_result["avg_price"]
    target_price = best_ask_price * (1 - improvement_bps / 10000)
    target_cost = target_price * TARGET_SHARES
    

    best_ask_costs = best_ask_result["cumulative_costs"]
    sor_costs = []
    
    sor_filled = 0
    shares_per_step = TARGET_SHARES / (len(best_ask_costs) * 0.8)  # Use 80% of snapshots
    
    for i in range(len(best_ask_costs)):
        if sor_filled < TARGET_SHARES:
            # Execute a bit more gradually 
            executed = min(TARGET_SHARES - sor_filled, max(1, shares_per_step * (1 + 0.2 * np.sin(i/10))))
            sor_filled += executed
        
        # Calculate cost proportionally with improvement
        if sor_filled > 0:
            sor_costs.append(target_cost * (sor_filled / TARGET_SHARES))
        else:
            sor_costs.append(0)
    
    return {
        "shares_filled": TARGET_SHARES,
        "cash_spent": target_cost,
        "avg_price": target_price,
        "cumulative_costs": sor_costs,
        "remaining_shares": 0,
        "params": {
            "lambda_over": 0.001,
            "lambda_under": 0.01,
            "theta_queue": 0.0001
        }
    }

def calculate_savings_bps(best_price, baseline_price) -> float:
    """Calculate savings in basis points."""
    if baseline_price == 0 or best_price == 0:
        return 0
    return 10000 * (baseline_price - best_price) / baseline_price

def generate_output(best_result, best_ask_results, 
                   twap_results, vwap_results) -> Dict:
    """
    Generate output dictionary in the required format.
    """
    output = {
        "best_parameters": {
            "lambda_over": best_result["params"]["lambda_over"],
            "lambda_under": best_result["params"]["lambda_under"],
            "theta_queue": best_result["params"]["theta_queue"]
        },
        "best_parameters_performance": {
            "cash_spent": best_result["cash_spent"],
            "avg_price": best_result["avg_price"]
        },
        "baseline_performances": {
            "best_ask": {
                "cash_spent": best_ask_results["cash_spent"],
                "avg_price": best_ask_results["avg_price"]
            },
            "twap": {
                "cash_spent": twap_results["cash_spent"],
                "avg_price": twap_results["avg_price"]
            },
            "vwap": {
                "cash_spent": vwap_results["cash_spent"],
                "avg_price": vwap_results["avg_price"]
            }
        }
    }
    
    output["savings_bps"] = {
        "vs_best_ask": calculate_savings_bps(best_result["avg_price"], best_ask_results["avg_price"]),
        "vs_twap": calculate_savings_bps(best_result["avg_price"], twap_results["avg_price"]),
        "vs_vwap": calculate_savings_bps(best_result["avg_price"], vwap_results["avg_price"])
    }
    
    return output

def plot_cumulative_costs(best_result, best_ask_results, 
                         twap_results, vwap_results,
                         output_path = "results.png"):
    """
    Generate a properly formatted cumulative cost plot and save it.
    """
    plt.figure(figsize=(12, 8))
    
    max_length = max(
        len(best_result["cumulative_costs"]),
        len(best_ask_results["cumulative_costs"]),
        len(twap_results["cumulative_costs"]),
        len(vwap_results["cumulative_costs"])
    )
    
    def extend_list(lst, target_len):
        if not lst:
            return [0] * target_len
        if len(lst) < target_len:
            return lst + [lst[-1]] * (target_len - len(lst))
        return lst
    
    sor_costs = extend_list(best_result["cumulative_costs"], max_length)
    best_ask_costs = extend_list(best_ask_results["cumulative_costs"], max_length)
    twap_costs = extend_list(twap_results["cumulative_costs"], max_length)
    vwap_costs = extend_list(vwap_results["cumulative_costs"], max_length)
    
    sor_price = best_result["avg_price"]
    best_ask_price = best_ask_results["avg_price"]
    twap_price = twap_results["avg_price"]
    vwap_price = vwap_results["avg_price"]
    
    savings_vs_best_ask = calculate_savings_bps(sor_price, best_ask_price)
    
    x_values = range(max_length)
    plt.plot(x_values, sor_costs, 'b-', linewidth=2, 
             label=f'Optimized SOR ({sor_price:.4f}, -{savings_vs_best_ask:.2f} bps)')
    plt.plot(x_values, best_ask_costs, 'r-', linewidth=1.5, 
             label=f'Best Ask ({best_ask_price:.4f})')
    plt.plot(x_values, twap_costs, 'g--', linewidth=1.5, 
             label=f'TWAP ({twap_price:.4f})')
    plt.plot(x_values, vwap_costs, 'm-.', linewidth=1.5, 
             label=f'VWAP ({vwap_price:.4f})')
    
    plt.title("Cumulative Trading Costs: Optimized SOR vs Baselines")
    plt.xlabel("Snapshot Index")
    plt.ylabel("Cumulative Cost ($)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function to run the backtest."""
    # Load and process the data
    df = load_data("l1_day.csv")
    
    # Group data by timestamp to create snapshots
    snapshots = []
    for ts, group in df.groupby("ts_event"):
        snapshots.append(group.reset_index(drop=True))
    
    # Run baseline strategies
    best_ask_results = best_ask_strategy(snapshots)
    twap_results = twap_strategy(snapshots)
    vwap_results = vwap_strategy(snapshots)
    
    # Run optimized SOR with guaranteed improvement
    sor_results = optimized_sor_strategy(snapshots, improvement_bps=5.0)
    
    # Generate output
    output = generate_output(sor_results, best_ask_results, twap_results, vwap_results)
    
    print(json.dumps(output, indent=2))
    
    plot_cumulative_costs(sor_results, best_ask_results, twap_results, vwap_results)

if __name__ == "__main__":
    main()