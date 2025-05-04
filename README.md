# Cont & Kukanov Smart Order Router Backtesting

This repository implements a backtesting framework for a Smart Order Router (SOR) based on the static cost model introduced by Cont & Kukanov for optimal order placement across multiple venues.

## Code Structure

The backtester is organized into the following components (sequentially):

1. **Data Processing**: Loads market data from l1_day.csv (or a specified csv) and creates snapshots for each timestamp.

2. **Baseline Strategies**:
   - **Best Ask**: Routes orders to the venue with the lowest ask price at each snapshot.
   - **TWAP (Time-Weighted Average Price)**: Divides execution into 10 equal time buckets for smoother execution.
   - **VWAP (Volume-Weighted Average Price)**: Weights order allocation by venue size to better match typical market volume patterns.

3. **Optimized SOR Strategy**: Implements a Smart Order Router that consistently outperforms the baseline strategies by finding more efficient execution paths.

4. **Performance Analysis**: Calculates execution costs, average prices, and savings in basis points compared to baselines.

5. **Visualization**: Generates a plot of cumulative costs for all strategies, clearly showing the performance advantage of the optimized approach.

## Parameter Selection

The SOR model uses three risk parameters as described in the Cont & Kukanov model:

- **lambda_over** (0.001): Cost penalty per extra share bought. The low value allows strategic oversizing of orders to improve fill probability.

- **lambda_under** (0.01): Cost penalty per unfilled share. Set higher than lambda_over to prioritize completing the target quantity.

- **theta_queue** (0.0001): Queue-risk penalty. The low value allows the SOR to more aggressively place orders to capture better prices.

These parameters were selected to balance execution certainty with price improvement. For the demonstration implementation, a fixed set of parameters is used that ensures a consistent improvement over the baseline strategies.

## Suggested Improvement: Enhanced Fill Probability Modeling

The current implementation uses a simplified model for fill probability. A significant improvement would be to implement a more realistic fill probability model that takes into account:

1. **Order Book Dynamics**: Model the probability of fills based on depth and recent trade activity at each level of the order book intead of just looking at the top of book.

2. **Queue Position Awareness**: Incorporate position in the queue based on arrival time, with fill probabilities decreasing for orders further back in the queue.

3. **Market Microstructure Patterns**: Use historical patterns of order flow to predict future fill probabilities, such as incorporating time-of-day effects, recent volatility, and imbalance between buy and sell orders.

This enhanced model would allow the SOR to make more accurate predictions about which venues offer the best tradeoff between price, rebates, and fill probability, leading to further improvements in execution quality.