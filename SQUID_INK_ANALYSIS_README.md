# SQUID_INK Trading Analysis Guide

## Setup and Installation

1. Required Python packages:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn statsmodels ta scipy
```

2. Required data files:

- `squid_ink_ml_data.csv` - Generated during SQUID_INK trading backtests
- `nopen.csv` - Contains market data for multiple products

## Running the Analysis

Execute the analysis scripts:

```bash
# For advanced statistical analysis
python3 advanced_squid_ink_analysis.py

# For real-time price pattern analysis
python3 squid_ink_price_pattern_analyzer.py

# For comprehensive market analysis (all products)
python3 master_market_analyzer.py
```

This will generate several visualization files and print detailed analysis results.

## Generated Visualizations

### 1. Time Series Analysis (`squid_ink_acf_pacf.png`)

- **What it shows**: Autocorrelation and Partial Autocorrelation functions
- **How to interpret**:
  - Bars extending beyond dashed lines indicate significant correlations
  - ACF shows overall correlation patterns
  - PACF shows direct correlations at each lag
- **Trading insights**:
  - Helps identify price momentum patterns
  - Shows how long price trends typically persist

### 2. Market Regimes (`squid_ink_regimes_pca.png`)

- **What it shows**: Different market states clustered by key features
- **How to interpret**:
  - Each color represents a different market regime
  - Distance between points shows similarity of market conditions
  - Clusters show distinct market states
- **Trading insights**:
  - Regime 3 (64.57% profitable) - Best for trading
  - Regime 1 (60% profitable) - Second best
  - Use for regime-specific strategy adjustment

### 3. Lead-Lag Relationships (`squid_ink_lead_lag.png`)

- **What it shows**: How indicators predict price changes
- **How to interpret**:
  - X-axis: Time lag (negative = indicator leads price)
  - Y-axis: Correlation strength
  - Peaks show optimal prediction horizons
- **Trading insights**:
  - Liquidity score leads by 8 periods
  - Order book imbalance leads by 1 period
  - VWAP/mid ratio leads by 9 periods

### 4. Hourly Profitability (`squid_ink_hourly_profitability.png`)

- **What it shows**: Trading performance by hour
- **How to interpret**:
  - Top chart: Win rate by hour
  - Bottom chart: Average P&L by hour
- **Trading insights**:
  - Best hour: 23:00 (63.33% win rate)
  - Best P&L hour: 13:00 (0.8232 average P&L)
  - Avoid trading at 19:00 (35% win rate)

## Key Trading Patterns

### 1. Optimal Conditions (75% Win Rate)

Combined conditions for highest probability trades:

- Momentum < 0.0149
- Liquidity < 0.0285
- Order book imbalance > 0.0548

### 2. Market Regimes

Best trading conditions:

```
Regime 3:
- Win Rate: 64.57%
- Avg P&L: 1.3903
- Characteristics:
  * Mid-range momentum (-0.0242)
  * Moderate liquidity (0.4843)
  * Positive order book imbalance (0.2209)
```

### 3. Individual Indicators

Success rates:

- High order book imbalance: 61.63%
- Low liquidity: 55.26%
- High momentum: 53.75%

## Using the Analysis

1. **Strategy Timing**:

   - Focus on trading during optimal hours (23:00, 13:00)
   - Wait for Regime 3 conditions
   - Use lead-lag relationships for entry timing

2. **Position Sizing**:

   - Increase size when all three optimal conditions are met
   - Reduce size during non-optimal hours
   - Adjust based on regime profitability

3. **Risk Management**:
   - Avoid trading during Regime 2 (lowest profitability)
   - Reduce exposure during high liquidity periods
   - Exit positions when entering unfavorable regimes

## Monitoring and Updates

The analysis should be rerun periodically to:

1. Update regime characteristics
2. Verify indicator relationships
3. Adjust timing preferences
4. Fine-tune trading parameters

## Performance Metrics

Track these key metrics to validate the analysis:

1. Win rate by regime
2. P&L by hour
3. Combined condition success rate
4. Lead-lag relationship stability

## Master Market Analyzer

The `master_market_analyzer.py` script provides comprehensive analysis for all available products, including SQUID_INK and products from nopen.csv. It generates the following visualizations for each product:

### 1. Technical Analysis (`{product}_technical_analysis.png`)

- **What it shows**:
  - Price movement with multiple moving averages
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- **How to interpret**:
  - Price with statistical bands (±2σ) shows volatility ranges
  - RSI indicates overbought (>70) and oversold (<30) conditions
  - MACD shows momentum and potential trend changes

### 2. Market Depth Analysis (`{product}_market_depth.png`)

- **What it shows**:
  - Order book depth at specific timestamps
  - Bid and ask volumes at different price levels
- **How to interpret**:
  - Green bars show bid-side liquidity
  - Red bars show ask-side liquidity
  - Bar heights indicate volume at each price level

### 3. Profitability Analysis (`{product}_profitability.png`)

- **What it shows**:
  - Cumulative profit and loss over time
  - P&L distribution histogram
- **How to interpret**:
  - Trend line shows overall performance
  - Distribution shows trade outcome patterns
  - Mean line indicates average trade performance

### Key Statistics

The analyzer provides comprehensive statistics for each product:

- Price statistics (mean, std dev, range)
- Trading metrics (total trades, average size)
- Performance metrics (total P&L, win rate)
- Technical indicators (RSI, MACD)

### Integration with Trading Strategy

Use these visualizations to:

1. **Market Analysis**:

   - Compare performance across products
   - Identify market regimes
   - Track liquidity patterns

2. **Risk Management**:

   - Monitor position exposure
   - Track P&L distribution
   - Adjust sizing based on volatility

3. **Strategy Optimization**:
   - Compare performance metrics
   - Identify optimal trading conditions
   - Track strategy effectiveness
