import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_squid_ink_data(num_rows=1000):
    """
    Generate dummy SQUID_INK trading data for analysis
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [int((start_date + timedelta(minutes=i*10)).timestamp()) for i in range(num_rows)]
    
    # Generate price data with some trend and noise
    base_price = 100
    trend = np.cumsum(np.random.normal(0, 0.5, num_rows)) + np.sin(np.linspace(0, 15, num_rows))
    prices = base_price + trend
    
    # Generate bid/ask prices
    spreads = np.random.uniform(0.5, 2.0, num_rows)
    best_bids = prices - spreads/2
    best_asks = prices + spreads/2
    
    # Generate trading data
    data = {
        'timestamp': timestamps,
        'best_bid': best_bids,
        'best_ask': best_asks,
        'mid_price': prices,
        'vwap': prices + np.random.normal(0, 0.3, num_rows),
        'current_ema': prices + np.cumsum(np.random.normal(0, 0.1, num_rows)),
        'short_ema': prices + np.cumsum(np.random.normal(0, 0.15, num_rows)),
        'long_ema': prices + np.cumsum(np.random.normal(0, 0.05, num_rows)),
        'trend_signal': np.random.choice([-1, 0, 1], num_rows, p=[0.3, 0.1, 0.6]),
        'dynamic_width': np.random.uniform(1.0, 5.0, num_rows),
        'position': np.random.randint(-50, 51, num_rows),
        'orders_placed': np.random.randint(0, 10, num_rows),
        'buy_volume': np.random.randint(0, 40, num_rows),
        'sell_volume': np.random.randint(0, 40, num_rows),
        'order_book_imbalance': np.random.uniform(-1.0, 1.0, num_rows),
        'liquidity_score': np.random.uniform(0.0, 1.0, num_rows),
        'momentum': np.random.normal(0, 1.0, num_rows),
    }
    
    # Generate recent P&L with some correlation to other features
    pl_base = 0.4 * data['momentum'] + 0.3 * data['order_book_imbalance'] - 0.2 * np.abs(data['position'])/50
    data['recent_pl'] = pl_base + np.random.normal(0, 2.0, num_rows)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create some patterns in the data
    # 1. Strong up trend period
    uptrend_start = int(num_rows * 0.2)
    uptrend_end = int(num_rows * 0.3)
    df.loc[uptrend_start:uptrend_end, 'trend_signal'] = 1
    df.loc[uptrend_start:uptrend_end, 'momentum'] = np.abs(df.loc[uptrend_start:uptrend_end, 'momentum']) + 0.5
    df.loc[uptrend_start:uptrend_end, 'mid_price'] = df.loc[uptrend_start:uptrend_end, 'mid_price'] * 1.1
    
    # 2. Volatile period with high liquidity
    vol_start = int(num_rows * 0.5)
    vol_end = int(num_rows * 0.6)
    df.loc[vol_start:vol_end, 'dynamic_width'] = df.loc[vol_start:vol_end, 'dynamic_width'] * 1.5
    df.loc[vol_start:vol_end, 'liquidity_score'] = np.random.uniform(0.7, 1.0, vol_end-vol_start+1)
    
    # 3. Profitable trading conditions
    profit_start = int(num_rows * 0.7)
    profit_end = int(num_rows * 0.8)
    df.loc[profit_start:profit_end, 'order_book_imbalance'] = np.random.uniform(0.3, 0.8, profit_end-profit_start+1)
    df.loc[profit_start:profit_end, 'recent_pl'] = np.abs(df.loc[profit_start:profit_end, 'recent_pl']) + 3.0
    
    # Save to CSV
    df.to_csv('squid_ink_ml_data.csv', index=False)
    print(f"Generated {num_rows} rows of dummy SQUID_INK trading data and saved to 'squid_ink_ml_data.csv'")
    return df

if __name__ == "__main__":
    generate_squid_ink_data(1500) 