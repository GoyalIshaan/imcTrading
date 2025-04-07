import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ta

# Set style
plt.style.use('default')
sns.set_theme()

def load_and_prepare_data():
    # Load the data
    df = pd.read_csv('squid_ink_ml_data.csv')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Calculate additional technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['mid_price']).rsi()
    macd = ta.trend.MACD(df['mid_price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Calculate rolling statistics
    df['price_std'] = df['mid_price'].rolling(window=20).std()
    df['price_mean'] = df['mid_price'].rolling(window=20).mean()
    
    return df

def plot_price_and_indicators(df):
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])
    
    # Plot 1: Price and Moving Averages
    ax1.plot(df['timestamp'], df['mid_price'], label='Mid Price', alpha=0.7)
    ax1.plot(df['timestamp'], df['short_ema'], label='Short EMA', alpha=0.6)
    ax1.plot(df['timestamp'], df['long_ema'], label='Long EMA', alpha=0.6)
    ax1.plot(df['timestamp'], df['price_mean'], label='20-period MA', alpha=0.6)
    ax1.fill_between(df['timestamp'], 
                     df['price_mean'] - 2*df['price_std'],
                     df['price_mean'] + 2*df['price_std'],
                     alpha=0.2, label='±2σ Band')
    ax1.set_title('SQUID_INK Price and Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI
    ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: MACD
    ax3.plot(df['timestamp'], df['macd'], label='MACD', color='blue')
    ax3.plot(df['timestamp'], df['macd_signal'], label='Signal', color='orange')
    ax3.bar(df['timestamp'], df['macd_diff'], label='MACD Histogram', alpha=0.3)
    ax3.set_title('MACD Indicator')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('squid_ink_analysis.png')
    plt.close()

def plot_wave_analysis(df):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Price with wave pattern
    ax1.plot(df['timestamp'], df['mid_price'], label='Price', alpha=0.7)
    
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    n = 20  # window size
    max_idx = argrelextrema(df['mid_price'].values, np.greater, order=n)[0]
    min_idx = argrelextrema(df['mid_price'].values, np.less, order=n)[0]
    
    ax1.plot(df['timestamp'].iloc[max_idx], df['mid_price'].iloc[max_idx], 'r^', label='Local Maxima')
    ax1.plot(df['timestamp'].iloc[min_idx], df['mid_price'].iloc[min_idx], 'gv', label='Local Minima')
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['mid_price'], 1)
    p = np.poly1d(z)
    ax1.plot(df['timestamp'], p(range(len(df))), '--', label='Trend Line', alpha=0.5)
    
    ax1.set_title('SQUID_INK Price Wave Pattern Analysis')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Price Distribution
    sns.histplot(data=df, x='mid_price', bins=50, ax=ax2)
    ax2.axvline(df['mid_price'].mean(), color='r', linestyle='--', label=f'Mean: {df["mid_price"].mean():.2f}')
    ax2.set_title('Price Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('squid_ink_wave_analysis.png')
    plt.close()

def plot_trading_signals(df):
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax.plot(df['timestamp'], df['mid_price'], label='Price', alpha=0.7)
    
    # Plot buy signals (when price is below lower band)
    buy_signals = df[df['mid_price'] < (df['price_mean'] - 2*df['price_std'])]
    ax.scatter(buy_signals['timestamp'], buy_signals['mid_price'], 
              color='g', marker='^', s=100, label='Buy Signal')
    
    # Plot sell signals (when price is above upper band)
    sell_signals = df[df['mid_price'] > (df['price_mean'] + 2*df['price_std'])]
    ax.scatter(sell_signals['timestamp'], sell_signals['mid_price'], 
              color='r', marker='v', s=100, label='Sell Signal')
    
    # Add bands
    ax.fill_between(df['timestamp'], 
                    df['price_mean'] - 2*df['price_std'],
                    df['price_mean'] + 2*df['price_std'],
                    alpha=0.2, label='Trading Bands')
    
    ax.set_title('SQUID_INK Trading Signals')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('squid_ink_trading_signals.png')
    plt.close()

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Generate visualizations
    plot_price_and_indicators(df)
    plot_wave_analysis(df)
    plot_trading_signals(df)
    
    # Print summary statistics
    print("\nSQUID_INK Price Analysis Summary:")
    print(f"Mean Price: {df['mid_price'].mean():.2f}")
    print(f"Price Standard Deviation: {df['mid_price'].std():.2f}")
    print(f"Price Range: {df['mid_price'].min():.2f} - {df['mid_price'].max():.2f}")
    print(f"Average RSI: {df['rsi'].mean():.2f}")
    
    # Calculate cycle statistics
    price_diff = df['mid_price'].diff()
    zero_crossings = np.where(np.diff(np.signbit(price_diff)))[0]
    if len(zero_crossings) > 1:
        avg_cycle_length = np.mean(np.diff(zero_crossings))
        print(f"Average Cycle Length: {avg_cycle_length:.2f} periods")

if __name__ == "__main__":
    main() 