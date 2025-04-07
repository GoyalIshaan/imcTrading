import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.gridspec import GridSpec
import json
import warnings
import time
from functools import wraps
import webbrowser
from pathlib import Path

# Import interactive visualization libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Performance decorator for timing functions
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

#############################
# DATA PARSING OPTIMIZATION #
#############################

@timing_decorator
def parse_trade_data_optimized(file_path, chunksize=10000):
    """
    Parse the CSV file with optimized memory usage and type conversions
    """
    result_dfs = []
    
    # Define expected columns
    columns = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 
               'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 
               'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 
               'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
    
    # Process the file in chunks to reduce memory usage
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep=';', chunksize=chunksize)):
            if chunk_num % 10 == 0:
                print(f"Processing chunk {chunk_num}...")
            
            # Ensure we have the right number of columns
            if chunk.shape[1] == len(columns):
                chunk.columns = columns
            else:
                available_columns = chunk.shape[1]
                print(f"Warning: Expected {len(columns)} columns but found {available_columns}")
                chunk.columns = columns[:available_columns]
            
            # Convert columns efficiently with appropriate types
            for col in chunk.columns:
                if col in ['day', 'product']:
                    chunk[col] = chunk[col].astype('category')  # Use category for string columns with repeats
                elif col == 'timestamp':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                else:
                    # Convert numeric columns and optimize dtypes
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    
                    # Use smaller dtypes where possible
                    if col.endswith('_volume'):
                        chunk[col] = chunk[col].astype('float32')  # Volumes need less precision
                    else:
                        chunk[col] = chunk[col].astype('float64')  # Prices need more precision
            
            # Calculate mid price if not present
            if 'mid_price' not in chunk.columns:
                chunk['mid_price'] = (chunk['bid_price_1'] + chunk['ask_price_1']) / 2
            
            # Calculate mid price change
            chunk['mid_price_change'] = chunk['mid_price'].diff()
            
            result_dfs.append(chunk)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()
    
    # Combine all chunks
    if result_dfs:
        final_df = pd.concat(result_dfs, ignore_index=True)
        print(f"Parsed {len(final_df)} rows with {len(final_df.columns)} columns")
        
        # Recalculate mid price change for the entire dataset to handle chunk boundaries
        final_df['mid_price_change'] = final_df['mid_price'].diff()
        
        return final_df
    else:
        print("No data was processed")
        return pd.DataFrame()

# Original parse function for backward compatibility
def parse_trade_data(file_path):
    """Parse the CSV file with semicolon-delimited data in a single column"""
    return parse_trade_data_optimized(file_path)

##########################
# INTERACTIVE VISUALIZER #
##########################

def create_interactive_dashboard(df, product, output_dir=None):
    """
    Create an interactive dashboard with Plotly for a specific product
    
    Args:
        df (pd.DataFrame): DataFrame with trading data
        product (str): Product to visualize
        output_dir (str, optional): Directory to save HTML output
    
    Returns:
        go.Figure: Plotly figure object with the dashboard
    """
    # Filter data for the specified product
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return None
    
    # Sort by timestamp
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Create interactive subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            f'Price Movement for {product}',
            'Order Book Imbalance',
            'Profit and Loss',
            'Bid-Ask Spread'
        )
    )
    
    # 1. Price chart with bid/ask
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['mid_price'],
            mode='lines',
            name='Mid Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['bid_price_1'],
            mode='lines',
            name='Best Bid',
            line=dict(color='green', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['ask_price_1'],
            mode='lines',
            name='Best Ask',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. Order book imbalance
    product_df['total_bid_volume'] = product_df['bid_volume_1'] + product_df['bid_volume_2'] + product_df['bid_volume_3']
    product_df['total_ask_volume'] = product_df['ask_volume_1'] + product_df['ask_volume_2'] + product_df['ask_volume_3']
    
    # Calculate imbalance safely
    with np.errstate(divide='ignore', invalid='ignore'):
        product_df['volume_imbalance'] = (
            product_df['total_bid_volume'] - product_df['total_ask_volume']
        ) / (product_df['total_bid_volume'] + product_df['total_ask_volume'])
    
    # Replace inf/-inf with NaN, then fill NaN with 0
    product_df['volume_imbalance'] = product_df['volume_imbalance'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['volume_imbalance'],
            mode='lines',
            name='OB Imbalance',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=product_df['timestamp'].min(),
        y0=0,
        x1=product_df['timestamp'].max(),
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=2, col=1
    )
    
    # 3. P&L chart
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['profit_and_loss'],
            mode='lines',
            name='P&L',
            line=dict(color='darkgreen', width=1.5)
        ),
        row=3, col=1
    )
    
    # 4. Spread chart
    product_df['spread'] = product_df['ask_price_1'] - product_df['bid_price_1']
    
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['spread'],
            mode='lines',
            name='Bid-Ask Spread',
            line=dict(color='orange', width=1.5)
        ),
        row=4, col=1
    )
    
    # Update layout for better appearance
    fig.update_layout(
        height=900,
        width=1200,
        title_text=f"Interactive Trading Dashboard for {product}",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_layout(
        xaxis4=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    # Add buttons for time periods
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.1,
                y=1.15,
                buttons=list([
                    dict(
                        label="All Data",
                        method="relayout",
                        args=[{"xaxis.range": [product_df['timestamp'].min(), product_df['timestamp'].max()]}]
                    ),
                    dict(
                        label="Last 25%",
                        method="relayout",
                        args=[{"xaxis.range": [
                            product_df['timestamp'].quantile(0.75),
                            product_df['timestamp'].max()
                        ]}]
                    ),
                    dict(
                        label="Last 10%",
                        method="relayout",
                        args=[{"xaxis.range": [
                            product_df['timestamp'].quantile(0.9),
                            product_df['timestamp'].max()
                        ]}]
                    )
                ]),
            )
        ]
    )
    
    # Save the interactive plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{product}_interactive_dashboard.html")
        fig.write_html(output_file)
        print(f"Interactive dashboard saved to {output_file}")
    
    return fig

##########################
# TRADING ALPHA SIGNALS  #
##########################

def calculate_trading_alpha_signals(df, product):
    """
    Calculate various trading metrics and potential alpha signals
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
        product (str): Product to analyze
    
    Returns:
        pd.DataFrame: DataFrame with trading signals and metrics
    """
    # Filter data for the specific product
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return pd.DataFrame()
    
    # Sort by timestamp
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"Calculating trading alpha signals for {product}...")
    
    # 1. Basic price and return calculations
    product_df['returns'] = product_df['mid_price'].pct_change()
    product_df['log_returns'] = np.log(product_df['mid_price'] / product_df['mid_price'].shift(1))
    
    # 2. Calculate various window statistics
    windows = [5, 10, 20, 50]
    
    for window in windows:
        # Simple moving averages
        product_df[f'sma_{window}'] = product_df['mid_price'].rolling(window=window).mean()
        
        # Exponential moving averages
        product_df[f'ema_{window}'] = product_df['mid_price'].ewm(span=window, adjust=False).mean()
        
        # Moving average convergence divergence
        if window == 20:  # Standard MACD parameters
            product_df['macd'] = product_df['mid_price'].ewm(span=12, adjust=False).mean() - \
                               product_df['mid_price'].ewm(span=26, adjust=False).mean()
            product_df['macd_signal'] = product_df['macd'].ewm(span=9, adjust=False).mean()
            product_df['macd_hist'] = product_df['macd'] - product_df['macd_signal']
        
        # Volatility and returns
        product_df[f'volatility_{window}'] = product_df['returns'].rolling(window=window).std()
        product_df[f'rolling_return_{window}'] = product_df['returns'].rolling(window=window).sum()
        
        # Z-score (mean reversion signal)
        product_df[f'price_zscore_{window}'] = (
            product_df['mid_price'] - product_df[f'sma_{window}']
        ) / product_df['mid_price'].rolling(window=window).std()
    
    # 3. Technical indicators
    # Relative Strength Index (RSI)
    delta = product_df['mid_price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / (avg_loss + 0.0001)  # Relative strength
    product_df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    window = 20
    product_df['bb_middle'] = product_df['mid_price'].rolling(window=window).mean()
    product_df['bb_std'] = product_df['mid_price'].rolling(window=window).std()
    product_df['bb_upper'] = product_df['bb_middle'] + 2 * product_df['bb_std']
    product_df['bb_lower'] = product_df['bb_middle'] - 2 * product_df['bb_std']
    product_df['bb_width'] = (product_df['bb_upper'] - product_df['bb_lower']) / product_df['bb_middle']
    product_df['bb_pct'] = (product_df['mid_price'] - product_df['bb_lower']) / (product_df['bb_upper'] - product_df['bb_lower'])
    
    # Average True Range (ATR)
    high_low = product_df['ask_price_1'] - product_df['bid_price_1']
    high_close = abs(product_df['ask_price_1'] - product_df['mid_price'].shift())
    low_close = abs(product_df['bid_price_1'] - product_df['mid_price'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    product_df['atr'] = true_range.rolling(window=14).mean()
    product_df['atr_pct'] = product_df['atr'] / product_df['mid_price'] * 100
    
    # 4. Order book signals
    # Bid-ask imbalance
    product_df['bid_ask_imbalance'] = (
        product_df['bid_volume_1'] - product_df['ask_volume_1']
    ) / (product_df['bid_volume_1'] + product_df['ask_volume_1'] + 0.0001)
    
    # Moving average of imbalance
    product_df['bid_ask_imbalance_sma'] = product_df['bid_ask_imbalance'].rolling(window=10).mean()
    
    # Order book pressure (combines price and volume)
    product_df['ob_pressure'] = product_df['bid_ask_imbalance'] * product_df['mid_price_change'].abs()
    
    # 5. Advanced signals
    # Momentum signals
    for window in [5, 10, 20]:
        product_df[f'momentum_{window}'] = product_df['mid_price'] / product_df['mid_price'].shift(window) - 1
        product_df[f'volume_momentum_{window}'] = (
            product_df['bid_volume_1'] + product_df['ask_volume_1']
        ).rolling(window=window).sum().pct_change()
    
    # Mean reversion signals
    product_df['mean_rev_signal'] = -product_df['price_zscore_20']  # Negative z-score for mean reversion
    
    # Trend signals
    product_df['trend_signal'] = np.where(
        product_df['sma_5'] > product_df['sma_20'],
        1,  # Uptrend
        np.where(
            product_df['sma_5'] < product_df['sma_20'],
            -1,  # Downtrend
            0   # No clear trend
        )
    )
    
    # Breakout signals
    product_df['high_20'] = product_df['mid_price'].rolling(window=20).max()
    product_df['low_20'] = product_df['mid_price'].rolling(window=20).min()
    
    product_df['breakout_signal'] = np.where(
        product_df['mid_price'] > product_df['high_20'].shift(1),
        1,  # Bullish breakout
        np.where(
            product_df['mid_price'] < product_df['low_20'].shift(1),
            -1,  # Bearish breakout
            0   # No breakout
        )
    )
    
    # 6. Performance and risk metrics
    # Calculate Sharpe ratio (simplified)
    window = 20  # Equivalent to ~1 hour for 3-minute bars
    risk_free_rate = 0  # Assuming zero for short-term trading
    
    product_df['rolling_sharpe'] = (
        product_df['rolling_return_20'] - risk_free_rate
    ) / (product_df['volatility_20'] + 0.0001) * np.sqrt(252 * 20)
    
    # Calculate Sortino ratio (downside risk only)
    downside_returns = product_df['returns'].copy()
    downside_returns[downside_returns > 0] = 0
    product_df['downside_volatility_20'] = downside_returns.rolling(window=20).std()
    
    product_df['rolling_sortino'] = (
        product_df['rolling_return_20'] - risk_free_rate
    ) / (product_df['downside_volatility_20'] + 0.0001) * np.sqrt(252 * 20)
    
    # Maximum drawdown over rolling window
    window = 50
    rolling_max = product_df['mid_price'].rolling(window=window, min_periods=1).max()
    drawdown = (product_df['mid_price'] / rolling_max - 1)
    product_df['max_drawdown_50'] = drawdown.rolling(window=window).min()
    
    # 7. Combined alpha signal
    # This is a simple example combining multiple signals
    product_df['alpha_signal'] = (
        0.3 * product_df['mean_rev_signal'].fillna(0) +  # Mean reversion component
        0.3 * product_df['trend_signal'].fillna(0) +     # Trend component
        0.2 * product_df['bid_ask_imbalance'].fillna(0) + # Order book component
        0.1 * product_df['breakout_signal'].fillna(0) +  # Breakout component
        0.1 * np.sign(product_df['macd_hist']).fillna(0)  # MACD component
    )
    
    # Normalize the alpha signal
    product_df['alpha_signal_norm'] = (
        product_df['alpha_signal'] / product_df['alpha_signal'].rolling(window=50).std()
    ).fillna(0)
    
    # Print summary of alpha signals
    print("\n=== Trading Alpha Signals Summary ===")
    print(f"Mean Reversion Signal (avg): {product_df['mean_rev_signal'].mean():.4f}")
    print(f"Trend Signal Distribution:")
    print(product_df['trend_signal'].value_counts(normalize=True))
    print(f"Breakout Signal Frequency: {(product_df['breakout_signal'] != 0).mean():.2%}")
    print(f"Average Rolling Sharpe: {product_df['rolling_sharpe'].mean():.4f}")
    print(f"Average Rolling Sortino: {product_df['rolling_sortino'].mean():.4f}")
    print(f"Maximum Drawdown: {product_df['max_drawdown_50'].min():.2%}")
    print(f"Alpha Signal Range: {product_df['alpha_signal_norm'].min():.2f} to {product_df['alpha_signal_norm'].max():.2f}")
    
    return product_df

def visualize_trading_signals(df, output_dir=None):
    """
    Create visualizations for trading signals and alpha analysis
    
    Args:
        df (pd.DataFrame): DataFrame with trading signals
        output_dir (str, optional): Directory to save visualizations
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price with Moving Averages
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_upper'],
            name="Upper Band",
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_middle'],
            name="Middle Band",
            line=dict(color='orange', width=1),
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_lower'],
            name="Lower Band",
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.7
        )
    )
    
    # Fill area between upper and lower bands
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
            y=df['bb_upper'].tolist() + df['bb_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='none'
        )
    )
    
    fig.update_layout(
        title="Bollinger Bands",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode='x unified'
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "bollinger_bands.html")
        fig.write_html(output_file)
        print(f"Bollinger Bands visualization saved to {output_file}")
    
    # 3. RSI Indicator
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rsi'],
            name="RSI",
            line=dict(color='purple', width=1)
        ),
        secondary_y=True
    )
    
    # Add overbought/oversold reference lines
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=70,
        x1=df['timestamp'].max(),
        y1=70,
        line=dict(color="red", width=1, dash="dash"),
        yref="y2"
    )
    
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=30,
        x1=df['timestamp'].max(),
        y1=30,
        line=dict(color="green", width=1, dash="dash"),
        yref="y2"
    )
    
    fig.update_layout(
        title="RSI Indicator",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(
        title_text="RSI", 
        secondary_y=True,
        range=[0, 100]
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "rsi_indicator.html")
        fig.write_html(output_file)
        print(f"RSI indicator visualization saved to {output_file}")
    
    # 4. MACD Indicator
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['macd'],
            name="MACD",
            line=dict(color='green', width=1)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['macd_signal'],
            name="Signal Line",
            line=dict(color='red', width=1)
        ),
        secondary_y=True
    )
    
    # Add MACD histogram
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['macd_hist'],
            name="MACD Histogram",
            marker=dict(
                color=df['macd_hist'].apply(lambda x: 'green' if x > 0 else 'red'),
                opacity=0.7
            )
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="MACD Indicator",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="MACD", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "macd_indicator.html")
        fig.write_html(output_file)
        print(f"MACD indicator visualization saved to {output_file}")
    
    # 5. Order Book Signals
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bid_ask_imbalance'],
            name="Bid-Ask Imbalance",
            line=dict(color='purple', width=1)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bid_ask_imbalance_sma'],
            name="Imbalance SMA",
            line=dict(color='orange', width=1, dash='dash')
        ),
        secondary_y=True
    )
    
    # Add zero reference line
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=0,
        x1=df['timestamp'].max(),
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        yref="y2"
    )
    
    fig.update_layout(
        title="Order Book Imbalance Signals",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Imbalance", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "order_book_signals.html")
        fig.write_html(output_file)
        print(f"Order book signals visualization saved to {output_file}")
    
    # 6. Alpha Signal with Price
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['alpha_signal_norm'],
            name="Alpha Signal",
            line=dict(color='red', width=1)
        ),
        secondary_y=True
    )
    
    # Add horizontal lines for signal thresholds
    for threshold in [1, -1]:
        fig.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=threshold,
            x1=df['timestamp'].max(),
            y1=threshold,
            line=dict(color="black", width=1, dash="dash"),
            yref="y2"
        )
    
    # Highlight strong signal regions
    buy_regions = df[df['alpha_signal_norm'] > 1]
    sell_regions = df[df['alpha_signal_norm'] < -1]
    
    for i, row in buy_regions.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['timestamp'],
            y0=0,
            x1=row['timestamp'] + 50,  # arbitrary width for visibility
            y1=1,
            yref="paper",
            fillcolor="green",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    
    for i, row in sell_regions.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['timestamp'],
            y0=0,
            x1=row['timestamp'] + 50,  # arbitrary width for visibility
            y1=1,
            yref="paper",
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title="Alpha Signal with Price",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Alpha Signal", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "alpha_signal.html")
        fig.write_html(output_file)
        print(f"Alpha signal visualization saved to {output_file}")
    
    # 7. Performance Metrics
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rolling_return_20'],
            name="Rolling Return (20)",
            line=dict(color='green', width=1)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rolling_sharpe'],
            name="Sharpe Ratio",
            line=dict(color='blue', width=1)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rolling_sortino'],
            name="Sortino Ratio",
            line=dict(color='purple', width=1, dash='dash')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Performance Metrics",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Return", secondary_y=False)
    fig.update_yaxes(title_text="Risk-Adjusted Return", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "performance_metrics.html")
        fig.write_html(output_file)
        print(f"Performance metrics visualization saved to {output_file}")
    
    # 8. Drawdown Analysis
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['max_drawdown_50'] * 100,  # Convert to percentage
            name="Max Drawdown (50-period)",
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        )
    )
    
    fig.update_layout(
        title="Maximum Drawdown Analysis",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        yaxis=dict(
            tickformat='.2f',
            autorange="reversed"  # Invert y-axis so drawdowns go down
        )
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "drawdown_analysis.html")
        fig.write_html(output_file)
        print(f"Drawdown analysis visualization saved to {output_file}")
    
    # Return last figure for display if not saving to file
    return fig

###########################
# MARKET MICROSTRUCTURE   #
###########################

def analyze_market_microstructure(df, product):
    """
    Analyze market microstructure metrics to understand trading dynamics
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
        product (str): Product to analyze
    
    Returns:
        pd.DataFrame: DataFrame with market microstructure metrics
    """
    # Filter data for the specific product
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return pd.DataFrame()
    
    # Sort by timestamp
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"Analyzing market microstructure for {product}...")
    
    # 1. Effective spread
    product_df['effective_spread'] = 2 * abs(product_df['mid_price'] - 
                                           (product_df['bid_price_1'] + product_df['ask_price_1']) / 2)
    
    # 2. Realized spread (approximation)
    horizon = 5
    product_df['mid_price_future'] = product_df['mid_price'].shift(-horizon)
    product_df['price_impact'] = (product_df['mid_price_future'] - product_df['mid_price'])
    product_df['realized_spread'] = product_df['effective_spread'] - product_df['price_impact'].abs()
    
    # 3. Price impact measures
    product_df['mid_price_change'] = product_df['mid_price'].diff()
    product_df['mid_price_pct_change'] = product_df['mid_price'].pct_change() * 100
    
    # 4. Trade size and price impact relationship
    product_df['trade_size_proxy'] = (product_df['bid_volume_1'] + product_df['ask_volume_1']) / 2
    
    # 5. Amihud illiquidity
    with np.errstate(divide='ignore', invalid='ignore'):
        product_df['amihud'] = abs(product_df['mid_price_pct_change']) / (product_df['trade_size_proxy'] + 0.0001)
    product_df['amihud'] = product_df['amihud'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 6. Order flow toxicity proxy (VPIN-like)
    window = 10
    product_df['volume_imbalance'] = abs(product_df['bid_volume_1'] - product_df['ask_volume_1'])
    product_df['vpin_proxy'] = product_df['volume_imbalance'].rolling(window=window).sum() / \
                              (product_df['trade_size_proxy'].rolling(window=window).sum() + 0.0001)
    
    # 7. Roll implicit spread estimator
    window = 20
    product_df['roll_cov'] = product_df['mid_price_change'].rolling(window=window).apply(
        lambda x: -np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan
    )
    product_df['roll_spread'] = 2 * np.sqrt(np.maximum(0, product_df['roll_cov']))
    
    # 8. Kyle's lambda (price impact parameter)
    window = 20
    for w in [window, window*2]:
        product_df[f'kyle_lambda_{w}'] = product_df['mid_price_change'].rolling(window=w).cov(
            product_df['volume_imbalance']
        ) / (product_df['volume_imbalance'].rolling(window=w).var() + 0.0001)
    
    # 9. Autocorrelation of returns (market efficiency)
    lags = [1, 5, 10]
    for lag in lags:
        product_df[f'return_autocorr_{lag}'] = product_df['mid_price_pct_change'].rolling(window=window).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
        )
    
    # 10. Volatility measures
    for w in [10, 20, 50]:
        product_df[f'volatility_{w}'] = product_df['mid_price_pct_change'].rolling(window=w).std()
        if 'ask_price_1' in product_df.columns and 'bid_price_1' in product_df.columns:
            product_df[f'range_volatility_{w}'] = (
                (product_df['ask_price_1'] - product_df['bid_price_1']) / product_df['mid_price']
            ).rolling(window=w).mean() * 100
    
    print("\n=== Market Microstructure Analysis Summary ===")
    print(f"Average Effective Spread: {product_df['effective_spread'].mean():.5f}")
    print(f"Average Realized Spread: {product_df['realized_spread'].mean():.5f}")
    print(f"Average Price Impact: {product_df['price_impact'].abs().mean():.5f}")
    print(f"Average Amihud Illiquidity: {product_df['amihud'].mean():.5f}")
    print(f"Average VPIN (Order Flow Toxicity): {product_df['vpin_proxy'].mean():.5f}")
    print(f"Average Roll Implicit Spread: {product_df['roll_spread'].mean():.5f}")
    print(f"Average Kyle's Lambda ({window}): {product_df[f'kyle_lambda_{window}'].mean():.5f}")
    print(f"Return Autocorrelation (Lag 1): {product_df['return_autocorr_1'].mean():.5f}")
    print(f"Volatility (20-period): {product_df['volatility_20'].mean():.5f}%")
    
    return product_df

def visualize_market_microstructure(df, output_dir=None):
    """
    Create visualizations for market microstructure analysis
    
    Args:
        df (pd.DataFrame): DataFrame with market microstructure metrics
        output_dir (str, optional): Directory to save visualizations
    """
    if df.empty:
        print("No data to visualize")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Spread Analysis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['effective_spread'],
            name="Effective Spread",
            line=dict(color='red', width=1)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['roll_spread'],
            name="Roll Spread",
            line=dict(color='orange', width=1, dash='dash')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Spread Analysis",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Spread", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "spread_analysis.html")
        fig.write_html(output_file)
        print(f"Spread analysis visualization saved to {output_file}")
    
    # 2. Price Impact Analysis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price_pct_change'],
            name="Price Change %",
            line=dict(color='blue', width=1)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['amihud'],
            name="Amihud Illiquidity",
            line=dict(color='purple', width=1)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Price Impact and Liquidity",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price Change %", secondary_y=False)
    fig.update_yaxes(title_text="Amihud Illiquidity", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "price_impact.html")
        fig.write_html(output_file)
        print(f"Price impact visualization saved to {output_file}")
    
    # 3. Order Flow Toxicity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['vpin_proxy'],
            name="VPIN (Flow Toxicity)",
            line=dict(color='red', width=1)
        ),
        secondary_y=True
    )
    
    high_toxicity = df[df['vpin_proxy'] > df['vpin_proxy'].quantile(0.9)]
    
    for i, row in high_toxicity.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['timestamp'],
            y0=0,
            x1=row['timestamp'] + 50,
            y1=1,
            yref="paper",
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title="Order Flow Toxicity (VPIN)",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="VPIN", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "order_flow_toxicity.html")
        fig.write_html(output_file)
        print(f"Order flow toxicity visualization saved to {output_file}")
    
    # 4. Market Efficiency (Return Autocorrelation)
    fig = go.Figure()
    
    for lag in [1, 5, 10]:
        if f'return_autocorr_{lag}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[f'return_autocorr_{lag}'],
                    name=f"Autocorr Lag {lag}",
                    line=dict(width=1)
                )
            )
    
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=0,
        x1=df['timestamp'].max(),
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    fig.update_layout(
        title="Market Efficiency (Return Autocorrelation)",
        xaxis_title="Time",
        yaxis_title="Autocorrelation",
        hovermode='x unified'
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "market_efficiency.html")
        fig.write_html(output_file)
        print(f"Market efficiency visualization saved to {output_file}")
    
    # 5. Volatility Analysis
    fig = go.Figure()
    
    for w in [10, 20, 50]:
        if f'volatility_{w}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[f'volatility_{w}'],
                    name=f"Volatility {w}-period",
                    line=dict(width=1)
                )
            )
    
    fig.update_layout(
        title="Price Volatility",
        xaxis_title="Time",
        yaxis_title="Volatility (%)",
        hovermode='x unified'
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "volatility.html")
        fig.write_html(output_file)
        print(f"Volatility visualization saved to {output_file}")
    
    return fig

def create_order_book_heatmap(df, product, output_dir=None):
    """
    Create an interactive order book heatmap visualization
    
    Args:
        df (pd.DataFrame): DataFrame with trading data
        product (str): Product to visualize
        output_dir (str, optional): Directory to save HTML output
    
    Returns:
        go.Figure: Plotly figure object with the heatmap
    """
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return None
    
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    sample_rate = max(1, len(product_df) // 100)
    sampled_df = product_df.iloc[::sample_rate].reset_index(drop=True)
    
    bid_volume_cols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    ask_volume_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
    bid_price_cols = ['bid_price_1', 'bid_price_2', 'bid_price_3']
    ask_price_cols = ['ask_price_1', 'ask_price_2', 'ask_price_3']
    
    for cols in [bid_volume_cols, ask_volume_cols, bid_price_cols, ask_price_cols]:
        for col in cols:
            if col not in sampled_df.columns:
                sampled_df[col] = 0
    
    bid_volumes = sampled_df[bid_volume_cols].values
    ask_volumes = sampled_df[ask_volume_cols].values
    
    bid_prices = sampled_df[bid_price_cols].values
    ask_prices = sampled_df[ask_price_cols].values
    
    max_volume = max(np.nanmax(bid_volumes), np.nanmax(ask_volumes))
    if max_volume > 0:
        bid_volumes_norm = bid_volumes / max_volume
        ask_volumes_norm = ask_volumes / max_volume
    else:
        bid_volumes_norm = bid_volumes
        ask_volumes_norm = ask_volumes
    
    order_book = np.zeros((len(sampled_df), 6))
    order_book[:, 0:3] = -bid_volumes_norm
    order_book[:, 3:6] = ask_volumes_norm
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=order_book,
            x=['Bid 3', 'Bid 2', 'Bid 1', 'Ask 1', 'Ask 2', 'Ask 3'],
            y=sampled_df['timestamp'],
            colorscale='RdBu',
            zmid=0,
            text=np.array([
                [f"Bid 3: {sampled_df[bid_price_cols[2]].iloc[i]:.2f}<br>Vol: {sampled_df[bid_volume_cols[2]].iloc[i]:.2f}" for i in range(len(sampled_df))],
                [f"Bid 2: {sampled_df[bid_price_cols[1]].iloc[i]:.2f}<br>Vol: {sampled_df[bid_volume_cols[1]].iloc[i]:.2f}" for i in range(len(sampled_df))],
                [f"Bid 1: {sampled_df[bid_price_cols[0]].iloc[i]:.2f}<br>Vol: {sampled_df[bid_volume_cols[0]].iloc[i]:.2f}" for i in range(len(sampled_df))],
                [f"Ask 1: {sampled_df[ask_price_cols[0]].iloc[i]:.2f}<br>Vol: {sampled_df[ask_volume_cols[0]].iloc[i]:.2f}" for i in range(len(sampled_df))],
                [f"Ask 2: {sampled_df[ask_price_cols[1]].iloc[i]:.2f}<br>Vol: {sampled_df[ask_volume_cols[1]].iloc[i]:.2f}" for i in range(len(sampled_df))],
                [f"Ask 3: {sampled_df[ask_price_cols[2]].iloc[i]:.2f}<br>Vol: {sampled_df[ask_volume_cols[2]].iloc[i]:.2f}" for i in range(len(sampled_df))]
            ]).transpose(),
            hoverinfo='text',
        )
    )
    
    fig.update_layout(
        title=f'Order Book Depth Heatmap for {product}',
        height=800,
        width=1000,
        yaxis=dict(
            title='Time',
            autorange='reversed'
        ),
        xaxis=dict(
            title='Order Book Level',
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=['Bid 3', 'Bid 2', 'Bid 1', 'Ask 1', 'Ask 2', 'Ask 3']
        ),
        coloraxis_colorbar=dict(
            title='Volume',
            tickvals=[-1, 0, 1],
            ticktext=['Max Bid', 'Neutral', 'Max Ask']
        ),
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{product}_ob_heatmap.html")
        fig.write_html(output_file)
        print(f"Order book heatmap saved to {output_file}")
    
    return fig

def create_price_volume_interactive(df, product, output_dir=None):
    """
    Create an interactive price and volume chart
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
        product (str): Product to visualize
        output_dir (str, optional): Directory to save HTML output
    
    Returns:
        go.Figure: Plotly figure with the chart
    """
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return None
    
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    product_df['total_volume'] = (
        product_df['bid_volume_1'].abs() + 
        product_df['bid_volume_2'].abs() + 
        product_df['bid_volume_3'].abs() +
        product_df['ask_volume_1'].abs() + 
        product_df['ask_volume_2'].abs() + 
        product_df['ask_volume_3'].abs()
    )
    
    if len(product_df) > 1000:
        sample_rate = len(product_df) // 1000
        sampled_df = product_df.iloc[::sample_rate].reset_index(drop=True)
    else:
        sampled_df = product_df
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=product_df['timestamp'],
            y=product_df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=sampled_df['timestamp'],
            y=sampled_df['total_volume'],
            name="Volume",
            marker=dict(color='rgba(255, 0, 0, 0.5)')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title_text=f"Price and Volume Chart for {product}",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Mid Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{product}_price_volume_interactive.html")
        fig.write_html(output_file)
        print(f"Interactive price-volume chart saved to {output_file}")
    
    return fig

##########################
# ORDER BOOK ANALYSIS    #
##########################

def analyze_order_book_dynamics(df, product):
    """
    Analyze order book dynamics including liquidity, depth, and skew
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
        product (str): Product to analyze
    
    Returns:
        pd.DataFrame: DataFrame with order book analysis
    """
    product_df = df[df['product'] == product].copy()
    
    if product_df.empty:
        print(f"No data found for product: {product}")
        return pd.DataFrame()
    
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"Analyzing order book dynamics for {product}...")
    
    for side in ['bid', 'ask']:
        for level in range(1, 4):
            col = f'{side}_volume_{level}'
            if col in product_df.columns:
                product_df[col] = product_df[col].abs()
    
    product_df['bid_liquidity'] = (
        product_df['bid_volume_1'] * product_df['bid_price_1'] +
        product_df['bid_volume_2'] * product_df['bid_price_2'] + 
        product_df['bid_volume_3'] * product_df['bid_price_3']
    )
    
    product_df['ask_liquidity'] = (
        product_df['ask_volume_1'] * product_df['ask_price_1'] +
        product_df['ask_volume_2'] * product_df['ask_price_2'] + 
        product_df['ask_volume_3'] * product_df['ask_price_3']
    )
    
    product_df['total_liquidity'] = product_df['bid_liquidity'] + product_df['ask_liquidity']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        product_df['ob_skew'] = (
            product_df['bid_liquidity'] - product_df['ask_liquidity']
        ) / (product_df['bid_liquidity'] + product_df['ask_liquidity'])
    product_df['ob_skew'] = product_df['ob_skew'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    product_df['bid_price_spacing'] = product_df['bid_price_1'] - product_df['bid_price_2']
    product_df['ask_price_spacing'] = product_df['ask_price_2'] - product_df['ask_price_1']
    product_df['price_spacing_ratio'] = product_df['ask_price_spacing'] / product_df['bid_price_spacing']
    
    product_df['bid_top_concentration'] = product_df['bid_volume_1'] / (
        product_df['bid_volume_1'] + product_df['bid_volume_2'] + product_df['bid_volume_3'] + 0.0001
    )
    
    product_df['ask_top_concentration'] = product_df['ask_volume_1'] / (
        product_df['ask_volume_1'] + product_df['ask_volume_2'] + product_df['ask_volume_3'] + 0.0001
    )
    
    product_df['bid_decay'] = product_df['bid_volume_1'] / (product_df['bid_volume_2'] + 0.0001)
    product_df['ask_decay'] = product_df['ask_volume_1'] / (product_df['ask_volume_2'] + 0.0001)
    
    product_df['book_pressure'] = product_df['ob_skew'] * np.log1p(product_df['total_liquidity'])
    
    product_df['spread'] = product_df['ask_price_1'] - product_df['bid_price_1']
    product_df['spread_pct'] = product_df['spread'] / product_df['mid_price'] * 100
    
    product_df['bid_shape'] = (product_df['bid_price_1'] - product_df['bid_price_2']) / (
        product_df['bid_price_2'] - product_df['bid_price_3'] + 0.0001
    )
    
    product_df['ask_shape'] = (product_df['ask_price_2'] - product_df['ask_price_1']) / (
        product_df['ask_price_3'] - product_df['ask_price_2'] + 0.0001
    )
    
    window_sizes = [5, 10, 20]
    
    for window in window_sizes:
        product_df[f'liquidity_change_{window}'] = product_df['total_liquidity'].pct_change(window)
        product_df[f'spread_vol_{window}'] = product_df['spread'].rolling(window=window).std()
        product_df[f'ob_skew_ma_{window}'] = product_df['ob_skew'].rolling(window=window).mean()
    
    print("\n=== Order Book Analysis Summary ===")
    print(f"Average Spread: {product_df['spread'].mean():.5f}")
    print(f"Average Spread %: {product_df['spread_pct'].mean():.4f}%")
    print(f"Average Bid Liquidity: {product_df['bid_liquidity'].mean():.2f}")
    print(f"Average Ask Liquidity: {product_df['ask_liquidity'].mean():.2f}")
    print(f"Average Book Skew: {product_df['ob_skew'].mean():.4f}")
    print(f"Top of Book Concentration - Bids: {product_df['bid_top_concentration'].mean()*100:.2f}%")
    print(f"Top of Book Concentration - Asks: {product_df['ask_top_concentration'].mean()*100:.2f}%")
    
    return product_df

def visualize_order_book_analysis(df, output_dir=None):
    """
    Create visualizations for order book analysis
    
    Args:
        df (pd.DataFrame): DataFrame with order book metrics
        output_dir (str, optional): Directory to save visualizations
    """
    if df.empty:
        print("No data to visualize")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Order Book Skew vs Price Movement
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'],
            name="Mid Price",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ob_skew'],
            name="Order Book Skew",
            line=dict(color='red', width=1)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Order Book Skew vs Price",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="OB Skew", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "ob_skew_vs_price.html")
        fig.write_html(output_file)
        print(f"Order book skew visualization saved to {output_file}")
    
    # 2. Liquidity and Spread Analysis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['total_liquidity'],
            name="Total Liquidity",
            line=dict(color='green', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['spread'],
            name="Spread",
            line=dict(color='orange', width=1)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Liquidity vs Spread",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Total Liquidity", secondary_y=False)
    fig.update_yaxes(title_text="Spread", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "liquidity_vs_spread.html")
        fig.write_html(output_file)
        print(f"Liquidity and spread visualization saved to {output_file}")
    
    # 3. Order Book Shape Analysis
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bid_shape'],
            name="Bid Shape",
            line=dict(color='green', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ask_shape'],
            name="Ask Shape",
            line=dict(color='red', width=1)
        )
    )
    
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=1,
        x1=df['timestamp'].max(),
        y1=1,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    fig.update_layout(
        title="Order Book Shape Analysis",
        xaxis_title="Time",
        yaxis_title="Shape Coefficient",
        hovermode='x unified'
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "order_book_shape.html")
        fig.write_html(output_file)
        print(f"Order book shape visualization saved to {output_file}")
    
    # 4. Volume Concentration Analysis
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bid_top_concentration'],
            name="Bid Top Concentration",
            line=dict(color='green', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ask_top_concentration'],
            name="Ask Top Concentration",
            line=dict(color='red', width=1)
        )
    )
    
    fig.update_layout(
        title="Order Book Volume Concentration",
        xaxis_title="Time",
        yaxis_title="Top Level Concentration",
        hovermode='x unified'
    )
    
    if output_dir:
        output_file = os.path.join(output_dir, "volume_concentration.html")
        fig.write_html(output_file)
        print(f"Volume concentration visualization saved to {output_file}")
    
    # 5. Book Pressure and Price Impact Analysis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['mid_price'].diff(),
            name="Price Change",
            line=dict(color='blue', width=1)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['book_pressure'],
            name="Book Pressure",
            line=dict(color='purple', width=1)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Book Pressure vs Price Change",
        xaxis_title="Time",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price Change", secondary_y=False)
    fig.update_yaxes(title_text="Book Pressure", secondary_y=True)
    
    if output_dir:
        output_file = os.path.join(output_dir, "book_pressure.html")
        fig.write_html(output_file)
        print(f"Book pressure visualization saved to {output_file}")
        
    return fig

#############################
# DATA EXPORT FUNCTIONS     #
#############################

def export_analysis_to_csv(df, product, output_dir):
    """
    Export all analysis data to CSV files for LLM consumption
    
    Args:
        df (pd.DataFrame): DataFrame with trading data
        product (str): Product being analyzed
        output_dir (str): Directory to save CSV files
    
    Returns:
        list: List of exported CSV file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []
    
    # 1. Export raw data with basic metrics
    raw_data = df.copy()
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], unit='s')
    raw_data['date'] = raw_data['timestamp'].dt.date
    raw_data['time'] = raw_data['timestamp'].dt.time
    
    # Add basic metrics
    raw_data['spread'] = raw_data['ask_price_1'] - raw_data['bid_price_1']
    raw_data['spread_pct'] = raw_data['spread'] / raw_data['mid_price'] * 100
    raw_data['total_volume'] = (
        raw_data['bid_volume_1'] + raw_data['bid_volume_2'] + raw_data['bid_volume_3'] +
        raw_data['ask_volume_1'] + raw_data['ask_volume_2'] + raw_data['ask_volume_3']
    )
    
    raw_data_file = os.path.join(output_dir, f"{product}_raw_data.csv")
    raw_data.to_csv(raw_data_file, index=False)
    exported_files.append(raw_data_file)
    print(f"Raw data exported to {raw_data_file}")
    
    # 2. Export order book analysis
    ob_df = analyze_order_book_dynamics(df, product)
    if not ob_df.empty:
        ob_file = os.path.join(output_dir, f"{product}_order_book_analysis.csv")
        ob_df.to_csv(ob_file, index=False)
        exported_files.append(ob_file)
        print(f"Order book analysis exported to {ob_file}")
    
    # 3. Export market microstructure analysis
    ms_df = analyze_market_microstructure(df, product)
    if not ms_df.empty:
        ms_file = os.path.join(output_dir, f"{product}_market_microstructure.csv")
        ms_df.to_csv(ms_file, index=False)
        exported_files.append(ms_file)
        print(f"Market microstructure analysis exported to {ms_file}")
    
    # 4. Export trading alpha signals
    alpha_df = calculate_trading_alpha_signals(df, product)
    if not alpha_df.empty:
        alpha_file = os.path.join(output_dir, f"{product}_alpha_signals.csv")
        alpha_df.to_csv(alpha_file, index=False)
        exported_files.append(alpha_file)
        print(f"Alpha signals exported to {alpha_file}")
    
    # 5. Export summary statistics
    summary_stats = {
        'product': product,
        'total_rows': len(df),
        'date_range': f"{raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}",
        'avg_spread': raw_data['spread'].mean(),
        'avg_spread_pct': raw_data['spread_pct'].mean(),
        'avg_volume': raw_data['total_volume'].mean(),
        'avg_bid_liquidity': ob_df['bid_liquidity'].mean() if not ob_df.empty else None,
        'avg_ask_liquidity': ob_df['ask_liquidity'].mean() if not ob_df.empty else None,
        'avg_effective_spread': ms_df['effective_spread'].mean() if not ms_df.empty else None,
        'avg_price_impact': ms_df['price_impact'].abs().mean() if not ms_df.empty else None,
        'avg_amihud': ms_df['amihud'].mean() if not ms_df.empty else None,
        'avg_vpin': ms_df['vpin_proxy'].mean() if not ms_df.empty else None,
        'avg_roll_spread': ms_df['roll_spread'].mean() if not ms_df.empty else None,
        'avg_kyle_lambda': ms_df['kyle_lambda_20'].mean() if not ms_df.empty else None,
        'return_autocorr': ms_df['return_autocorr_1'].mean() if not ms_df.empty else None,
        'volatility': ms_df['volatility_20'].mean() if not ms_df.empty else None,
        'mean_rev_signal': alpha_df['mean_rev_signal'].mean() if not alpha_df.empty else None,
        'trend_signal_dist': str(alpha_df['trend_signal'].value_counts(normalize=True).to_dict()) if not alpha_df.empty else None,
        'breakout_freq': (alpha_df['breakout_signal'] != 0).mean() if not alpha_df.empty else None,
        'avg_sharpe': alpha_df['rolling_sharpe'].mean() if not alpha_df.empty else None,
        'avg_sortino': alpha_df['rolling_sortino'].mean() if not alpha_df.empty else None,
        'max_drawdown': alpha_df['max_drawdown_50'].min() if not alpha_df.empty else None,
        'alpha_signal_range': f"{alpha_df['alpha_signal_norm'].min():.2f} to {alpha_df['alpha_signal_norm'].max():.2f}" if not alpha_df.empty else None
    }
    
    summary_file = os.path.join(output_dir, f"{product}_summary_stats.csv")
    pd.DataFrame([summary_stats]).to_csv(summary_file, index=False)
    exported_files.append(summary_file)
    print(f"Summary statistics exported to {summary_file}")
    
    # 6. Export daily aggregated data
    daily_data = raw_data.groupby('date').agg({
        'mid_price': ['first', 'max', 'min', 'last', 'mean', 'std'],
        'total_volume': 'sum',
        'spread': 'mean',
        'spread_pct': 'mean',
        'profit_and_loss': 'sum'
    }).reset_index()
    
    daily_data.columns = ['date', 'open', 'high', 'low', 'close', 'mean_price', 'price_std', 
                         'total_volume', 'avg_spread', 'avg_spread_pct', 'daily_pnl']
    
    daily_file = os.path.join(output_dir, f"{product}_daily_data.csv")
    daily_data.to_csv(daily_file, index=False)
    exported_files.append(daily_file)
    print(f"Daily aggregated data exported to {daily_file}")
    
    return exported_files

##########################
# MAIN FUNCTION / RUNNER #
##########################

def open_html_in_browser(file_path):
    """
    Open an HTML file in the default web browser
    
    Args:
        file_path (str): Path to the HTML file
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)
    # Convert to file URL
    file_url = f'file://{abs_path}'
    # Open in browser
    webbrowser.open(file_url)

def main():
    """
    Main function to run the enhanced trading visualizer
    """
    parser = argparse.ArgumentParser(description='Enhanced trade data visualizer.')
    parser.add_argument('file', help='Path to the CSV file containing trade data')
    parser.add_argument('--product', help='Filter data for a specific product')
    parser.add_argument('--output', help='Directory to save visualization images and HTML files')
    parser.add_argument('--interactive', action='store_true', help='Create interactive visualizations')
    parser.add_argument('--order-book', action='store_true', help='Perform order book analysis')
    parser.add_argument('--microstructure', action='store_true', help='Perform market microstructure analysis')
    parser.add_argument('--alpha', action='store_true', help='Calculate trading alpha signals')
    parser.add_argument('--all', action='store_true', help='Perform all analyses')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing large files')
    parser.add_argument('--open-browser', action='store_true', help='Open visualizations in browser automatically')
    parser.add_argument('--export-csv', action='store_true', help='Export analysis data to CSV files for LLM consumption')
    
    args = parser.parse_args()
    
    print(f"Parsing trade data from {args.file}...")
    df = parse_trade_data_optimized(args.file, chunksize=args.chunk_size)
    
    if df.empty:
        print("No data was parsed. Exiting.")
        return
    
    products = [args.product] if args.product else df['product'].unique()
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"Output will be saved to {args.output}")
    
    generated_files = []  # Keep track of generated files
    exported_csv_files = []  # Keep track of exported CSV files
    
    for product in products:
        print(f"\n=== Processing {product} ===")
        
        product_df = df[df['product'] == product].copy()
        
        if product_df.empty:
            print(f"No data found for product: {product}")
            continue
        
        # Export data to CSV if requested
        if args.export_csv:
            print(f"\nExporting analysis data for {product} to CSV files...")
            csv_files = export_analysis_to_csv(product_df, product, args.output)
            exported_csv_files.extend(csv_files)
        
        if args.interactive or args.all:
            print("\nCreating interactive visualizations...")
            dashboard = create_interactive_dashboard(df, product, args.output)
            if dashboard:
                dashboard_file = os.path.join(args.output, f"{product}_interactive_dashboard.html")
                generated_files.append(dashboard_file)
            
            heatmap = create_order_book_heatmap(df, product, args.output)
            if heatmap:
                heatmap_file = os.path.join(args.output, f"{product}_ob_heatmap.html")
                generated_files.append(heatmap_file)
            
            pv_chart = create_price_volume_interactive(df, product, args.output)
            if pv_chart:
                pv_file = os.path.join(args.output, f"{product}_price_volume_interactive.html")
                generated_files.append(pv_file)
        
        if args.order_book or args.all:
            print("\nPerforming order book analysis...")
            ob_df = analyze_order_book_dynamics(df, product)
            if not ob_df.empty:
                visualize_order_book_analysis(ob_df, args.output)
                # Add order book analysis files
                for file in ['ob_skew_vs_price.html', 'liquidity_vs_spread.html', 
                           'order_book_shape.html', 'volume_concentration.html', 
                           'book_pressure.html']:
                    generated_files.append(os.path.join(args.output, file))
        
        if args.microstructure or args.all:
            print("\nPerforming market microstructure analysis...")
            ms_df = analyze_market_microstructure(df, product)
            if not ms_df.empty:
                visualize_market_microstructure(ms_df, args.output)
                # Add microstructure analysis files
                for file in ['spread_analysis.html', 'price_impact.html', 
                           'order_flow_toxicity.html', 'market_efficiency.html', 
                           'volatility.html']:
                    generated_files.append(os.path.join(args.output, file))
        
        if args.alpha or args.all:
            print("\nCalculating trading alpha signals...")
            alpha_df = calculate_trading_alpha_signals(df, product)
            if not alpha_df.empty:
                visualize_trading_signals(alpha_df, args.output)
                # Add alpha signal files
                for file in ['bollinger_bands.html', 'rsi_indicator.html', 
                           'macd_indicator.html', 'order_book_signals.html', 
                           'alpha_signal.html', 'performance_metrics.html', 
                           'drawdown_analysis.html']:
                    generated_files.append(os.path.join(args.output, file))
    
    print("\nAll analyses completed!")
    
    # Print summary of exported CSV files
    if args.export_csv and exported_csv_files:
        print("\n=== Exported CSV Files ===")
        for file in exported_csv_files:
            print(f"- {file}")
        print("\nThese CSV files contain all the analysis data and can be uploaded to an LLM for further analysis.")
    
    # Open files in browser if requested
    if args.open_browser and generated_files:
        print("\nOpening visualizations in browser...")
        # Open the main dashboard first
        main_dashboard = next((f for f in generated_files if 'interactive_dashboard.html' in f), None)
        if main_dashboard:
            open_html_in_browser(main_dashboard)
            # Wait a bit before opening other files to avoid overwhelming the browser
            time.sleep(2)
        
        # Open other files
        for file in generated_files:
            if file != main_dashboard:  # Skip the main dashboard as it's already open
                open_html_in_browser(file)
                time.sleep(1)  # Small delay between opening files

if __name__ == "__main__":
    main()