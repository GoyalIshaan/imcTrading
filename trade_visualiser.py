import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import os

def parse_trade_data(file_path):
    """
    Parse the CSV file with semicolon-delimited data in a single column
    """
    # Read the CSV file with a single column
    df = pd.read_csv(file_path, header=None, names=['combined_data'])
    
    # Split the combined column into separate columns
    columns = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 
               'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 
               'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 
               'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
    
    # Split the combined data column by semicolon
    df_split = df['combined_data'].str.split(';', expand=True)
    
    # Ensure we have the right number of columns
    if df_split.shape[1] == len(columns):
        df_split.columns = columns
    else:
        # If column count doesn't match, try to handle it gracefully
        available_columns = df_split.shape[1]
        print(f"Warning: Expected {len(columns)} columns but found {available_columns}")
        df_split.columns = columns[:available_columns]
    
    # Convert numeric columns
    numeric_columns = ['bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 
                       'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 
                       'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 
                       'mid_price', 'profit_and_loss']
    
    for col in numeric_columns:
        if col in df_split.columns:
            df_split[col] = pd.to_numeric(df_split[col], errors='coerce')
    
    # Convert timestamp to numeric if it's not already
    if 'timestamp' in df_split.columns:
        df_split['timestamp'] = pd.to_numeric(df_split['timestamp'], errors='coerce')
    
    return df_split
def plot_missed_opportunities(df, product, output_dir=None, position_limit=20):
    """
    Plot missed EV zones due to position limits
    """
    product_df = df[df['product'] == product].copy()
    product_df = product_df.sort_values(by='timestamp').reset_index(drop=True)

    # Simulate position signal from price changes
    product_df['position_signal'] = np.sign(product_df['mid_price'].diff().fillna(0))
    product_df['position'] = product_df['position_signal'].cumsum().clip(-position_limit, position_limit)

    # Mark when position is capped
    product_df['at_limit'] = (abs(product_df['position']) >= position_limit)

    # Calculate return over short horizon (5 steps)
    horizon = 5
    product_df['future_return'] = product_df['mid_price'].shift(-horizon) - product_df['mid_price']

    # Identify missed EV opportunities
    def is_missed(row):
        if row['position'] == position_limit and row['future_return'] > 0:
            return True
        elif row['position'] == -position_limit and row['future_return'] < 0:
            return True
        return False

    product_df['missed_ev'] = product_df.apply(is_missed, axis=1)

    # Plot with missed EV zones highlighted
    plt.figure(figsize=(14, 6))
    plt.plot(product_df['timestamp'], product_df['mid_price'], label='Mid Price', color='blue')

    # Shade missed EV regions
    for _, row in product_df[product_df['missed_ev']].iterrows():
        plt.axvspan(row['timestamp'], row['timestamp'] + horizon * 100, color='lightgreen', alpha=0.3)

    plt.title(f'Missed EV Zones Due to Position Caps – {product}')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.legend()
    plt.grid(True)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{product}_missed_ev.png"))
        plt.close()
    

def inject_missed_opportunity_plots(df, output_dir=None):
    """
    Wrapper to loop over all products and generate missed EV plots
    """
    for product in df['product'].unique():
        plot_missed_opportunities(df, product, output_dir)
        print(f"Missed EV plot created for {product}")
def create_visualizations(df, product=None, output_dir=None):
    """
    Create various visualizations from the trade data
    """
    print("\n=== Debugging Data ===")
    print("Total rows:", len(df))
    print("Columns:", df.columns)
    print("Unique Products:", df['product'].unique())
    
    # Print the first few rows of numeric columns
    numeric_columns = ['bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', 'mid_price']
    print("\nFirst few rows of numeric columns:")
    print(df[numeric_columns].head())
    
    # Check for NaN or zero values
    print("\nMissing Values:")
    print(df[numeric_columns].isna().sum())
    
    print("\nColumn Value Ranges:")
    for col in numeric_columns:
        print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")
    
    # If a specific product is requested, filter the dataframe
    if product:
        df = df[df['product'] == product]
        
        if df.empty:
            print(f"No data found for product: {product}")
            return
    
    # Group by product for analysis
    products = df['product'].unique()
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # For each product, create a set of visualizations
    for product in products:
        product_df = df[df['product'] == product].copy()
        
        # Ensure numeric columns are properly typed and handle NaN values
        bid_volume_cols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
        ask_volume_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
        bid_price_cols = ['bid_price_1', 'bid_price_2', 'bid_price_3']
        ask_price_cols = ['ask_price_1', 'ask_price_2', 'ask_price_3']
        
        # Ensure numeric type and fill NaN
        for cols in [bid_volume_cols, ask_volume_cols, bid_price_cols, ask_price_cols]:
            for col in cols:
                product_df[col] = pd.to_numeric(product_df[col], errors='coerce').fillna(0)
        
        # 1. Price Movement and P&L Chart
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(product_df['timestamp'], product_df['mid_price'], label='Mid Price', color='blue')
        ax1.plot(product_df['timestamp'], product_df['bid_price_1'], label='Best Bid', color='green', alpha=0.5)
        ax1.plot(product_df['timestamp'], product_df['ask_price_1'], label='Best Ask', color='red', alpha=0.5)
        ax1.set_title(f'Price Movement for {product}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # P&L chart
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(product_df['timestamp'], product_df['profit_and_loss'], label='P&L', color='purple')
        ax2.set_title('Profit and Loss')
        ax2.set_ylabel('P&L')
        ax2.grid(True)
    

        
        # Spread chart
        ax3 = plt.subplot(gs[2], sharex=ax1)
        product_df['spread'] = product_df['ask_price_1'] - product_df['bid_price_1']
        ax3.plot(product_df['timestamp'], product_df['spread'], label='Bid-Ask Spread', color='orange')
        ax3.set_title('Bid-Ask Spread')
        ax3.set_xlabel('Timestamp')
        ax3.set_ylabel('Spread')
        ax3.grid(True)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{product}_price_pnl.png"))
        
        # 2. Order Book Depth Visualization (as a heatmap)
        plt.figure(figsize=(12, 10))
        
        # Sample the data to reduce visualization complexity
        sample_rate = max(1, len(product_df) // 100)
        sampled_df = product_df.iloc[::sample_rate].reset_index(drop=True)
        
        # Create volume matrices for visualization
        bid_volumes = sampled_df[bid_volume_cols].values
        ask_volumes = sampled_df[ask_volume_cols].values
        
        # Normalize volumes safely
        max_volume = max(bid_volumes.max(), ask_volumes.max())
        bid_volumes = bid_volumes / max_volume if max_volume > 0 else bid_volumes
        ask_volumes = ask_volumes / max_volume if max_volume > 0 else ask_volumes
        
        # Combine into one matrix for the heatmap (bids negative, asks positive)
        order_book = np.zeros((len(sampled_df), 6))
        order_book[:, 0:3] = -bid_volumes  # Negative for bids
        order_book[:, 3:6] = ask_volumes
        
        plt.imshow(order_book, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
        
        # Add labels
        tick_labels = bid_price_cols + ask_price_cols
        if len(sampled_df) > 10:
            plt.yticks(np.arange(0, len(sampled_df), len(sampled_df)//10), 
                       sampled_df['timestamp'].iloc[::len(sampled_df)//10])
        else:
            plt.yticks(np.arange(0, len(sampled_df)), 
                       sampled_df['timestamp'])
            plt.xticks(np.arange(6), tick_labels, rotation=45)
        
        plt.title(f'Order Book Depth Heatmap for {product}')
        plt.xlabel('Price Levels')
        plt.ylabel('Time')
        plt.colorbar(label='Normalized Volume (Green=Ask, Red=Bid)')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{product}_order_book.png"))
        
        # 4. Market Imbalance Analysis
        plt.figure(figsize=(12, 6))
        
        # Calculate imbalance metric (total bid volume vs total ask volume)
        product_df['total_bid_volume'] = product_df[bid_volume_cols].sum(axis=1)
        product_df['total_ask_volume'] = product_df[ask_volume_cols].sum(axis=1)
        
        # Compute volume imbalance with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            product_df['volume_imbalance'] = (
                product_df['total_bid_volume'] - product_df['total_ask_volume']
            ) / (product_df['total_bid_volume'] + product_df['total_ask_volume'])
        
        # Replace infinite values with 0
        product_df['volume_imbalance'] = product_df['volume_imbalance'].replace([np.inf, -np.inf], 0)
        
        plt.plot(product_df['timestamp'], product_df['volume_imbalance'], label='Order Book Imbalance', color='purple')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'Market Imbalance for {product}')
        plt.xlabel('Timestamp')
        plt.ylabel('Imbalance (Bid-Ask)/(Bid+Ask)')
        plt.legend()
        plt.grid(True)
        # 1. Price Movement and P&L Chart
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(product_df['timestamp'], product_df['mid_price'], label='Mid Price', color='blue')
        ax1.plot(product_df['timestamp'], product_df['bid_price_1'], label='Best Bid', color='green', alpha=0.5)
        ax1.plot(product_df['timestamp'], product_df['ask_price_1'], label='Best Ask', color='red', alpha=0.5)
        ax1.set_title(f'Price Movement for {product}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        buy_indices = product_df.index[::100]     # Simulate buys every 100 rows
        sell_indices = product_df.index[50::100]  # Simulate sells offset by 50

        # Plot buy trades (green upward triangle)
        ax1.scatter(product_df.loc[buy_indices, 'timestamp'],
                    product_df.loc[buy_indices, 'mid_price'],
                    color='green', marker='^', label='Buy Trade', s=50, zorder=5)

        # Plot sell trades (red downward triangle)
        ax1.scatter(product_df.loc[sell_indices, 'timestamp'],
                    product_df.loc[sell_indices, 'mid_price'],
                    color='red', marker='v', label='Sell Trade', s=50, zorder=5)
                
        if output_dir:
                    plt.savefig(os.path.join(output_dir, f"{product}_imbalance.png"))
        inject_missed_opportunity_plots(product_df, output_dir)

     

    # If not saving to files, show the plots
    if not output_dir:
        plt.show()
    else:
        print(f"Visualizations saved to {output_dir}")
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Visualize trading data from CSV files.')
    parser.add_argument('file', help='Path to the CSV file containing trade data')
    parser.add_argument('--product', help='Filter data for a specific product')
    parser.add_argument('--output', help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    print(f"Parsing trade data from {args.file}...")
    df = parse_trade_data(args.file)
    
    print(f"Creating visualizations...")
    create_visualizations(df, args.product, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
# Existing code remains the same...

# NEW CODE STARTS HERE - add this after the existing main() function
def analyze_positions(df):
    """
    Comprehensive position analysis function
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
    
    Returns:
        dict: Detailed position analysis metrics
    """
    # Group by product to analyze positions
    position_analysis = {}
    
    for product in df['product'].unique():
        product_df = df[df['product'] == product].copy()
        
        # Position tracking metrics
        product_analysis = {
            # Basic position statistics
            'total_trades': len(product_df),
            'avg_trade_size': product_df['mid_price'].mean(),
            
            # Position sizing analysis
            'max_position': product_df['mid_price'].max(),
            'min_position': product_df['mid_price'].min(),
            
            # Volatility and risk metrics
            'position_std_dev': product_df['mid_price'].std(),
            'position_variance': product_df['mid_price'].var(),
            
            # Time-based position metrics
            'position_holding_time': {
                'mean': None,
                'median': None,
                'max': None,
                'min': None
            },
            
            # Risk-adjusted metrics
            'sharpe_ratio': None,
            'sortino_ratio': None,
        }
        
        # Advanced position tracking
        product_analysis['position_distribution'] = {
            'percentiles': {
                '10th': np.percentile(product_df['mid_price'], 10),
                '25th': np.percentile(product_df['mid_price'], 25),
                '50th': np.percentile(product_df['mid_price'], 50),
                '75th': np.percentile(product_df['mid_price'], 75),
                '90th': np.percentile(product_df['mid_price'], 90),
            }
        }
        
        # Position concentration analysis
        product_analysis['position_concentration'] = {
            'unique_prices': len(product_df['mid_price'].unique()),
            'price_range': product_df['mid_price'].max() - product_df['mid_price'].min(),
        }
        
        position_analysis[product] = product_analysis
    
    return position_analysis

def visualize_positions(df, output_dir=None):
    """
    Create advanced position visualization
    
    Args:
        df (pd.DataFrame): DataFrame with trade data
        output_dir (str, optional): Directory to save visualizations
    """
    # Analyze positions
    position_analysis = analyze_positions(df)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Position metrics grid
    gs = GridSpec(2, 2)
    
    # 1. Position Distribution Boxplot
    ax1 = plt.subplot(gs[0, 0])
    product_positions = {}
    for product in df['product'].unique():
        product_positions[product] = df[df['product'] == product]['mid_price']
    
    ax1.boxplot(list(product_positions.values()), labels=list(product_positions.keys()))
    ax1.set_title('Position Distribution by Product')
    ax1.set_ylabel('Mid Price')
    
    # 2. Position Concentration Heatmap
    ax2 = plt.subplot(gs[0, 1])
    concentration_data = [
        position_analysis[product]['position_concentration']['unique_prices'] 
        for product in position_analysis.keys()
    ]
    sns.heatmap(
        [concentration_data], 
        annot=True, 
        cmap='YlGnBu', 
        xticklabels=list(position_analysis.keys()),
        cbar_kws={'label': 'Unique Price Levels'}
    )
    ax2.set_title('Position Concentration')
    
    # 3. Detailed Position Metrics Table
    ax3 = plt.subplot(gs[1, :])
    metrics_table = []
    headers = ['Product', 'Avg Price', 'Max Price', 'Min Price', 'Price StdDev', 'Unique Prices']
    
    for product, analysis in position_analysis.items():
        metrics_table.append([
            product,
            f"{analysis['avg_trade_size']:.2f}",
            f"{analysis['max_position']:.2f}",
            f"{analysis['min_position']:.2f}",
            f"{analysis['position_std_dev']:.2f}",
            analysis['position_concentration']['unique_prices']
        ])
    
    ax3.axis('off')
    table = ax3.table(
        cellText=metrics_table,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax3.set_title('Detailed Position Metrics')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'position_analysis.png'))
        plt.close()
    else:
        plt.show()
    
    # Print detailed analysis
    print("\n=== Detailed Position Analysis ===")
    for product, analysis in position_analysis.items():
        print(f"\nProduct: {product}")
        print(f"Total Trades: {analysis['total_trades']}")
        print(f"Average Trade Price: {analysis['avg_trade_size']:.2f}")
        print(f"Price Range: {analysis['min_position']:.2f} - {analysis['max_position']:.2f}")
        print(f"Price Standard Deviation: {analysis['position_std_dev']:.2f}")
        print("Position Distribution Percentiles:")
        for percentile, value in analysis['position_distribution']['percentiles'].items():
            print(f"  {percentile}: {value:.2f}")
 


# Modify the main function to include position visualization
def main():
    parser = argparse.ArgumentParser(description='Visualize trading data from CSV files.')
    parser.add_argument('file', help='Path to the CSV file containing trade data')
    parser.add_argument('--product', help='Filter data for a specific product')
    parser.add_argument('--output', help='Directory to save visualization images')
    # Add this line to support the --positions flag
    parser.add_argument('--positions', action='store_true', help='Generate position analysis')
    
    args = parser.parse_args()
    
    print(f"Parsing trade data from {args.file}...")
    df = parse_trade_data(args.file)
    
    print(f"Creating visualizations...")
    create_visualizations(df, args.product, args.output)
    
    # Add this block to handle position analysis
    if args.positions:
        visualize_positions(df, args.output)
    
    print("Done!")
    