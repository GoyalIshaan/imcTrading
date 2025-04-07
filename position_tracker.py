# position_tracker.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datamodel import OrderDepth, TradingState, Order, Observation
import os
from datetime import datetime
import warnings
import sys
sys.path.append('/Users/mavinkhanday/Desktop/imc-prosperity3')
from trader import Trade
warnings.filterwarnings('ignore')

class PositionTracker:
    def __init__(self, trader_class):
        """
        Initialize the position tracker with a trading algorithm
        
        Args:
            trader_class: The class of your trading algorithm
        """
        self.trader = trader_class()
        self.trader_data = ""
        self.positions = {}
        self.cash = 0
        self.trades = []
        self.position_history = []
        self.pnl_history = []
        self.drawdowns = []
        self.price_history = {}
        self.results_df = None
        self.metrics = {}
    
    def create_trading_state(self, row, position=None):
        """Convert a row of CSV data into a TradingState object"""
        if position is None:
            position = self.positions.copy()
            
        # Parse the row
        data = row.split(';')
        day, timestamp, product = data[0], int(data[1]), data[2]
        
        # Create order depths
        order_depths = {}
        order_depths[product] = OrderDepth()
        
        # Populate buy orders
        try:
            if data[3] and data[4]:  # bid_price_1, bid_volume_1
                order_depths[product].buy_orders[int(float(data[3]))] = int(float(data[4]))
            if data[5] and data[6]:  # bid_price_2, bid_volume_2
                order_depths[product].buy_orders[int(float(data[5]))] = int(float(data[6]))
            if data[7] and data[8]:  # bid_price_3, bid_volume_3
                order_depths[product].buy_orders[int(float(data[7]))] = int(float(data[8]))
            
            # Populate sell orders
            if data[9] and data[10]:  # ask_price_1, ask_volume_1
                order_depths[product].sell_orders[int(float(data[9]))] = -int(float(data[10]))
            if data[11] and data[12]:  # ask_price_2, ask_volume_2
                order_depths[product].sell_orders[int(float(data[11]))] = -int(float(data[12]))
            if data[13] and data[14]:  # ask_price_3, ask_volume_3
                order_depths[product].sell_orders[int(float(data[13]))] = -int(float(data[14]))
        except (IndexError, ValueError):
            # Handle malformed data gracefully
            pass
        
        # Store price for this product
        try:
            mid_price = float(data[15]) if len(data) > 15 and data[15] else 0
            if product not in self.price_history:
                self.price_history[product] = []
            self.price_history[product].append((timestamp, mid_price))
        except (IndexError, ValueError):
            mid_price = 0
            
        # Create a basic observation object with the mid price
        observations = Observation({product: mid_price}, {})
        
        # Create a minimal trading state
        state = TradingState(
            traderData=self.trader_data,
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=position,
            observations=observations
        )
        
        return state, product, mid_price, timestamp
    
    def run_backtest(self, csv_file, max_position=100):
        """
        Run a backtest using historical market data
        
        Args:
            csv_file: Path to CSV file with market data
            max_position: Maximum allowed position size
        """
        # Reset state
        self.trader_data = ""
        self.positions = {}
        self.cash = 0
        self.trades = []
        self.position_history = []
        self.pnl_history = []
        self.price_history = {}
        
        # Results storage
        results = []
        
        # Read CSV data
        print(f"Starting backtest on {csv_file}...")
        line_count = 0
        
        with open(csv_file, 'r') as f:
            # Skip header if present
            first_line = f.readline()
            if "day;timestamp" in first_line:
                pass
            else:
                f.seek(0)  # Reset if no header
            
            # Process each line
            for line in f:
                line_count += 1
                if not line.strip():
                    continue
                    
                try:
                    # Create trading state from this line
                    state, product, mid_price, timestamp = self.create_trading_state(line)
                    print(f"Current state - Product: {product}, Positions: {self.positions}")
                    if product in state.order_depths:
                        print(f"Order depths: Buy: {state.order_depths[product].buy_orders}, Sell: {state.order_depths[product].sell_orders}")
                        print(f"Mid price: {mid_price}, Timestamp: {timestamp}")
                    # Run trading algorithm on this state
                    orders, conversions, new_trader_data = self.trader.run(state)
                    self.trader_data = new_trader_data
                    
                    # Process the orders (simulate execution)
                    for symbol, order_list in orders.items():
                        for order in order_list:
                            # Apply position limits
                            new_position = self.positions.get(symbol, 0) + order.quantity
                            if abs(new_position) > max_position:
                                # Skip order that would exceed limits
                                continue
                                
                            # Simulate order execution at the requested price
                            if order.quantity > 0:  # Buy order
                                if state.order_depths[symbol].sell_orders:
                                    # Check if we can execute at requested price
                                    if min(state.order_depths[symbol].sell_orders.keys()) <= order.price:
                                        exec_price = order.price
                                        self.positions[symbol] = self.positions.get(symbol, 0) + order.quantity
                                        self.cash -= exec_price * order.quantity  # Pay money to buy
                                        self.trades.append({
                                            'timestamp': timestamp,
                                            'product': symbol,
                                            'price': exec_price,
                                            'quantity': order.quantity,
                                            'direction': 'BUY',
                                            'position': self.positions[symbol]
                                        })
                            elif order.quantity < 0:  # Sell order
                                if state.order_depths[symbol].buy_orders:
                                    # Check if we can execute at requested price
                                    if max(state.order_depths[symbol].buy_orders.keys()) >= order.price:
                                        exec_price = order.price
                                        self.positions[symbol] = self.positions.get(symbol, 0) + order.quantity
                                        self.cash += exec_price * abs(order.quantity)  # Receive money for selling
                                        self.trades.append({
                                            'timestamp': timestamp,
                                            'product': symbol,
                                            'price': exec_price,
                                            'quantity': order.quantity,
                                            'direction': 'SELL',
                                            'position': self.positions[symbol]
                                        })
                    
                    # Calculate current portfolio value and P&L
                    portfolio_value = self.cash
                    for prod, pos in self.positions.items():
                        price_data = self.price_history.get(prod, [])
                        if price_data:
                            last_price = price_data[-1][1]
                            portfolio_value += pos * last_price
                    
                    # Store position and P&L history
                    self.position_history.append({
                        'timestamp': timestamp,
                        'product': product,
                        'position': self.positions.get(product, 0),
                        'price': mid_price,
                        'portfolio_value': portfolio_value
                    })
                    
                    self.pnl_history.append(portfolio_value)
                    
                    # Calculate drawdown
                    current_max = max(self.pnl_history)
                    current_dd = (current_max - portfolio_value) / current_max if current_max > 0 else 0
                    self.drawdowns.append(current_dd)
                    
                    # Store detailed results
                    results.append({
                        'timestamp': timestamp,
                        'product': product,
                        'position': self.positions.get(product, 0),
                        'price': mid_price,
                        'portfolio_value': portfolio_value,
                        'cash': self.cash,
                        'drawdown': current_dd
                    })
                
                except Exception as e:
                    print(f"Error processing line {line_count}: {e}")
                    continue
        
        print(f"Backtest completed with {line_count} market states and {len(self.trades)} trades.")
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Calculate performance metrics
        self.calculate_metrics()
        
        return self.results_df
    
    def calculate_metrics(self):
        """Calculate performance metrics from backtest results"""
        if self.results_df is None or len(self.results_df) == 0:
            return
            
        # Get daily returns
        self.results_df['daily_return'] = self.results_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_days = len(self.results_df['timestamp'].unique())
        total_trades = len(self.trades)
        
        # Basic metrics
        self.metrics = {
            'initial_portfolio': self.results_df['portfolio_value'].iloc[0],
            'final_portfolio': self.results_df['portfolio_value'].iloc[-1],
            'absolute_return': self.results_df['portfolio_value'].iloc[-1] - self.results_df['portfolio_value'].iloc[0],
            'return_pct': (self.results_df['portfolio_value'].iloc[-1] / self.results_df['portfolio_value'].iloc[0] - 1) * 100 if self.results_df['portfolio_value'].iloc[0] > 0 else 0,
            'max_drawdown': max(self.drawdowns) * 100,
            'trades_per_day': total_trades / total_days if total_days > 0 else 0,
            'win_rate': sum(1 for t in self.trades if 
                           (t['direction'] == 'BUY' and t['price'] < self.price_history[t['product']][-1][1]) or
                           (t['direction'] == 'SELL' and t['price'] > self.price_history[t['product']][-1][1])
                          ) / total_trades if total_trades > 0 else 0,
        }
        
        # Calculate Sharpe Ratio (if we have enough data)
        if len(self.results_df) > 1:
            daily_returns = self.results_df['daily_return'].dropna()
            if len(daily_returns) > 0:
                mean_return = daily_returns.mean()
                std_return = daily_returns.std()
                risk_free_rate = 0.0  # Assuming 0% risk-free rate
                self.metrics['sharpe_ratio'] = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Maximum position
        max_positions = {}
        for product in self.results_df['product'].unique():
            product_df = self.results_df[self.results_df['product'] == product]
            max_positions[product] = product_df['position'].abs().max()
        self.metrics['max_positions'] = max_positions
        
        return self.metrics
    
    def visualize_performance(self, output_dir=None):
        """
        Create comprehensive performance visualizations
        
        Args:
            output_dir: Directory to save visualizations (if None, displays instead)
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No backtest results to visualize.")
            return
            
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get unique products
        products = self.results_df['product'].unique()
        
        # 1. Overall Performance Dashboard
        plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2)
        
        # Portfolio Value Chart
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(self.results_df['timestamp'], self.results_df['portfolio_value'], 'b-', linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontsize=14)
        ax1.set_ylabel('Value')
        ax1.grid(True)
        
        # Drawdown Chart
        ax2 = plt.subplot(gs[1, 0])
        ax2.fill_between(self.results_df['timestamp'], 0, self.results_df['drawdown'] * 100, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)', fontsize=14)
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        # Position Size Chart (stacked for multiple products)
        ax3 = plt.subplot(gs[1, 1])
        for product in products:
            product_df = self.results_df[self.results_df['product'] == product]
            ax3.plot(product_df['timestamp'], product_df['position'], label=product)
        ax3.set_title('Position Size by Product', fontsize=14)
        ax3.set_ylabel('Position')
        ax3.grid(True)
        ax3.legend()
        
        # Trade Analysis
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Trade Count by Direction
            ax4 = plt.subplot(gs[2, 0])
            trade_counts = trades_df['direction'].value_counts()
            ax4.bar(trade_counts.index, trade_counts.values, color=['green', 'red'])
            ax4.set_title('Trade Count by Direction', fontsize=14)
            ax4.set_ylabel('Count')
            
            # Trade Distribution by Price
            ax5 = plt.subplot(gs[2, 1])
            for product in trades_df['product'].unique():
                product_trades = trades_df[trades_df['product'] == product]
                sns.kdeplot(product_trades['price'], label=product, ax=ax5)
            ax5.set_title('Trade Price Distribution', fontsize=14)
            ax5.set_xlabel('Price')
            ax5.legend()
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'overall_performance.png'))
        else:
            plt.show()
        
        # 2. Product-Specific Performance
        for product in products:
            product_df = self.results_df[self.results_df['product'] == product]
            
            plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
            
            # Price and Trades
            ax1 = plt.subplot(gs[0])
            ax1.plot(product_df['timestamp'], product_df['price'], label='Price', color='blue')
            
            # Mark buy and sell trades
            product_trades = [t for t in self.trades if t['product'] == product]
            buys = [t for t in product_trades if t['direction'] == 'BUY']
            sells = [t for t in product_trades if t['direction'] == 'SELL']
            
            if buys:
                buy_times = [t['timestamp'] for t in buys]
                buy_prices = [t['price'] for t in buys]
                ax1.scatter(buy_times, buy_prices, marker='^', color='green', s=100, label='Buy')
                
            if sells:
                sell_times = [t['timestamp'] for t in sells]
                sell_prices = [t['price'] for t in sells]
                ax1.scatter(sell_times, sell_prices, marker='v', color='red', s=100, label='Sell')
            
            ax1.set_title(f'Price and Trades for {product}', fontsize=14)
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Position Size
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(product_df['timestamp'], product_df['position'], color='purple')
            ax2.fill_between(product_df['timestamp'], 0, product_df['position'], alpha=0.2, color='purple')
            ax2.set_title('Position Size', fontsize=14)
            ax2.set_ylabel('Position')
            ax2.grid(True)
            
            # P&L Attribution (estimated)
            if len(product_df) > 1:
                ax3 = plt.subplot(gs[2], sharex=ax1)
                product_df['position_value'] = product_df['position'] * product_df['price']
                product_df['pnl_change'] = product_df['position_value'].diff()
                ax3.bar(product_df['timestamp'], product_df['pnl_change'], color='green', 
                       width=10, alpha=0.6)
                ax3.set_title('Estimated P&L Change', fontsize=14)
                ax3.set_xlabel('Timestamp')
                ax3.set_ylabel('Value Change')
                ax3.grid(True)
            
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{product}_analysis.png'))
            else:
                plt.show()
        
        # 3. Performance Metrics Summary
        plt.figure(figsize=(12, 8))
        metrics_to_plot = {
            'return_pct': 'Return (%)',
            'max_drawdown': 'Max Drawdown (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'win_rate': 'Win Rate (%)'
        }
        
        values = []
        for key in metrics_to_plot:
            if key in self.metrics:
                # Convert to percentage for win_rate
                if key == 'win_rate':
                    values.append(self.metrics[key] * 100)
                else:
                    values.append(self.metrics[key])
            else:
                values.append(0)
                
        bars = plt.bar(list(metrics_to_plot.values()), values, color=['green', 'red', 'blue', 'purple'])
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Performance Metrics Summary', fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
        else:
            plt.show()
            
        # Print metrics summary
        print("\n=== Performance Metrics Summary ===")
        print(f"Initial Portfolio Value: {self.metrics['initial_portfolio']:.2f}")
        print(f"Final Portfolio Value: {self.metrics['final_portfolio']:.2f}")
        print(f"Absolute Return: {self.metrics['absolute_return']:.2f}")
        print(f"Return (%): {self.metrics['return_pct']:.2f}%")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        if 'sharpe_ratio' in self.metrics:
            print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {self.metrics['win_rate']*100:.2f}%")
        print(f"Trades per Day: {self.metrics['trades_per_day']:.2f}")
        print("\nMaximum Positions:")
        for product, max_pos in self.metrics['max_positions'].items():
            print(f"  {product}: {max_pos}")
        print()
        
        return True

# Example usage
def run_position_tracking(csv_file, trader_class, output_dir=None):
    """
    Run position tracking and visualization
    
    Args:
        csv_file: Path to CSV file with market data
        trader_class: Your trading algorithm class
        output_dir: Directory to save visualizations (if None, displays instead)
    """
    tracker = PositionTracker(trader_class)
    tracker.run_backtest(csv_file)
    tracker.visualize_performance(output_dir)
    
    return tracker

# If running as a standalone script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Track positions and performance for a trading algorithm')
    parser.add_argument('csv_file', help='Path to CSV file with market data')
    parser.add_argument('--output', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Import your Trader class
    # Change this import to match your trader implementation
    from trader import Trader
    
    run_position_tracking(args.csv_file, Trader, args.output)