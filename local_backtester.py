import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict, List, Any

# Import from your datamodel
from datamodel import (
    Order, TradingState, OrderDepth, Listing, 
    Observation, Trade
)
from trader import Trader  # Your trading algorithm

class LocalBacktester:
    def __init__(self, csv_path: str, trader_class: type, max_position: int = 20):
        """
        Initialize backtester with CSV data and trading strategy
        """
        # Read CSV file
        try:
            self.df = pd.read_csv(csv_path, sep=';')
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise
        
        # Validate CSV columns
        required_columns = [
            'timestamp', 'product', 
            'bid_price_1', 'bid_volume_1', 
            'ask_price_1', 'ask_volume_1', 
            'mid_price'
        ]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Initialize trader and backtesting parameters
        self.trader = trader_class()
        self.max_position = max_position
        
        # Performance tracking
        self.trades = []
        self.portfolio_value = []
        self.positions = {}
        self.cash = 0.0
    
    def prepare_trading_state(self, row: pd.Series) -> TradingState:
        """
        Prepare trading state for a given row of market data
        """
        # Create order depth
        product = row['product']
        order_depth = OrderDepth()
        
        # Add buy orders
        for i in range(1, 4):
            bid_price_col = f'bid_price_{i}'
            bid_volume_col = f'bid_volume_{i}'
            if not pd.isna(row[bid_price_col]) and not pd.isna(row[bid_volume_col]):
                order_depth.buy_orders[int(row[bid_price_col])] = int(row[bid_volume_col])
        
        # Add sell orders
        for i in range(1, 4):
            ask_price_col = f'ask_price_{i}'
            ask_volume_col = f'ask_volume_{i}'
            if not pd.isna(row[ask_price_col]) and not pd.isna(row[ask_volume_col]):
                order_depth.sell_orders[int(row[ask_price_col])] = -int(row[ask_volume_col])
        
        # Create trading state
        state = TradingState(
            traderData="",
            timestamp=int(row['timestamp']),
            listings={product: Listing(product, product, product)},
            order_depths={product: order_depth},
            own_trades={},
            market_trades={},
            position=self.positions,
            observations=Observation({}, {})
        )
        
        return state
    
    def execute_trades(self, state: TradingState, orders: Dict[str, List[Order]], mid_price: float):
    
        for product, product_orders in orders.items():
            print(f"\n--- Processing Product: {product} ---")
            
            # Get current order depth
            order_depth = state.order_depths.get(product)
            if not order_depth:
                print(f"No order depth found for {product}")
                continue
            
            print("Buy Orders:", order_depth.buy_orders)
            print("Sell Orders:", order_depth.sell_orders)
            
            for order in product_orders:
                print(f"\nCurrent Order: {order}")
                
                # Check position limits
                current_position = self.positions.get(product, 0)
                if current_position + order.quantity > self.max_position or \
                current_position + order.quantity < -self.max_position:
                    print(f"Position limit exceeded. Current: {current_position}, Order: {order.quantity}")
                    continue
                
                # Detailed matching logic
                if order.quantity > 0:  # Buy order
                    matching_sells = [
                        (price, volume) for price, volume in order_depth.sell_orders.items() 
                        if price <= order.price
                    ]
                    
                    print(f"Buy order matching sells: {matching_sells}")
                    
                    if matching_sells:
                        # Sort by lowest sell price
                        matching_sells.sort(key=lambda x: x[0])
                        
                        # Execute trade
                        exec_price, exec_volume = matching_sells[0]
                        trade_volume = min(order.quantity, abs(exec_volume))
                        
                        print(f"Executing buy: price={exec_price}, volume={trade_volume}")
                        
                        # Update positions and cash
                        self.positions[product] = self.positions.get(product, 0) + trade_volume
                        self.cash -= trade_volume * exec_price
                        
                        # Record trade
                        self.trades.append({
                            'timestamp': state.timestamp,
                            'product': product,
                            'type': 'BUY',
                            'price': exec_price,
                            'volume': trade_volume
                        })
                    else:
                        print("No matching sell orders found")
                
                elif order.quantity < 0:  # Sell order
                    matching_buys = [
                        (price, volume) for price, volume in order_depth.buy_orders.items() 
                        if price >= abs(order.price)
                    ]
                    
                    print(f"Sell order matching buys: {matching_buys}")
                    
                    if matching_buys:
                        # Sort by highest buy price
                        matching_buys.sort(key=lambda x: x[0], reverse=True)
                        
                        # Execute trade
                        exec_price, exec_volume = matching_buys[0]
                        trade_volume = min(abs(order.quantity), exec_volume)
                        
                        print(f"Executing sell: price={exec_price}, volume={trade_volume}")
                        
                        # Update positions and cash
                        self.positions[product] = self.positions.get(product, 0) - trade_volume
                        self.cash += trade_volume * exec_price
                        
                        # Record trade
                        self.trades.append({
                            'timestamp': state.timestamp,
                            'product': product,
                            'type': 'SELL',
                            'price': exec_price,
                            'volume': trade_volume
                        })
                    else:
                        print("No matching buy orders found")
        
        return orders
    def run_backtest(self):
        """
        Run the full backtest
        
        Returns:
            dict: Performance metrics
        """
        # Group by unique timestamps and products
        grouped = self.df.groupby(['timestamp', 'product'])
        
        for (timestamp, product), group in grouped:
            row = group.iloc[0]
            
            # Prepare trading state
            state = self.prepare_trading_state(row)
            
            # Run trading algorithm
            try:
                orders, _, _ = self.trader.run(state)
                
                # Execute trades
                self.execute_trades(state, orders, row['mid_price'])
            except Exception as e:
                print(f"Error at timestamp {timestamp}, product {product}: {e}")
            
            # Calculate portfolio value
            current_value = self.cash
            for pos_product, position in self.positions.items():
                if pos_product == product:
                    current_value += position * row['mid_price']
            
            self.portfolio_value.append({
                'timestamp': timestamp,
                'product': product,
                'portfolio_value': current_value,
                'position': self.positions.get(product, 0),
                'mid_price': row['mid_price']
            })
        
        # Calculate performance metrics
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            dict: Performance metrics
        """
        pv_df = pd.DataFrame(self.portfolio_value)
        
        # Print debug information
        print(f"Portfolio Value Entries: {len(self.portfolio_value)}")
        print(f"Trades Recorded: {len(self.trades)}")
        
        # If no trades, create an empty DataFrame with expected columns
        if not self.trades:
            trades_df = pd.DataFrame(columns=['type', 'product', 'price', 'volume'])
        else:
            trades_df = pd.DataFrame(self.trades)
        
        # Handle case where initial portfolio value might be zero
        initial_value = pv_df['portfolio_value'].iloc[0] if not pv_df.empty else 0
        final_value = pv_df['portfolio_value'].iloc[-1] if not pv_df.empty else 0
        
        metrics = {
            'initial_portfolio_value': initial_value,
            'final_portfolio_value': final_value,
            'total_return': final_value - initial_value,
            'return_percentage': (final_value / initial_value - 1) * 100 if initial_value != 0 else 0,
            'total_trades': len(trades_df),
            'buy_trades': len(trades_df[trades_df['type'] == 'BUY']) if 'type' in trades_df.columns else 0,
            'sell_trades': len(trades_df[trades_df['type'] == 'SELL']) if 'type' in trades_df.columns else 0,
            'max_position': pv_df.groupby('product')['position'].max().to_dict() if not pv_df.empty else {}
        }
        
        # Print out metrics for debugging
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        return metrics
    
    def plot_performance(self):
        """
        Plot performance visualization
        Requires matplotlib
        """
        import matplotlib.pyplot as plt
        
        pv_df = pd.DataFrame(self.portfolio_value)
        trades_df = pd.DataFrame(self.trades)
        
        plt.figure(figsize=(15, 10))
        
        # Portfolio Value
        plt.subplot(2, 2, 1)
        plt.plot(pv_df['timestamp'], pv_df['portfolio_value'])
        plt.title('Portfolio Value')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        
        # Positions
        plt.subplot(2, 2, 2)
        for product in pv_df['product'].unique():
            product_df = pv_df[pv_df['product'] == product]
            plt.plot(product_df['timestamp'], product_df['position'], label=product)
        plt.title('Positions')
        plt.xlabel('Timestamp')
        plt.ylabel('Position')
        plt.legend()
        
        # Trade Distribution
        plt.subplot(2, 2, 3)
        trades_df['type'].value_counts().plot(kind='bar')
        plt.title('Trade Distribution')
        plt.xlabel('Trade Type')
        plt.ylabel('Count')
        
        # Mid Price
        plt.subplot(2, 2, 4)
        for product in pv_df['product'].unique():
            product_df = pv_df[pv_df['product'] == product]
            plt.plot(product_df['timestamp'], product_df['mid_price'], label=product)
        plt.title('Mid Price')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Command-line interface for running the backtester
    """
    parser = argparse.ArgumentParser(description='Run local backtester for trading strategy')
    parser.add_argument('csv_file', help='Path to the CSV file with market data')
    parser.add_argument('--max-position', type=int, default=20, 
                        help='Maximum position size per product (default: 20)')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot performance visualization')
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist.")
        return
    
    # Run backtester
    backtester = LocalBacktester(
        csv_path=args.csv_file, 
        trader_class=Trader, 
        max_position=args.max_position
    )
    
    # Run backtest and print metrics
    metrics = backtester.run_backtest()
    
    print("\n=== Backtest Performance Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Optional performance plot
    if args.plot:
        backtester.plot_performance()

if __name__ == "__main__":
    main()