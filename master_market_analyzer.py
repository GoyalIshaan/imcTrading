import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ta
from typing import Dict, List, Optional
import os

class MarketAnalyzer:
    def __init__(self):
        plt.style.use('default')
        sns.set_theme()
        self.data = {}
        self.products = []
        
    def load_squid_ink_data(self, filepath: str = 'squid_ink_ml_data.csv') -> None:
        """Load and prepare SQUID_INK data"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return
            
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['product'] = 'SQUID_INK'
        
        # Calculate technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['mid_price']).rsi()
        macd = ta.trend.MACD(df['mid_price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Calculate rolling statistics
        df['price_std'] = df['mid_price'].rolling(window=20).std()
        df['price_mean'] = df['mid_price'].rolling(window=20).mean()
        
        self.data['SQUID_INK'] = df
        if 'SQUID_INK' not in self.products:
            self.products.append('SQUID_INK')
            
    def load_nopen_data(self, filepath: str = 'nopen.csv') -> None:
        """Load and prepare nopen data"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return
            
        df = pd.read_csv(filepath, sep=';')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Process each product separately
        for product in df['product'].unique():
            product_df = df[df['product'] == product].copy()
            
            # Calculate technical indicators
            product_df['rsi'] = ta.momentum.RSIIndicator(product_df['mid_price']).rsi()
            macd = ta.trend.MACD(product_df['mid_price'])
            product_df['macd'] = macd.macd()
            product_df['macd_signal'] = macd.macd_signal()
            product_df['macd_diff'] = macd.macd_diff()
            
            # Calculate rolling statistics
            product_df['price_std'] = product_df['mid_price'].rolling(window=20).std()
            product_df['price_mean'] = product_df['mid_price'].rolling(window=20).mean()
            
            self.data[product] = product_df
            if product not in self.products:
                self.products.append(product)
                
    def plot_technical_analysis(self, product: str, save_path: Optional[str] = None) -> None:
        """Generate technical analysis plots for a specific product"""
        if product not in self.data:
            print(f"Warning: No data found for {product}")
            return
            
        df = self.data[product]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])
        
        # Plot 1: Price and Moving Averages
        ax1.plot(df['timestamp'], df['mid_price'], label='Mid Price', alpha=0.7)
        if 'short_ema' in df.columns:
            ax1.plot(df['timestamp'], df['short_ema'], label='Short EMA', alpha=0.6)
            ax1.plot(df['timestamp'], df['long_ema'], label='Long EMA', alpha=0.6)
        ax1.plot(df['timestamp'], df['price_mean'], label='20-period MA', alpha=0.6)
        ax1.fill_between(df['timestamp'], 
                        df['price_mean'] - 2*df['price_std'],
                        df['price_mean'] + 2*df['price_std'],
                        alpha=0.2, label='±2σ Band')
        ax1.set_title(f'{product} Price and Moving Averages')
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
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_market_depth(self, product: str, timestamp: Optional[datetime] = None, save_path: Optional[str] = None) -> None:
        """Generate market depth visualization for a specific product and timestamp"""
        if product not in self.data:
            print(f"Warning: No data found for {product}")
            return
            
        df = self.data[product]
        
        # If timestamp not provided, use the most recent one
        if timestamp is None:
            timestamp = df['timestamp'].max()
            
        # Get data for the specific timestamp
        snapshot = df[df['timestamp'] == timestamp].iloc[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bid side
        bid_prices = [snapshot[f'bid_price_{i}'] for i in range(1, 4) if pd.notna(snapshot[f'bid_price_{i}'])]
        bid_volumes = [snapshot[f'bid_volume_{i}'] for i in range(1, 4) if pd.notna(snapshot[f'bid_volume_{i}'])]
        ax.barh(bid_prices, bid_volumes, color='g', alpha=0.6, label='Bids')
        
        # Plot ask side
        ask_prices = [snapshot[f'ask_price_{i}'] for i in range(1, 4) if pd.notna(snapshot[f'ask_price_{i}'])]
        ask_volumes = [snapshot[f'ask_volume_{i}'] for i in range(1, 4) if pd.notna(snapshot[f'ask_volume_{i}'])]
        ax.barh(ask_prices, [-v for v in ask_volumes], color='r', alpha=0.6, label='Asks')
        
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'{product} Market Depth at {timestamp}')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Price')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_profitability_analysis(self, product: str, save_path: Optional[str] = None) -> None:
        """Generate profitability analysis plots"""
        if product not in self.data:
            print(f"Warning: No data found for {product}")
            return
            
        df = self.data[product]
        
        if 'profit_and_loss' not in df.columns:
            print(f"Warning: No P&L data found for {product}")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Cumulative P&L
        df['cumulative_pnl'] = df['profit_and_loss'].cumsum()
        ax1.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative P&L')
        ax1.set_title(f'{product} Cumulative Profit and Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: P&L Distribution
        sns.histplot(data=df, x='profit_and_loss', bins=50, ax=ax2)
        ax2.axvline(df['profit_and_loss'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {df["profit_and_loss"].mean():.2f}')
        ax2.set_title('P&L Distribution')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_summary_statistics(self, product: str) -> Dict:
        """Generate summary statistics for a product"""
        if product not in self.data:
            print(f"Warning: No data found for {product}")
            return {}
            
        df = self.data[product]
        stats = {
            'Mean Price': df['mid_price'].mean(),
            'Price Std Dev': df['mid_price'].std(),
            'Price Range': (df['mid_price'].min(), df['mid_price'].max()),
            'Average RSI': df['rsi'].mean(),
            'Total Trades': len(df),
            'Average Trade Size': df['mid_price'].diff().abs().mean()
        }
        
        if 'profit_and_loss' in df.columns:
            stats.update({
                'Total P&L': df['profit_and_loss'].sum(),
                'Average P&L per Trade': df['profit_and_loss'].mean(),
                'P&L Std Dev': df['profit_and_loss'].std(),
                'Win Rate': (df['profit_and_loss'] > 0).mean() * 100
            })
            
        return stats
        
    def analyze_all_products(self, output_dir: str = 'analysis_output') -> None:
        """Generate comprehensive analysis for all products"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for product in self.products:
            print(f"\nAnalyzing {product}...")
            
            # Generate technical analysis
            self.plot_technical_analysis(
                product, 
                os.path.join(output_dir, f'{product.lower()}_technical_analysis.png')
            )
            
            # Generate market depth analysis
            self.plot_market_depth(
                product,
                save_path=os.path.join(output_dir, f'{product.lower()}_market_depth.png')
            )
            
            # Generate profitability analysis if P&L data available
            if 'profit_and_loss' in self.data[product].columns:
                self.plot_profitability_analysis(
                    product,
                    os.path.join(output_dir, f'{product.lower()}_profitability.png')
                )
                
            # Print summary statistics
            stats = self.generate_summary_statistics(product)
            print(f"\n{product} Summary Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")

def main():
    analyzer = MarketAnalyzer()
    
    # Load all available data
    analyzer.load_squid_ink_data()
    analyzer.load_nopen_data()
    
    # Generate comprehensive analysis
    analyzer.analyze_all_products()

if __name__ == "__main__":
    main() 