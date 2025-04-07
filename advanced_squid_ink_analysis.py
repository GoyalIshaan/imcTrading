import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class AdvancedSquidInkAnalyzer:
    def __init__(self, data_path='squid_ink_ml_data.csv'):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the SQUID_INK trading data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            print(f"Columns: {list(self.df.columns)}")
            
            # Convert timestamp to datetime
            self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
            self.df.set_index('datetime', inplace=True)
            
            # Create target variables
            self.df['next_mid_price'] = self.df['mid_price'].shift(-1)
            self.df['price_change'] = self.df['mid_price'].pct_change()
            self.df['next_price_change'] = self.df['price_change'].shift(-1)
            self.df['is_profitable'] = self.df['recent_pl'] > 0
            
            # Calculate additional features
            self.df['vwap_mid_ratio'] = self.df['vwap'] / self.df['mid_price']
            self.df['bid_ask_spread'] = self.df['best_ask'] - self.df['best_bid']
            self.df['ema_diff'] = self.df['short_ema'] - self.df['long_ema']
            
            # Drop NaN values
            self.df = self.df.dropna()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def time_series_analysis(self):
        """Perform detailed time series analysis"""
        print("\n===== Time Series Analysis =====")
        
        # Check for stationarity
        adf_result = adfuller(self.df['mid_price'])
        print(f"ADF Test p-value: {adf_result[1]:.4f}")
        print(f"Stationary: {adf_result[1] < 0.05}")
        
        # Autocorrelation analysis
        acf_values = acf(self.df['price_change'], nlags=20)
        pacf_values = pacf(self.df['price_change'], nlags=20)
        
        # Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.bar(range(len(acf_values)), acf_values)
        ax1.axhline(y=0, linestyle='-', color='gray')
        ax1.axhline(y=1.96/np.sqrt(len(self.df)), linestyle='--', color='gray')
        ax1.axhline(y=-1.96/np.sqrt(len(self.df)), linestyle='--', color='gray')
        ax1.set_title('Autocorrelation Function (ACF)')
        
        ax2.bar(range(len(pacf_values)), pacf_values)
        ax2.axhline(y=0, linestyle='-', color='gray')
        ax2.axhline(y=1.96/np.sqrt(len(self.df)), linestyle='--', color='gray')
        ax2.axhline(y=-1.96/np.sqrt(len(self.df)), linestyle='--', color='gray')
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.savefig('squid_ink_acf_pacf.png')
        
        # Seasonal decomposition
        try:
            # Resample to daily data for seasonal decomposition
            daily_data = self.df['mid_price'].resample('D').mean()
            decomposition = seasonal_decompose(daily_data, model='additive', period=7)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            
            plt.tight_layout()
            plt.savefig('squid_ink_seasonal_decomposition.png')
        except Exception as e:
            print(f"Seasonal decomposition error: {e}")
        
        # Store results
        self.results['time_series'] = {
            'is_stationary': adf_result[1] < 0.05,
            'adf_p_value': adf_result[1],
            'acf_values': acf_values,
            'pacf_values': pacf_values
        }
    
    def regime_analysis(self):
        """Analyze different market regimes and their characteristics"""
        print("\n===== Market Regime Analysis =====")
        
        # Select features for regime analysis
        features = ['mid_price', 'vwap', 'current_ema', 'short_ema', 'long_ema', 
                   'trend_signal', 'dynamic_width', 'order_book_imbalance', 
                   'liquidity_score', 'momentum']
        
        X = self.df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X_scaled)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        self.df['market_regime'] = clusters
        
        # Analyze each regime
        regime_stats = {}
        for i in range(4):
            regime_data = self.df[self.df['market_regime'] == i]
            
            # Calculate statistics
            stats = {
                'count': len(regime_data),
                'avg_price': regime_data['mid_price'].mean(),
                'avg_momentum': regime_data['momentum'].mean(),
                'avg_liquidity': regime_data['liquidity_score'].mean(),
                'avg_imbalance': regime_data['order_book_imbalance'].mean(),
                'profitability': regime_data['is_profitable'].mean(),
                'avg_pl': regime_data['recent_pl'].mean()
            }
            
            regime_stats[i] = stats
            print(f"\nRegime {i} characteristics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
        
        # Plot regimes
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
        ax.set_title('Market Regimes (PCA)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Regime')
        plt.savefig('squid_ink_regimes_pca.png')
        
        # Store results
        self.results['regime_analysis'] = {
            'regime_stats': regime_stats,
            'pca_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def conditional_probability_analysis(self):
        """Analyze conditional probabilities of profitable trades"""
        print("\n===== Conditional Probability Analysis =====")
        
        # Define thresholds for key indicators
        momentum_threshold = 0.0149
        liquidity_threshold = 0.0285
        imbalance_threshold = 0.0548
        
        # Calculate conditional probabilities
        conditions = {
            'low_momentum': self.df['momentum'] < momentum_threshold,
            'high_momentum': self.df['momentum'] >= momentum_threshold,
            'low_liquidity': self.df['liquidity_score'] < liquidity_threshold,
            'high_liquidity': self.df['liquidity_score'] >= liquidity_threshold,
            'high_imbalance': self.df['order_book_imbalance'] > imbalance_threshold,
            'low_imbalance': self.df['order_book_imbalance'] <= imbalance_threshold
        }
        
        # Single condition probabilities
        single_conditions = {}
        for name, condition in conditions.items():
            prob = self.df[condition]['is_profitable'].mean()
            count = self.df[condition]['is_profitable'].count()
            single_conditions[name] = {'probability': prob, 'count': count}
            print(f"{name}: {prob:.4f} ({count} samples)")
        
        # Combined condition probabilities
        combined_conditions = {
            'low_momentum_and_low_liquidity': (self.df['momentum'] < momentum_threshold) & 
                                             (self.df['liquidity_score'] < liquidity_threshold),
            'low_momentum_and_high_imbalance': (self.df['momentum'] < momentum_threshold) & 
                                              (self.df['order_book_imbalance'] > imbalance_threshold),
            'low_liquidity_and_high_imbalance': (self.df['liquidity_score'] < liquidity_threshold) & 
                                               (self.df['order_book_imbalance'] > imbalance_threshold),
            'all_three': (self.df['momentum'] < momentum_threshold) & 
                        (self.df['liquidity_score'] < liquidity_threshold) & 
                        (self.df['order_book_imbalance'] > imbalance_threshold)
        }
        
        combined_results = {}
        for name, condition in combined_conditions.items():
            prob = self.df[condition]['is_profitable'].mean()
            count = self.df[condition]['is_profitable'].count()
            combined_results[name] = {'probability': prob, 'count': count}
            print(f"{name}: {prob:.4f} ({count} samples)")
        
        # Store results
        self.results['conditional_probability'] = {
            'single_conditions': single_conditions,
            'combined_conditions': combined_results
        }
    
    def lead_lag_analysis(self):
        """Analyze lead-lag relationships between indicators and price"""
        print("\n===== Lead-Lag Analysis =====")
        
        # Define indicators to analyze
        indicators = ['momentum', 'liquidity_score', 'order_book_imbalance', 
                     'trend_signal', 'vwap_mid_ratio']
        
        # Calculate cross-correlation for different lags
        max_lag = 10
        lead_lag_results = {}
        
        for indicator in indicators:
            correlations = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Indicator leads price
                    corr = self.df[indicator].shift(-lag).corr(self.df['price_change'])
                else:
                    # Price leads indicator
                    corr = self.df[indicator].shift(lag).corr(self.df['price_change'])
                correlations.append(corr)
            
            lead_lag_results[indicator] = correlations
            
            # Find best lead/lag
            best_corr = max(correlations)
            best_lag = correlations.index(best_corr) - max_lag
            print(f"{indicator}: Best correlation {best_corr:.4f} at lag {best_lag}")
        
        # Plot lead-lag relationships
        plt.figure(figsize=(12, 8))
        for indicator, correlations in lead_lag_results.items():
            plt.plot(range(-max_lag, max_lag + 1), correlations, label=indicator)
        
        plt.axhline(y=0, color='k', linestyle='-')
        plt.axvline(x=0, color='k', linestyle='-')
        plt.title('Lead-Lag Analysis')
        plt.xlabel('Lag (negative = indicator leads price)')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig('squid_ink_lead_lag.png')
        
        # Store results
        self.results['lead_lag_analysis'] = lead_lag_results
    
    def profitability_by_time(self):
        """Analyze profitability patterns by time of day"""
        print("\n===== Time-Based Profitability Analysis =====")
        
        # Extract hour from datetime index
        self.df['hour'] = self.df.index.hour
        
        # Calculate profitability by hour
        hourly_profitability = self.df.groupby('hour')['is_profitable'].mean()
        hourly_pl = self.df.groupby('hour')['recent_pl'].mean()
        
        # Plot profitability by hour
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        hourly_profitability.plot(kind='bar', ax=ax1)
        ax1.set_title('Profitability by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Probability of Profitable Trade')
        ax1.grid(True)
        
        hourly_pl.plot(kind='bar', ax=ax2)
        ax2.set_title('Average P&L by Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average P&L')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('squid_ink_hourly_profitability.png')
        
        # Find best and worst hours
        best_hour = hourly_profitability.idxmax()
        worst_hour = hourly_profitability.idxmin()
        best_hour_pl = hourly_pl.idxmax()
        worst_hour_pl = hourly_pl.idxmin()
        
        print(f"Best hour for profitability: {best_hour} ({hourly_profitability[best_hour]:.4f})")
        print(f"Worst hour for profitability: {worst_hour} ({hourly_profitability[worst_hour]:.4f})")
        print(f"Best hour for P&L: {best_hour_pl} ({hourly_pl[best_hour_pl]:.4f})")
        print(f"Worst hour for P&L: {worst_hour_pl} ({hourly_pl[worst_hour_pl]:.4f})")
        
        # Store results
        self.results['time_based_analysis'] = {
            'hourly_profitability': hourly_profitability.to_dict(),
            'hourly_pl': hourly_pl.to_dict(),
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'best_hour_pl': best_hour_pl,
            'worst_hour_pl': worst_hour_pl
        }
    
    def run_advanced_analysis(self):
        """Run all advanced analysis methods"""
        if self.load_data():
            self.time_series_analysis()
            self.regime_analysis()
            self.conditional_probability_analysis()
            self.lead_lag_analysis()
            self.profitability_by_time()
            
            print("\n===== Advanced Analysis Summary =====")
            print("1. Time Series Analysis:")
            print(f"   - Data is {'stationary' if self.results['time_series']['is_stationary'] else 'non-stationary'}")
            
            print("\n2. Market Regimes:")
            for regime, stats in self.results['regime_analysis']['regime_stats'].items():
                print(f"   - Regime {regime}: {stats['count']} samples, {stats['profitability']:.2%} profitable")
            
            print("\n3. Conditional Probabilities:")
            best_single = max(self.results['conditional_probability']['single_conditions'].items(), 
                             key=lambda x: x[1]['probability'])
            best_combined = max(self.results['conditional_probability']['combined_conditions'].items(), 
                               key=lambda x: x[1]['probability'])
            print(f"   - Best single condition: {best_single[0]} ({best_single[1]['probability']:.2%})")
            print(f"   - Best combined condition: {best_combined[0]} ({best_combined[1]['probability']:.2%})")
            
            print("\n4. Time-Based Analysis:")
            print(f"   - Best trading hour: {self.results['time_based_analysis']['best_hour']} "
                  f"({self.results['time_based_analysis']['hourly_profitability'][self.results['time_based_analysis']['best_hour']]:.2%})")
            
            return True
        return False

if __name__ == "__main__":
    analyzer = AdvancedSquidInkAnalyzer()
    analyzer.run_advanced_analysis() 