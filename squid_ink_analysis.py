import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

class SquidInkAnalyzer:
    def __init__(self, data_path='squid_ink_ml_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.pca = None
    
    def load_data(self):
        """Load and perform initial processing of data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            print(f"Columns: {self.df.columns.tolist()}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
        
        # Display basic info
        print("\n===== Basic Information =====")
        print(self.df.info())
        print("\n===== Summary Statistics =====")
        print(self.df.describe())
        
        # Check for missing values
        print("\n===== Missing Values =====")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        # Create EDA plots
        self._create_eda_plots()
    
    def _create_eda_plots(self):
        """Create exploratory data analysis plots"""
        # 1. Distribution of key variables
        plt.figure(figsize=(15, 10))
        key_features = ['mid_price', 'vwap', 'current_ema', 'order_book_imbalance', 
                       'liquidity_score', 'momentum', 'recent_pl']
        
        for i, feature in enumerate(key_features):
            plt.subplot(3, 3, i+1)
            sns.histplot(self.df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
        
        plt.tight_layout()
        plt.savefig('squid_ink_distributions.png')
        
        # 2. Time series of key metrics
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(self.df['timestamp'], self.df['mid_price'], label='Mid Price')
        plt.plot(self.df['timestamp'], self.df['vwap'], label='VWAP')
        plt.legend()
        plt.title('Price Metrics Over Time')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.df['timestamp'], self.df['short_ema'], label='Short EMA')
        plt.plot(self.df['timestamp'], self.df['long_ema'], label='Long EMA')
        plt.legend()
        plt.title('Technical Indicators Over Time')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.df['timestamp'], self.df['recent_pl'], label='Recent P&L')
        plt.legend()
        plt.title('Performance Over Time')
        
        plt.tight_layout()
        plt.savefig('squid_ink_timeseries.png')
        
        # 3. Correlation matrix
        plt.figure(figsize=(12, 10))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig('squid_ink_correlation.png')
        
        print("\nEDA plots created and saved as PNG files.")
    
    def process_features(self):
        """Process features for model training"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return
        
        # Clean data - handle all NaN values first
        self.df = self.df.dropna()
        print(f"After initial NaN removal: {self.df.shape[0]} rows")
        
        # Engineer features
        self._engineer_features()
        
        # Define features and target
        features = ['best_bid', 'best_ask', 'mid_price', 'vwap', 'current_ema', 
                    'short_ema', 'long_ema', 'trend_signal', 'dynamic_width',
                    'position', 'order_book_imbalance', 'liquidity_score', 
                    'momentum', 'bid_ask_spread', 'ema_diff', 'vwap_mid_ratio']
        
        # Create different target variables for different prediction tasks
        targets = {
            'price_direction': self._calculate_price_direction(),
            'next_mid_price': self.df['mid_price'].shift(-1),
            'optimal_position': self._calculate_optimal_position(),
            'pl_prediction': self.df['recent_pl'].shift(-1)
        }
        
        # Drop NaN values created by shifts
        self.df = self.df.dropna()
        print(f"After feature engineering NaN removal: {self.df.shape[0]} rows")
        
        # Create multiple feature-target datasets
        self.datasets = {}
        for target_name, target_values in targets.items():
            # Make sure target_values is the right length and has no NaNs
            target_values = np.array(target_values)
            if len(target_values) > len(self.df):
                target_values = target_values[:len(self.df)]
            elif len(target_values) < len(self.df):
                self.df = self.df.iloc[:len(target_values)]
            
            # Check and handle NaNs
            X = self.df[features].copy()
            y = target_values[:len(X)].copy()
            
            # Replace any remaining NaNs with 0 in both X and y
            X = X.fillna(0)
            if np.isnan(y).any():
                print(f"Warning: NaN values found in target {target_name}. Replacing with 0.")
                y = np.nan_to_num(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            self.datasets[target_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': features
            }
            
        print(f"Features processed for {len(targets)} prediction tasks")
    
    def _engineer_features(self):
        """Create additional features from existing data"""
        # Bid-ask spread
        self.df['bid_ask_spread'] = self.df['best_ask'] - self.df['best_bid']
        
        # EMA differences
        self.df['ema_diff'] = self.df['short_ema'] - self.df['long_ema']
        
        # VWAP to mid price ratio
        self.df['vwap_mid_ratio'] = self.df['vwap'] / self.df['mid_price']
        self.df['vwap_mid_ratio'] = self.df['vwap_mid_ratio'].replace([np.inf, -np.inf], 1.0)
        
        # Lagged features
        for lag in [1, 3, 5]:
            self.df[f'mid_price_lag_{lag}'] = self.df['mid_price'].shift(lag)
            self.df[f'momentum_lag_{lag}'] = self.df['momentum'].shift(lag)
        
        # Moving averages
        for window in [3, 5, 10]:
            self.df[f'mid_price_ma_{window}'] = self.df['mid_price'].rolling(window=window).mean()
            self.df[f'liquidity_ma_{window}'] = self.df['liquidity_score'].rolling(window=window).mean()
        
        # Volatility measure (using rolling standard deviation)
        self.df['price_volatility'] = self.df['mid_price'].rolling(window=10).std()
        
        # Drop NaN values created by lags and rolling operations
        self.df = self.df.dropna()
        print(f"After feature engineering: {self.df.shape[0]} rows")
    
    def _calculate_price_direction(self):
        """Calculate price direction (1: up, 0: same, -1: down)"""
        next_price = self.df['mid_price'].shift(-1)
        current_price = self.df['mid_price']
        direction = np.sign(next_price - current_price)
        return direction
    
    def _calculate_optimal_position(self):
        """
        Estimate optimal position based on historical data
        Simple heuristic: positive position when price is going up, negative when down
        """
        future_return = self.df['mid_price'].shift(-5) / self.df['mid_price'] - 1
        # Scale to position limit (-50 to 50)
        scaled_position = 50 * np.sign(future_return) * np.minimum(np.abs(future_return) * 10, 1)
        return scaled_position
    
    def train_models(self, target='pl_prediction'):
        """Train multiple models for the specified target"""
        if not hasattr(self, 'datasets') or target not in self.datasets:
            print(f"Dataset for {target} not found. Please run process_features() first.")
            return
        
        dataset = self.datasets[target]
        X_train, y_train = dataset['X_train'], dataset['y_train']
        
        print(f"\n===== Training models for {target} =====")
        
        # Linear model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['linear'] = lr
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        print(f"Trained {len(self.models)} models for {target}")
    
    def evaluate_models(self, target='pl_prediction'):
        """Evaluate trained models"""
        if not hasattr(self, 'datasets') or target not in self.datasets:
            print(f"Dataset for {target} not found. Please run process_features() first.")
            return
        
        if not self.models:
            print("No models trained. Please run train_models() first.")
            return
        
        dataset = self.datasets[target]
        X_test, y_test = dataset['X_test'], dataset['y_test']
        feature_names = dataset['feature_names']
        
        print(f"\n===== Model Evaluation for {target} =====")
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"\nModel: {name}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importances ({name})')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(f'squid_ink_{target}_{name}_feature_importance.png')
        
        # Find best model based on MSE
        self.best_model = min(results.items(), key=lambda x: x[1]['mse'])[0]
        print(f"\nBest model for {target}: {self.best_model}")
        
        # Plot predictions vs actual
        best_model = self.models[self.best_model]
        y_pred = best_model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted ({target}, {self.best_model})')
        plt.savefig(f'squid_ink_{target}_predictions.png')
        
        return results
    
    def perform_pca_analysis(self):
        """Perform PCA to identify important patterns"""
        if not hasattr(self, 'datasets'):
            print("No processed data found. Please run process_features() first.")
            return
        
        # Use the first dataset for PCA
        target = list(self.datasets.keys())[0]
        X = np.vstack([self.datasets[target]['X_train'], self.datasets[target]['X_test']])
        
        # Perform PCA
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.savefig('squid_ink_pca_variance.png')
        
        # Plot first two principal components
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA: First Two Principal Components')
        plt.savefig('squid_ink_pca_components.png')
        
        # Print component loadings
        feature_names = self.datasets[target]['feature_names']
        loadings = self.pca.components_
        
        print("\n===== PCA Component Loadings =====")
        for i, component in enumerate(loadings[:3]):  # First 3 components
            sorted_indices = np.argsort(np.abs(component))[::-1]
            print(f"\nComponent {i+1} (explains {self.pca.explained_variance_ratio_[i]:.2%} of variance)")
            for idx in sorted_indices[:5]:  # Top 5 features
                print(f"{feature_names[idx]}: {component[idx]:.4f}")
    
    def perform_cluster_analysis(self):
        """Perform clustering to identify market regimes"""
        if not hasattr(self, 'datasets'):
            print("No processed data found. Please run process_features() first.")
            return
        
        # Use the first dataset for clustering
        target = list(self.datasets.keys())[0]
        X = np.vstack([self.datasets[target]['X_train'], self.datasets[target]['X_test']])
        
        # Determine optimal number of clusters
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertia, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.savefig('squid_ink_elbow_curve.png')
        
        # Choose number of clusters from elbow (example: 4)
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # If PCA is available, plot clusters on PCA components
        if self.pca is not None:
            X_pca = self.pca.transform(X)
            
            plt.figure(figsize=(10, 6))
            for i in range(n_clusters):
                plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.legend()
            plt.title('Clusters in PCA Space')
            plt.savefig('squid_ink_clusters.png')
        
        # Analyze cluster characteristics
        feature_names = self.datasets[target]['feature_names']
        cluster_centers = kmeans.cluster_centers_
        
        print("\n===== Cluster Characteristics =====")
        for i in range(n_clusters):
            print(f"\nCluster {i} characteristics:")
            sorted_indices = np.argsort(np.abs(cluster_centers[i]))[::-1]
            for idx in sorted_indices[:5]:  # Top 5 features
                print(f"{feature_names[idx]}: {cluster_centers[i][idx]:.4f}")
    
    def identify_profitable_patterns(self):
        """Identify patterns that lead to profitable trades"""
        if not hasattr(self, 'datasets') or 'pl_prediction' not in self.datasets:
            print("P&L prediction dataset not found. Please run process_features() first.")
            return
        
        if not self.models or 'lightgbm' not in self.models:
            print("LightGBM model not found. Please run train_models() first.")
            return
        
        # Get feature importances from LightGBM model
        lgb_model = self.models['lightgbm']
        feature_names = self.datasets['pl_prediction']['feature_names']
        
        importances = lgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n===== Most Important Features for Profitability =====")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Analyze conditions for profitable trades
        X = np.vstack([
            self.datasets['pl_prediction']['X_train'], 
            self.datasets['pl_prediction']['X_test']
        ])
        y = np.concatenate([
            self.datasets['pl_prediction']['y_train'], 
            self.datasets['pl_prediction']['y_test']
        ])
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame(X, columns=feature_names)
        analysis_df['profit'] = y
        
        # Categorize trades as profitable or not
        analysis_df['is_profitable'] = analysis_df['profit'] > 0
        
        # Analyze profitable vs. unprofitable trades
        profitable = analysis_df[analysis_df['is_profitable']]
        unprofitable = analysis_df[~analysis_df['is_profitable']]
        
        print("\n===== Profitable vs. Unprofitable Trade Conditions =====")
        for feature in feature_names:
            prof_mean = profitable[feature].mean()
            unprof_mean = unprofitable[feature].mean()
            diff_pct = (prof_mean - unprof_mean) / (np.abs(unprof_mean) + 1e-10) * 100
            
            if abs(diff_pct) > 10:  # Only show features with >10% difference
                print(f"{feature}: {diff_pct:.2f}% difference")
                print(f"  Profitable mean: {prof_mean:.4f}")
                print(f"  Unprofitable mean: {unprof_mean:.4f}")
        
        # Plot distributions of key features for profitable vs unprofitable trades
        key_features = [feature_names[i] for i in indices[:6]]  # Top 6 features
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(key_features):
            plt.subplot(2, 3, i+1)
            sns.kdeplot(profitable[feature], label='Profitable')
            sns.kdeplot(unprofitable[feature], label='Unprofitable')
            plt.title(feature)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('squid_ink_profitable_patterns.png')
        
        # Create decision rules
        print("\n===== Recommended Trading Rules =====")
        for feature in key_features:
            if profitable[feature].mean() > unprofitable[feature].mean():
                threshold = (profitable[feature].quantile(0.25) + unprofitable[feature].quantile(0.75)) / 2
                print(f"When {feature} > {threshold:.4f}, consider taking position")
            else:
                threshold = (profitable[feature].quantile(0.75) + unprofitable[feature].quantile(0.25)) / 2
                print(f"When {feature} < {threshold:.4f}, consider taking position")

def main():
    analyzer = SquidInkAnalyzer()
    
    # Load and explore data
    if analyzer.load_data():
        analyzer.explore_data()
        
        # Process features and train models
        analyzer.process_features()
        
        # Train and evaluate models for different targets
        for target in ['price_direction', 'next_mid_price', 'optimal_position', 'pl_prediction']:
            analyzer.train_models(target)
            analyzer.evaluate_models(target)
        
        # Perform deeper analysis
        analyzer.perform_pca_analysis()
        analyzer.perform_cluster_analysis()
        analyzer.identify_profitable_patterns()
        
        print("\nAnalysis completed. Check the generated PNG files for visualizations.")
    else:
        print("Failed to load data. Make sure 'squid_ink_ml_data.csv' exists.")

if __name__ == "__main__":
    main() 