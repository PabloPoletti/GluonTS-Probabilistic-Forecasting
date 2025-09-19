"""
🎯 GluonTS Probabilistic Forecasting Analysis
Complete Probabilistic Time Series Analysis with Uncertainty Quantification

This analysis demonstrates:
1. Probabilistic forecasting with confidence intervals
2. Deep learning models for complex patterns
3. Uncertainty quantification and risk assessment
4. Multi-step ahead forecasting
5. Model comparison with probabilistic metrics
6. Real-world applications with missing data handling

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Iterator
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# GluonTS imports
try:
    from gluonts.dataset.common import ListDataset, DataEntry
    from gluonts.dataset.field_names import FieldName
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.model.transformer import TransformerEstimator
    from gluonts.model.prophet import ProphetPredictor
    from gluonts.model.seasonal_naive import SeasonalNaivePredictor
    from gluonts.model.trivial.mean import MeanPredictor
    from gluonts.trainer import Trainer
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.transform import (
        AddTimeFeatures, AddAgeFeature, AddObservedValuesIndicator,
        Chain, RemoveFields, SetField, AsNumpyArray, ExpandDimArray,
        VstackFeatures, InstanceSplitter, ExpectedNumInstanceSampler
    )
    from gluonts.time_feature import (
        time_features_from_frequency_str, DayOfWeek, DayOfMonth, 
        DayOfYear, MonthOfYear, WeekOfYear
    )
    import gluonts
except ImportError as e:
    print(f"Warning: GluonTS not installed: {e}")
    print("Install with: pip install gluonts[mxnet,pro]")

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GluonTSProbabilisticAnalysis:
    """Complete GluonTS Probabilistic Analysis Pipeline"""
    
    def __init__(self):
        self.datasets = {}
        self.gluonts_datasets = {}
        self.models = {}
        self.predictors = {}
        self.forecasts = {}
        self.metrics = {}
        self.probabilistic_metrics = {}
        self.best_params = {}
        
    def load_probabilistic_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load datasets suitable for probabilistic forecasting"""
        print("📊 Loading probabilistic forecasting datasets...")
        
        datasets = {}
        
        # 1. Volatile cryptocurrency data
        print("Loading volatile crypto data (BTC)...")
        try:
            btc = yf.download("BTC-USD", period="2y", interval="1d")
            btc_df = pd.DataFrame({
                'timestamp': btc.index,
                'target': btc['Close'].values,
                'volume': btc['Volume'].values,
                'high': btc['High'].values,
                'low': btc['Low'].values
            }).reset_index(drop=True)
            datasets['BTC_Volatile'] = btc_df
        except Exception as e:
            print(f"Failed to load BTC data: {e}")
        
        # 2. Synthetic demand data with uncertainty
        print("Generating demand data with uncertainty...")
        demand_dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        
        # Base demand with increasing uncertainty over time
        base_demand = 1000
        trend = np.cumsum(np.random.normal(1, 2, len(demand_dates)))
        seasonal = 200 * np.sin(2 * np.pi * np.arange(len(demand_dates)) / 365.25)
        weekly = 100 * np.sin(2 * np.pi * np.arange(len(demand_dates)) / 7)
        
        # Increasing volatility over time (heteroscedasticity)
        volatility = 50 + 0.1 * np.arange(len(demand_dates))
        noise = np.random.normal(0, volatility)
        
        # Random demand spikes (uncertainty events)
        spike_probability = 0.05
        spikes = np.random.binomial(1, spike_probability, len(demand_dates)) * np.random.exponential(500, len(demand_dates))
        
        demand = np.maximum(base_demand + trend + seasonal + weekly + noise + spikes, 100)
        
        demand_df = pd.DataFrame({
            'timestamp': demand_dates,
            'target': demand,
            'volatility': volatility,
            'is_spike': (spikes > 0).astype(int)
        })
        datasets['Demand_Uncertain'] = demand_df
        
        # 3. Financial portfolio returns (multiple assets)
        print("Generating portfolio returns with correlations...")
        portfolio_dates = pd.date_range('2019-01-01', '2024-01-01', freq='B')  # Business days
        
        # Simulate correlated returns for 3 assets
        n_assets = 3
        correlation_matrix = np.array([[1.0, 0.6, 0.3],
                                     [0.6, 1.0, 0.4],
                                     [0.3, 0.4, 1.0]])
        
        # Generate correlated random returns
        random_returns = np.random.multivariate_normal(
            mean=[0.0005, 0.0003, 0.0007],  # Different expected returns
            cov=correlation_matrix * 0.02**2,  # 2% daily volatility
            size=len(portfolio_dates)
        )
        
        # Convert to price levels
        initial_prices = [100, 150, 80]
        prices = np.zeros((len(portfolio_dates), n_assets))
        prices[0] = initial_prices
        
        for i in range(1, len(portfolio_dates)):
            prices[i] = prices[i-1] * (1 + random_returns[i])
        
        # Create portfolio value (weighted sum)
        weights = [0.5, 0.3, 0.2]
        portfolio_value = np.sum(prices * weights, axis=1)
        
        portfolio_df = pd.DataFrame({
            'timestamp': portfolio_dates,
            'target': portfolio_value,
            'asset1': prices[:, 0],
            'asset2': prices[:, 1],
            'asset3': prices[:, 2]
        })
        datasets['Portfolio_Returns'] = portfolio_df
        
        # 4. Energy consumption with weather dependency
        print("Generating energy consumption with weather uncertainty...")
        energy_dates = pd.date_range('2021-01-01', '2024-01-01', freq='H')
        
        # Base consumption patterns
        base_consumption = 5000
        
        # Hourly patterns
        hour_of_day = np.tile(np.arange(24), len(energy_dates) // 24 + 1)[:len(energy_dates)]
        hourly_pattern = 1000 * np.sin(2 * np.pi * hour_of_day / 24) + 500 * np.sin(4 * np.pi * hour_of_day / 24)
        
        # Seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in energy_dates])
        seasonal_pattern = 1500 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Weather-dependent uncertainty
        temperature = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 5, len(energy_dates))
        weather_impact = np.where(
            np.abs(temperature - 20) > 10,
            500 * np.abs(temperature - 20) / 10,  # Higher consumption for extreme temps
            0
        )
        
        # Random outages (missing data simulation)
        outage_probability = 0.001
        outages = np.random.binomial(1, outage_probability, len(energy_dates))
        
        energy_noise = np.random.normal(0, 200, len(energy_dates))
        energy_consumption = base_consumption + hourly_pattern + seasonal_pattern + weather_impact + energy_noise
        energy_consumption = np.maximum(energy_consumption, 1000)
        
        # Introduce missing values for outages
        energy_consumption[outages == 1] = np.nan
        
        # Sample every 6 hours for performance
        sample_indices = np.arange(0, len(energy_consumption), 6)
        
        energy_df = pd.DataFrame({
            'timestamp': energy_dates[sample_indices],
            'target': energy_consumption[sample_indices],
            'temperature': temperature[sample_indices],
            'hour_of_day': hour_of_day[sample_indices]
        })
        datasets['Energy_Weather'] = energy_df
        
        self.datasets = datasets
        print(f"✅ Loaded {len(datasets)} probabilistic datasets")
        return datasets
    
    def convert_to_gluonts_format(self):
        """Convert datasets to GluonTS format"""
        print("\n🔄 Converting datasets to GluonTS format...")
        
        for name, df in self.datasets.items():
            try:
                # Determine frequency
                if name == 'Energy_Weather':
                    freq = '6H'
                elif name == 'Portfolio_Returns':
                    freq = 'B'
                else:
                    freq = 'D'
                
                # Create GluonTS dataset
                gluonts_data = []
                
                # Handle missing values
                target_values = df['target'].values
                if np.any(np.isnan(target_values)):
                    # Forward fill missing values
                    target_series = pd.Series(target_values).fillna(method='ffill').fillna(method='bfill')
                    target_values = target_series.values
                
                # Create data entry
                data_entry = {
                    FieldName.TARGET: target_values,
                    FieldName.START: pd.Timestamp(df['timestamp'].iloc[0]),
                    FieldName.ITEM_ID: name
                }
                
                # Add dynamic features if available
                dynamic_features = []
                for col in df.columns:
                    if col not in ['timestamp', 'target'] and not df[col].isna().all():
                        feature_values = df[col].fillna(method='ffill').fillna(method='bfill').values
                        dynamic_features.append(feature_values)
                
                if dynamic_features:
                    data_entry[FieldName.FEAT_DYNAMIC_REAL] = np.array(dynamic_features)
                
                gluonts_data.append(data_entry)
                
                # Create ListDataset
                dataset = ListDataset(gluonts_data, freq=freq)
                self.gluonts_datasets[name] = dataset
                
                print(f"  ✅ {name}: {len(target_values)} points, freq={freq}")
                
            except Exception as e:
                print(f"  ❌ Failed to convert {name}: {e}")
                continue
        
        print(f"✅ Converted {len(self.gluonts_datasets)} datasets to GluonTS format")
    
    def comprehensive_probabilistic_eda(self):
        """Probabilistic-focused EDA with uncertainty analysis"""
        print("\n📈 Performing Probabilistic EDA...")
        
        fig = make_subplots(
            rows=len(self.datasets), cols=4,
            subplot_titles=[f"{name} - Time Series" for name in self.datasets.keys()] +
                          [f"{name} - Volatility" for name in self.datasets.keys()] +
                          [f"{name} - Distribution" for name in self.datasets.keys()] +
                          [f"{name} - Uncertainty" for name in self.datasets.keys()],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, 
                   {"secondary_y": False}, {"secondary_y": False}] 
                   for _ in range(len(self.datasets))]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, df) in enumerate(self.datasets.items()):
            row = i + 1
            
            # 1. Time series plot
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['target'],
                    mode='lines',
                    name=f'{name}',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=row, col=1
            )
            
            # 2. Rolling volatility
            rolling_std = pd.Series(df['target']).rolling(30).std()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=rolling_std,
                    mode='lines',
                    name=f'{name} Volatility',
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ),
                row=row, col=2
            )
            
            # 3. Distribution
            fig.add_trace(
                go.Histogram(
                    x=df['target'].dropna(),
                    name=f'{name} Distribution',
                    nbinsx=30,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7
                ),
                row=row, col=3
            )
            
            # 4. Uncertainty metrics (returns or differences)
            returns = df['target'].pct_change().dropna()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'][1:],
                    y=returns,
                    mode='markers',
                    name=f'{name} Returns',
                    marker=dict(color=colors[i % len(colors)], size=3, opacity=0.6)
                ),
                row=row, col=4
            )
        
        fig.update_layout(
            height=300 * len(self.datasets),
            title_text="🎯 Probabilistic Forecasting EDA",
            showlegend=True
        )
        
        fig.write_html("gluonts_probabilistic_eda.html")
        print("✅ Probabilistic EDA completed. Dashboard saved as 'gluonts_probabilistic_eda.html'")
        
        # Uncertainty analysis
        print("\n📊 Uncertainty Analysis:")
        for name, df in self.datasets.items():
            target = df['target'].dropna()
            returns = target.pct_change().dropna()
            
            print(f"\n{name}:")
            print(f"  Mean: {target.mean():.2f}")
            print(f"  Volatility (std): {target.std():.2f}")
            print(f"  Coefficient of Variation: {(target.std() / target.mean()) * 100:.1f}%")
            print(f"  Return Volatility: {returns.std() * 100:.2f}%")
            print(f"  Skewness: {returns.skew():.2f}")
            print(f"  Kurtosis: {returns.kurtosis():.2f}")
            
            # VaR estimation (5% and 1%)
            var_5 = np.percentile(returns, 5)
            var_1 = np.percentile(returns, 1)
            print(f"  VaR (5%): {var_5 * 100:.2f}%")
            print(f"  VaR (1%): {var_1 * 100:.2f}%")
    
    def create_probabilistic_models(self, prediction_length: int = 30) -> Dict[str, object]:
        """Create GluonTS probabilistic models"""
        print(f"\n🧠 Creating probabilistic models (horizon: {prediction_length})...")
        
        # Trainer configuration
        trainer = Trainer(
            epochs=20,
            learning_rate=1e-3,
            batch_size=32,
            num_batches_per_epoch=50
        )
        
        models = {
            # Deep Learning Models
            'DeepAR': DeepAREstimator(
                freq='D',
                prediction_length=prediction_length,
                trainer=trainer,
                use_feat_dynamic_real=True,
                use_feat_static_cat=False,
                cardinality=[],
                num_layers=2,
                num_cells=40,
                cell_type='lstm',
                dropout_rate=0.1,
                use_symbol_block_predictor=False
            ),
            
            'SimpleFeedForward': SimpleFeedForwardEstimator(
                freq='D',
                prediction_length=prediction_length,
                trainer=trainer,
                num_hidden_dimensions=[40, 40],
                context_length=prediction_length * 2
            ),
            
            'Transformer': TransformerEstimator(
                freq='D',
                prediction_length=prediction_length,
                trainer=trainer,
                context_length=prediction_length * 2,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_model=32
            ),
            
            # Statistical Models
            'SeasonalNaive': SeasonalNaivePredictor(
                freq='D',
                prediction_length=prediction_length,
                season_length=7
            ),
            
            'MeanPredictor': MeanPredictor(
                freq='D',
                prediction_length=prediction_length
            )
        }
        
        # Add Prophet if available
        try:
            models['Prophet'] = ProphetPredictor(
                freq='D',
                prediction_length=prediction_length
            )
        except:
            print("  ⚠️ Prophet not available in GluonTS")
        
        self.models = models
        print(f"✅ Created {len(models)} probabilistic models")
        return models
    
    def train_and_evaluate_probabilistic_models(self, dataset_name: str, prediction_length: int = 30):
        """Train and evaluate probabilistic models"""
        print(f"\n🚀 Training probabilistic models on {dataset_name}...")
        
        if dataset_name not in self.gluonts_datasets:
            print(f"❌ Dataset {dataset_name} not available in GluonTS format")
            return
        
        dataset = self.gluonts_datasets[dataset_name]
        
        # Create models with appropriate frequency
        freq = dataset.metadata.freq
        models = {}
        
        # Update model frequencies
        for name, model_template in self.models.items():
            try:
                if hasattr(model_template, 'freq'):
                    # Create new model with correct frequency
                    model_params = model_template.__dict__.copy()
                    model_params['freq'] = freq
                    model_params['prediction_length'] = prediction_length
                    
                    if name == 'DeepAR':
                        models[name] = DeepAREstimator(**model_params)
                    elif name == 'SimpleFeedForward':
                        models[name] = SimpleFeedForwardEstimator(**model_params)
                    elif name == 'Transformer':
                        models[name] = TransformerEstimator(**model_params)
                    elif name == 'SeasonalNaive':
                        models[name] = SeasonalNaivePredictor(
                            freq=freq,
                            prediction_length=prediction_length,
                            season_length=7 if freq == 'D' else 24
                        )
                    elif name == 'MeanPredictor':
                        models[name] = MeanPredictor(
                            freq=freq,
                            prediction_length=prediction_length
                        )
                    else:
                        models[name] = model_template
                else:
                    models[name] = model_template
                    
            except Exception as e:
                print(f"  ⚠️ Failed to create {name}: {e}")
                continue
        
        # Train models and generate forecasts
        predictors = {}
        forecasts = {}
        
        for name, model in models.items():
            try:
                print(f"  Training {name}...")
                
                # Train model
                if hasattr(model, 'train'):
                    predictor = model.train(dataset)
                else:
                    predictor = model
                
                predictors[name] = predictor
                
                # Generate forecasts
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=predictor,
                    num_samples=100
                )
                
                forecasts[name] = {
                    'forecasts': list(forecast_it),
                    'timeseries': list(ts_it)
                }
                
                print(f"    ✅ {name} completed")
                
            except Exception as e:
                print(f"    ❌ {name} failed: {str(e)}")
                continue
        
        self.predictors[dataset_name] = predictors
        self.forecasts[dataset_name] = forecasts
        
        # Evaluate models
        self.evaluate_probabilistic_models(dataset_name)
        
        print(f"✅ Completed probabilistic training on {dataset_name}")
    
    def evaluate_probabilistic_models(self, dataset_name: str):
        """Evaluate models with probabilistic metrics"""
        print(f"\n📊 Evaluating probabilistic models for {dataset_name}...")
        
        if dataset_name not in self.forecasts:
            print("❌ No forecasts available for evaluation")
            return
        
        forecasts_dict = self.forecasts[dataset_name]
        
        # Initialize evaluator with probabilistic metrics
        evaluator = Evaluator(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        results = []
        probabilistic_results = []
        
        for name, forecast_data in forecasts_dict.items():
            try:
                forecasts = forecast_data['forecasts']
                timeseries = forecast_data['timeseries']
                
                # Evaluate
                agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts))
                
                # Standard metrics
                results.append({
                    'Model': name,
                    'MASE': agg_metrics.get('MASE', np.nan),
                    'sMAPE': agg_metrics.get('sMAPE', np.nan),
                    'MSIS': agg_metrics.get('MSIS', np.nan),  # Mean Scaled Interval Score
                    'mean_wQuantileLoss': agg_metrics.get('mean_wQuantileLoss', np.nan)
                })
                
                # Probabilistic metrics
                prob_metrics = {
                    'Model': name,
                    'Coverage_50': agg_metrics.get('coverage[0.5]', np.nan),
                    'Coverage_80': agg_metrics.get('coverage[0.8]', np.nan),
                    'Coverage_90': agg_metrics.get('coverage[0.9]', np.nan),
                    'QuantileLoss_10': agg_metrics.get('QuantileLoss[0.1]', np.nan),
                    'QuantileLoss_50': agg_metrics.get('QuantileLoss[0.5]', np.nan),
                    'QuantileLoss_90': agg_metrics.get('QuantileLoss[0.9]', np.nan)
                }
                probabilistic_results.append(prob_metrics)
                
            except Exception as e:
                print(f"  ❌ Evaluation failed for {name}: {e}")
                continue
        
        # Store results
        if results:
            self.metrics[dataset_name] = pd.DataFrame(results).sort_values('MASE')
            self.probabilistic_metrics[dataset_name] = pd.DataFrame(probabilistic_results)
            
            print(f"✅ Evaluation completed for {dataset_name}")
            print(f"🏆 Best model (MASE): {self.metrics[dataset_name].iloc[0]['Model']}")
        else:
            print("❌ No successful evaluations")
    
    def optimize_probabilistic_hyperparameters(self, dataset_name: str, model_type: str = 'DeepAR'):
        """Optimize hyperparameters for probabilistic models"""
        print(f"\n⚙️ Optimizing {model_type} hyperparameters for {dataset_name}...")
        
        if dataset_name not in self.gluonts_datasets:
            print("❌ Dataset not available")
            return
        
        dataset = self.gluonts_datasets[dataset_name]
        freq = dataset.metadata.freq
        
        def objective(trial):
            try:
                if model_type == 'DeepAR':
                    params = {
                        'freq': freq,
                        'prediction_length': 30,
                        'num_layers': trial.suggest_int('num_layers', 1, 3),
                        'num_cells': trial.suggest_int('num_cells', 20, 80),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                        'trainer': Trainer(
                            epochs=10,  # Reduced for optimization
                            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                            batch_size=trial.suggest_int('batch_size', 16, 64)
                        )
                    }
                    model = DeepAREstimator(**params)
                    
                elif model_type == 'SimpleFeedForward':
                    hidden_dim = trial.suggest_int('hidden_dim', 20, 100)
                    num_layers = trial.suggest_int('num_layers', 1, 3)
                    
                    params = {
                        'freq': freq,
                        'prediction_length': 30,
                        'num_hidden_dimensions': [hidden_dim] * num_layers,
                        'context_length': trial.suggest_int('context_length', 30, 120),
                        'trainer': Trainer(
                            epochs=10,
                            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                            batch_size=trial.suggest_int('batch_size', 16, 64)
                        )
                    }
                    model = SimpleFeedForwardEstimator(**params)
                    
                else:
                    return float('inf')
                
                # Train and evaluate
                predictor = model.train(dataset)
                
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=predictor,
                    num_samples=50  # Reduced for speed
                )
                
                forecasts = list(forecast_it)
                timeseries = list(ts_it)
                
                # Quick evaluation
                evaluator = Evaluator(quantiles=[0.5])
                agg_metrics, _ = evaluator(iter(timeseries), iter(forecasts))
                
                return agg_metrics.get('MASE', float('inf'))
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=1200)  # 20 minutes max
        
        self.best_params[f"{dataset_name}_{model_type}"] = study.best_params
        print(f"✅ Best parameters: {study.best_params}")
        print(f"✅ Best MASE: {study.best_value:.4f}")
    
    def create_probabilistic_visualization(self, dataset_name: str):
        """Create comprehensive probabilistic visualization"""
        print(f"\n📈 Creating probabilistic visualization for {dataset_name}...")
        
        if dataset_name not in self.forecasts:
            print("❌ No forecasts available")
            return
        
        forecasts_dict = self.forecasts[dataset_name]
        
        # Get the best model
        if dataset_name in self.metrics and not self.metrics[dataset_name].empty:
            best_model = self.metrics[dataset_name].iloc[0]['Model']
        else:
            best_model = list(forecasts_dict.keys())[0]
        
        if best_model not in forecasts_dict:
            print(f"❌ Best model {best_model} not in forecasts")
            return
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Probabilistic Forecasts with Confidence Intervals',
                'Model Performance Comparison',
                'Coverage Analysis',
                'Quantile Loss Analysis'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Probabilistic forecasts
        forecast_data = forecasts_dict[best_model]
        forecasts = forecast_data['forecasts']
        timeseries = forecast_data['timeseries']
        
        if forecasts and timeseries:
            forecast = forecasts[0]  # First forecast
            ts = timeseries[0]       # Corresponding time series
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=ts.index,
                    y=ts.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Forecast median
            forecast_index = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast.median),
                freq='D'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=forecast.median,
                    mode='lines+markers',
                    name='Forecast (Median)',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
            colors_ci = ['rgba(255,0,0,0.1)', 'rgba(255,0,0,0.15)', 'rgba(255,0,0,0.2)', 'rgba(255,0,0,0.25)']
            
            for i, (q_low, q_high) in enumerate([(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]):
                if hasattr(forecast, 'quantile'):
                    try:
                        lower = forecast.quantile(q_low)
                        upper = forecast.quantile(q_high)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=list(forecast_index) + list(forecast_index)[::-1],
                                y=list(lower) + list(upper)[::-1],
                                fill='toself',
                                fillcolor=colors_ci[i % len(colors_ci)],
                                line=dict(color='rgba(255,255,255,0)'),
                                name=f'{int((q_high-q_low)*100)}% CI',
                                hoverinfo="skip"
                            ),
                            row=1, col=1
                        )
                    except:
                        pass
        
        # 2. Model performance comparison
        if dataset_name in self.metrics:
            metrics_df = self.metrics[dataset_name]
            fig.add_trace(
                go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['MASE'],
                    name='MASE',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Coverage analysis
        if dataset_name in self.probabilistic_metrics:
            prob_df = self.probabilistic_metrics[dataset_name]
            coverage_cols = [col for col in prob_df.columns if 'Coverage' in col]
            
            for col in coverage_cols:
                fig.add_trace(
                    go.Bar(
                        x=prob_df['Model'],
                        y=prob_df[col],
                        name=col,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # 4. Quantile loss analysis
        if dataset_name in self.probabilistic_metrics:
            prob_df = self.probabilistic_metrics[dataset_name]
            ql_cols = [col for col in prob_df.columns if 'QuantileLoss' in col]
            
            for col in ql_cols:
                fig.add_trace(
                    go.Bar(
                        x=prob_df['Model'],
                        y=prob_df[col],
                        name=col,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text=f"🎯 GluonTS Probabilistic Analysis - {dataset_name}",
            showlegend=True
        )
        
        fig.write_html(f'gluonts_probabilistic_{dataset_name.lower()}.html')
        print(f"✅ Probabilistic visualization saved as 'gluonts_probabilistic_{dataset_name.lower()}.html'")
    
    def generate_probabilistic_report(self):
        """Generate comprehensive probabilistic analysis report"""
        print("\n📋 Generating probabilistic analysis report...")
        
        report = f"""
# 🎯 GluonTS Probabilistic Forecasting Analysis Report

## 🔮 Probabilistic Forecasting Overview
- **Framework**: GluonTS (Amazon's Probabilistic Time Series Toolkit)
- **Version**: {gluonts.__version__ if 'gluonts' in globals() else 'N/A'}
- **Focus**: Uncertainty quantification and probabilistic predictions
- **Models Tested**: {len(self.models)} probabilistic models
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Dataset Uncertainty Analysis
"""
        
        for name, df in self.datasets.items():
            target = df['target'].dropna()
            returns = target.pct_change().dropna()
            cv = (target.std() / target.mean()) * 100
            
            report += f"""
### {name.replace('_', ' ')}
- **Observations**: {len(target):,}
- **Mean Value**: {target.mean():.2f}
- **Volatility**: {target.std():.2f}
- **Coefficient of Variation**: {cv:.1f}%
- **Return Volatility**: {returns.std() * 100:.2f}%
- **Skewness**: {returns.skew():.2f}
- **Kurtosis**: {returns.kurtosis():.2f}
- **VaR (5%)**: {np.percentile(returns, 5) * 100:.2f}%
- **VaR (1%)**: {np.percentile(returns, 1) * 100:.2f}%
- **Uncertainty Level**: {'High' if cv > 20 else 'Moderate' if cv > 10 else 'Low'}
"""
        
        report += "\n## 🏆 Probabilistic Model Performance\n"
        
        for dataset_name, metrics_df in self.metrics.items():
            if not metrics_df.empty:
                report += f"\n### {dataset_name.replace('_', ' ')}\n"
                report += "**Point Forecast Accuracy**:\n"
                report += metrics_df[['Model', 'MASE', 'sMAPE', 'MSIS']].round(4).to_string(index=False)
                
                best_model = metrics_df.iloc[0]
                report += f"\n\n**Best Model**: {best_model['Model']}\n"
                report += f"- **MASE**: {best_model['MASE']:.4f}\n"
                report += f"- **sMAPE**: {best_model['sMAPE']:.2f}%\n"
                report += f"- **MSIS**: {best_model['MSIS']:.4f}\n"
        
        report += "\n## 📈 Probabilistic Performance Analysis\n"
        
        for dataset_name, prob_df in self.probabilistic_metrics.items():
            if not prob_df.empty:
                report += f"\n### {dataset_name.replace('_', ' ')}\n"
                report += "**Coverage Analysis** (Ideal coverage should match confidence level):\n"
                coverage_cols = [col for col in prob_df.columns if 'Coverage' in col]
                if coverage_cols:
                    report += prob_df[['Model'] + coverage_cols].round(3).to_string(index=False)
                
                report += "\n\n**Quantile Loss Analysis** (Lower is better):\n"
                ql_cols = [col for col in prob_df.columns if 'QuantileLoss' in col]
                if ql_cols:
                    report += prob_df[['Model'] + ql_cols].round(4).to_string(index=False)
        
        report += "\n## ⚙️ Hyperparameter Optimization Results\n"
        for key, params in self.best_params.items():
            report += f"\n### {key}\n"
            for param, value in params.items():
                report += f"- **{param}**: {value}\n"
        
        report += f"""

## 🔍 Key Probabilistic Insights
1. **Uncertainty Quantification**: Models provide confidence intervals for risk assessment
2. **Coverage Analysis**: Well-calibrated models show coverage close to nominal levels
3. **Deep Learning Advantage**: Neural models excel at capturing complex uncertainty patterns
4. **Quantile Forecasting**: Different quantiles useful for different business decisions
5. **Model Selection**: MASE for accuracy, coverage for calibration

## 💼 Business Applications
- **Risk Management**: VaR and confidence intervals for financial decisions
- **Inventory Planning**: Uncertainty bounds for demand forecasting
- **Resource Allocation**: Probabilistic forecasts for capacity planning
- **Decision Making**: Different quantiles for optimistic/pessimistic scenarios

## 🛠️ Technical Implementation
- **Deep Learning**: DeepAR, Transformer, SimpleFeedForward
- **Statistical**: Seasonal Naive, Mean Predictor, Prophet
- **Evaluation**: MASE, sMAPE, MSIS, Coverage, Quantile Loss
- **Optimization**: Optuna for hyperparameter tuning
- **Uncertainty**: Monte Carlo sampling for prediction intervals

## 📁 Generated Files
- `gluonts_probabilistic_eda.html` - Uncertainty-focused EDA
- `gluonts_probabilistic_*.html` - Individual dataset dashboards
- `gluonts_performance_*.csv` - Detailed performance metrics
- `gluonts_probabilistic_report.md` - This comprehensive report

## 🎯 GluonTS Framework Advantages
1. **Probabilistic Focus**: Built for uncertainty quantification
2. **Production Ready**: Scalable and optimized for real-world deployment
3. **Rich Models**: State-of-the-art deep learning and statistical models
4. **Comprehensive Evaluation**: Specialized metrics for probabilistic forecasting
5. **Flexible Architecture**: Easy integration with existing ML pipelines

---
*Probabilistic Analysis powered by GluonTS Framework*
*Author: Pablo Poletti | GitHub: https://github.com/PabloPoletti*
        """
        
        with open('gluonts_probabilistic_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed metrics
        for dataset_name, metrics_df in self.metrics.items():
            metrics_df.to_csv(f'gluonts_performance_{dataset_name.lower()}.csv', index=False)
        
        for dataset_name, prob_df in self.probabilistic_metrics.items():
            prob_df.to_csv(f'gluonts_probabilistic_{dataset_name.lower()}.csv', index=False)
        
        print("✅ Probabilistic report saved as 'gluonts_probabilistic_report.md'")

def main():
    """Main probabilistic analysis pipeline"""
    print("🎯 Starting GluonTS Probabilistic Forecasting Analysis")
    print("=" * 60)
    
    # Initialize analysis
    analysis = GluonTSProbabilisticAnalysis()
    
    # 1. Load probabilistic datasets
    analysis.load_probabilistic_datasets()
    
    # 2. Convert to GluonTS format
    analysis.convert_to_gluonts_format()
    
    # 3. Probabilistic EDA
    analysis.comprehensive_probabilistic_eda()
    
    # 4. Create probabilistic models
    analysis.create_probabilistic_models(prediction_length=30)
    
    # 5. Train and evaluate on each dataset
    for dataset_name in analysis.gluonts_datasets.keys():
        print(f"\n{'='*50}")
        print(f"Probabilistic Analysis: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Train models
            analysis.train_and_evaluate_probabilistic_models(dataset_name, prediction_length=30)
            
            # Create visualizations
            analysis.create_probabilistic_visualization(dataset_name)
            
            # Optimize for key datasets
            if dataset_name in ['BTC_Volatile', 'Demand_Uncertain']:
                analysis.optimize_probabilistic_hyperparameters(dataset_name, 'DeepAR')
            
        except Exception as e:
            print(f"❌ Probabilistic analysis failed for {dataset_name}: {e}")
            continue
    
    # 6. Generate probabilistic report
    analysis.generate_probabilistic_report()
    
    print("\n🎉 GluonTS Probabilistic Analysis completed successfully!")
    print("📁 Check the generated files for detailed probabilistic insights")

if __name__ == "__main__":
    main()
