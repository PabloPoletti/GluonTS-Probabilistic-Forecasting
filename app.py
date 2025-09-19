"""
🎯 GluonTS Probabilistic Forecasting Dashboard
Professional Probabilistic Time Series Forecasting with Uncertainty Quantification

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# GluonTS imports
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.model.transformer import TransformerEstimator
    from gluonts.model.prophet import ProphetPredictor
    from gluonts.model.seasonal_naive import SeasonalNaivePredictor
    from gluonts.trainer import Trainer
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    from gluonts.evaluation.backtest import make_evaluation_predictions
    import gluonts
except ImportError as e:
    st.error(f"Error importing GluonTS: {e}")
    st.stop()

# Additional imports
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="GluonTS Probabilistic Forecasting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff9a9e, #fecfef, #fecfef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .probabilistic-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .uncertainty-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9a9e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COLORS = {
    'primary': '#ff9a9e',
    'secondary': '#fecfef', 
    'accent': '#a8edea',
    'success': '#96CEB4',
    'warning': '#FECA57',
    'error': '#FF9FF3'
}

@st.cache_data
def generate_probabilistic_datasets() -> Dict[str, pd.DataFrame]:
    """Generate datasets suitable for probabilistic forecasting"""
    
    datasets = {}
    
    # 1. E-commerce sales with uncertainty
    dates = pd.date_range('2020-01-01', '2024-10-01', freq='D')
    np.random.seed(42)
    
    # Base sales pattern with volatility clusters
    base_sales = 1000
    trend = np.cumsum(np.random.normal(0.5, 2, len(dates)))
    seasonal = 300 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    weekly = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Add volatility clustering (high uncertainty periods)
    volatility = np.ones(len(dates)) * 50
    for i in range(100, len(dates)):
        if np.random.random() < 0.02:  # 2% chance of volatility spike
            volatility[i:i+30] *= 3  # 30-day high volatility period
    
    noise = np.random.normal(0, volatility)
    sales = np.maximum(base_sales + trend + seasonal + weekly + noise, 50)
    
    datasets['Ecommerce_Sales'] = pd.DataFrame({
        'ds': dates,
        'y': sales
    }).set_index('ds')
    
    # 2. Energy demand with weather uncertainty
    dates_hourly = pd.date_range('2023-01-01', '2024-09-01', freq='6H')
    np.random.seed(123)
    
    # Energy demand patterns
    base_demand = 5000
    
    # Temperature effect (heating/cooling)
    temp_cycle = 1000 * np.sin(2 * np.pi * np.arange(len(dates_hourly)) / (365*4))
    daily_cycle = 800 * np.sin(2 * np.pi * np.arange(len(dates_hourly)) / 4)
    
    # Weather uncertainty
    weather_uncertainty = np.random.gamma(2, 200, len(dates_hourly))
    
    demand = base_demand + temp_cycle + daily_cycle + weather_uncertainty
    demand = np.maximum(demand, 1000)
    
    datasets['Energy_Demand'] = pd.DataFrame({
        'ds': dates_hourly,
        'y': demand
    }).set_index('ds')
    
    # 3. Supply chain with disruption uncertainty
    dates_business = pd.date_range('2022-01-01', '2024-10-01', freq='B')
    np.random.seed(456)
    
    # Normal supply levels
    base_supply = 10000
    seasonal_supply = 2000 * np.sin(2 * np.pi * np.arange(len(dates_business)) / 260)
    
    # Random disruptions (uncertainty spikes)
    disruptions = np.zeros(len(dates_business))
    disruption_days = np.random.choice(len(dates_business), size=20, replace=False)
    
    for day in disruption_days:
        disruption_length = np.random.randint(3, 15)
        end_day = min(day + disruption_length, len(dates_business))
        disruptions[day:end_day] = -np.random.uniform(2000, 5000)
    
    normal_noise = np.random.normal(0, 300, len(dates_business))
    supply = np.maximum(base_supply + seasonal_supply + disruptions + normal_noise, 1000)
    
    datasets['Supply_Chain'] = pd.DataFrame({
        'ds': dates_business,
        'y': supply
    }).set_index('ds')
    
    return datasets

def create_gluonts_models(freq: str, prediction_length: int) -> Dict[str, object]:
    """Create GluonTS models for probabilistic forecasting"""
    
    trainer = Trainer(epochs=10, learning_rate=1e-3, batch_size=32)
    
    models = {
        'DeepAR': DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            use_feat_dynamic_real=False,
            use_feat_static_cat=False
        ),
        'SimpleFeedForward': SimpleFeedForwardEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer
        ),
        'Transformer': TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer
        ),
        'SeasonalNaive': SeasonalNaivePredictor(
            freq=freq,
            prediction_length=prediction_length,
            season_length=7 if freq == 'D' else 24
        )
    }
    
    return models

def prepare_gluonts_dataset(data: pd.DataFrame, freq: str) -> ListDataset:
    """Prepare data for GluonTS format"""
    
    # Convert to GluonTS format
    train_data = ListDataset([{
        "target": data['y'].values,
        "start": data.index[0]
    }], freq=freq)
    
    return train_data

def perform_probabilistic_evaluation(data: pd.DataFrame, models: Dict, freq: str, prediction_length: int) -> pd.DataFrame:
    """Perform probabilistic forecasting evaluation"""
    
    results = []
    
    # Prepare data
    train_data = prepare_gluonts_dataset(data.iloc[:-prediction_length], freq)
    test_data = prepare_gluonts_dataset(data, freq)
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    
    for name, model in models.items():
        try:
            st.write(f"🔄 Training {name}...")
            
            # Train model
            if hasattr(model, 'train'):
                predictor = model.train(train_data)
            else:
                predictor = model
            
            # Make predictions
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_data,
                predictor=predictor,
                num_samples=100
            )
            
            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            # Evaluate
            agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
            
            results.append({
                'Model': name,
                'MASE': agg_metrics.get('MASE', np.nan),
                'sMAPE': agg_metrics.get('sMAPE', np.nan),
                'Coverage_90': agg_metrics.get('coverage[0.9]', np.nan),
                'Coverage_50': agg_metrics.get('coverage[0.5]', np.nan)
            })
            
        except Exception as e:
            st.warning(f"Error with {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GluonTS Probabilistic Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="uncertainty-info">
    🎯 <strong>Professional Probabilistic Forecasting with Uncertainty Quantification</strong><br>
    Generate prediction intervals, quantiles, and robust uncertainty estimates for business decision-making
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Probabilistic Configuration")
        
        # Dataset selection
        datasets = generate_probabilistic_datasets()
        dataset_name = st.selectbox("📊 Select Uncertainty Dataset:", list(datasets.keys()))
        data = datasets[dataset_name]
        
        # Frequency mapping
        freq_mapping = {
            'Ecommerce_Sales': 'D',
            'Energy_Demand': '6H',
            'Supply_Chain': 'B'
        }
        freq = freq_mapping[dataset_name]
        
        # Model selection
        st.markdown("### 🤖 Probabilistic Models")
        prediction_length = st.slider("🔮 Prediction Horizon:", 7, 60, 14)
        
        available_models = create_gluonts_models(freq, prediction_length)
        
        selected_models = {}
        for model_name in available_models.keys():
            if st.checkbox(f"{model_name}", key=f"model_{model_name}"):
                selected_models[model_name] = available_models[model_name]
        
        # Uncertainty options
        st.markdown("### 📊 Uncertainty Visualization")
        show_quantiles = st.multiselect(
            "📈 Quantiles to Display:",
            [10, 25, 50, 75, 90],
            default=[10, 50, 90]
        )
        
        confidence_interval = st.selectbox("🎯 Main Confidence Interval:", [80, 90, 95], index=1)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Data visualization
        st.subheader("📊 Probabilistic Dataset Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f"Dataset: {dataset_name.replace('_', ' ')}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Uncertainty analysis
        st.markdown("### 📈 Uncertainty Analysis")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            volatility = data['y'].rolling(30).std().mean()
            st.metric("📊 Avg Volatility", f"{volatility:.1f}")
        with col_b:
            cv = (data['y'].std() / data['y'].mean()) * 100
            st.metric("📈 Coefficient of Variation", f"{cv:.1f}%")
        with col_c:
            st.metric("📅 Data Points", len(data))
        with col_d:
            st.metric("🔍 Frequency", freq)
    
    with col2:
        # Probabilistic info
        st.subheader("🎯 Uncertainty Info")
        
        # Selected models
        if selected_models:
            st.success(f"📊 {len(selected_models)} models selected")
            for model_name in selected_models.keys():
                st.markdown(f"• {model_name}")
        else:
            st.warning("⚠️ Please select at least one model")
        
        # GluonTS info
        st.info(f"🎯 GluonTS version: {gluonts.__version__}")
        
        # Uncertainty metrics
        recent_volatility = data['y'].tail(30).std()
        if recent_volatility > volatility:
            st.warning("📈 Increasing uncertainty detected")
        else:
            st.success("📉 Stable uncertainty pattern")
    
    # Probabilistic forecasting
    if selected_models and st.button("🚀 Run Probabilistic Forecasting", type="primary"):
        st.markdown("---")
        st.subheader("🎯 Probabilistic Forecasting Results")
        
        with st.spinner("Training probabilistic models and generating uncertainty estimates..."):
            # Perform evaluation
            comparison_results = perform_probabilistic_evaluation(
                data, selected_models, freq, prediction_length
            )
            
            if not comparison_results.empty:
                # Display comparison
                st.markdown("### 📊 Probabilistic Model Performance")
                
                # Sort by MASE
                comparison_results = comparison_results.sort_values('MASE')
                st.dataframe(comparison_results, hide_index=True)
                
                # Coverage analysis
                st.markdown("### 🎯 Coverage Analysis")
                
                col_coverage1, col_coverage2 = st.columns(2)
                
                with col_coverage1:
                    if 'Coverage_90' in comparison_results.columns:
                        avg_coverage_90 = comparison_results['Coverage_90'].mean()
                        st.metric("📊 Average 90% Coverage", f"{avg_coverage_90:.1%}")
                
                with col_coverage2:
                    if 'Coverage_50' in comparison_results.columns:
                        avg_coverage_50 = comparison_results['Coverage_50'].mean()
                        st.metric("📊 Average 50% Coverage", f"{avg_coverage_50:.1%}")
                
                # Generate sample probabilistic forecast
                st.markdown("### 🔮 Sample Probabilistic Forecast")
                
                if 'SeasonalNaive' in selected_models:
                    try:
                        # Use SeasonalNaive for demo (fastest)
                        train_data = prepare_gluonts_dataset(data.iloc[:-prediction_length], freq)
                        predictor = selected_models['SeasonalNaive']
                        
                        # Make prediction
                        forecast_it = predictor.predict(train_data)
                        forecast = next(iter(forecast_it))
                        
                        # Create probabilistic visualization
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=data.index[:-prediction_length],
                            y=data['y'].iloc[:-prediction_length],
                            mode='lines',
                            name='Training Data',
                            line=dict(color=COLORS['primary'], width=2)
                        ))
                        
                        # Actual test data
                        fig.add_trace(go.Scatter(
                            x=data.index[-prediction_length:],
                            y=data['y'].iloc[-prediction_length:],
                            mode='lines',
                            name='Actual',
                            line=dict(color=COLORS['secondary'], width=2)
                        ))
                        
                        # Forecast median
                        forecast_index = pd.date_range(
                            start=data.index[-prediction_length],
                            periods=prediction_length,
                            freq=freq
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_index,
                            y=forecast.median,
                            mode='lines+markers',
                            name='Forecast Median',
                            line=dict(color=COLORS['accent'], width=3, dash='dash')
                        ))
                        
                        # Confidence intervals
                        for quantile in [0.1, 0.9]:
                            quantile_values = forecast.quantile(quantile)
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=quantile_values,
                                mode='lines',
                                name=f'{int(quantile*100)}% Quantile',
                                line=dict(color=COLORS['warning'], width=1, dash='dot'),
                                opacity=0.7
                            ))
                        
                        # Fill between quantiles
                        fig.add_trace(go.Scatter(
                            x=list(forecast_index) + list(forecast_index)[::-1],
                            y=list(forecast.quantile(0.1)) + list(forecast.quantile(0.9))[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 202, 87, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='80% Confidence Interval',
                            hoverinfo="skip"
                        ))
                        
                        fig.update_layout(
                            title="🎯 Probabilistic Forecast with Uncertainty Bands",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            template="plotly_white",
                            height=600,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating probabilistic forecast: {str(e)}")
                
                # Download results
                csv = comparison_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Probabilistic Results",
                    data=csv,
                    file_name=f"gluonts_probabilistic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🎯 <strong>GluonTS Probabilistic Forecasting</strong> | 
    Built with GluonTS framework | 
    <a href="https://github.com/PabloPoletti" target="_blank">GitHub</a> | 
    <a href="mailto:lic.poletti@gmail.com">Contact</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
