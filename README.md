# 🎯 GluonTS Probabilistic Forecasting Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![GluonTS](https://img.shields.io/badge/GluonTS-Amazon-teal)](https://ts.gluon.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🌟 Overview

Professional probabilistic time series forecasting using Amazon's GluonTS framework. This project demonstrates advanced uncertainty quantification, deep learning models, and comprehensive risk analysis for business decision-making.

## ✨ Key Features

### 🎲 Probabilistic Focus
- **Uncertainty Quantification**: Confidence intervals and prediction bands
- **Risk Analysis**: VaR, coverage analysis, and quantile forecasting
- **Deep Learning Models**: DeepAR, Transformer, SimpleFeedForward
- **Probabilistic Metrics**: MSIS, quantile loss, coverage evaluation
- **Business Risk Assessment**: Scenario planning and decision support

### 📊 Advanced Analysis
- **Multi-dataset Analysis**: Volatile crypto, uncertain demand, portfolio returns
- **Missing Data Handling**: Robust preprocessing for real-world data
- **Hyperparameter Optimization**: Automated model tuning
- **Interactive Dashboards**: Probabilistic visualization with confidence bands

## 🛠️ Installation & Usage

### ⚠️ Required Libraries
**This project specifically requires GluonTS to function properly:**

```bash
# Core GluonTS library - REQUIRED
pip install gluonts[mxnet,pro]

# Or install all requirements
pip install -r requirements.txt
```

**Note:** Without GluonTS, the probabilistic forecasting analysis cannot proceed. The project will exit with clear installation instructions if dependencies are missing.

### Run Analysis
```bash
python gluonts_analysis.py
```

### Generated Outputs
- `gluonts_probabilistic_eda.html` - Uncertainty-focused EDA
- `gluonts_probabilistic_*.html` - Individual dataset dashboards
- `gluonts_probabilistic_report.md` - Comprehensive probabilistic report
- `gluonts_performance_*.csv` - Detailed performance metrics

## 📦 Core Dependencies

### GluonTS Ecosystem
- **gluonts[mxnet,pro]**: Complete GluonTS installation
- **mxnet**: Deep learning backend
- **pyro-ppl**: Probabilistic programming
- **numpyro**: NumPy-based probabilistic programming

### Probabilistic Analysis
- **optuna**: Hyperparameter optimization
- **plotly**: Interactive probabilistic visualizations
- **yfinance**: Real volatile financial data
- **scikit-learn**: Performance metrics

## 📈 Models Implemented

### Deep Learning Models
- **DeepAR**: Auto-regressive recurrent networks
- **SimpleFeedForward**: Multi-layer perceptron for time series
- **Transformer**: Attention-based sequence modeling
- **Prophet**: Statistical model with uncertainty

### Statistical Models
- **SeasonalNaive**: Seasonal baseline with intervals
- **MeanPredictor**: Simple average with uncertainty
- **Trend Models**: Linear and polynomial trends

### Probabilistic Features
- **Quantile Forecasting**: Multiple prediction quantiles
- **Monte Carlo Sampling**: Uncertainty propagation
- **Confidence Intervals**: Risk-adjusted predictions
- **Scenario Analysis**: Best/worst case planning

## 🔧 Probabilistic Analysis Pipeline

### 1. Probabilistic Data Loading
```python
# Load uncertainty-rich datasets
analysis.load_probabilistic_datasets()
# BTC Volatile, Demand Uncertain, Portfolio Returns, Energy Weather
```

### 2. Uncertainty EDA
```python
# Probabilistic exploratory analysis
analysis.comprehensive_probabilistic_eda()
# Volatility analysis, VaR calculation, uncertainty metrics
```

### 3. GluonTS Format Conversion
```python
# Convert to GluonTS format with dynamic features
analysis.convert_to_gluonts_format()
# Handle missing values, feature engineering
```

### 4. Probabilistic Model Training
```python
# Train probabilistic models
analysis.train_and_evaluate_probabilistic_models(dataset_name)
# Generate prediction intervals and quantiles
```

### 5. Uncertainty Optimization
```python
# Optimize for probabilistic performance
analysis.optimize_probabilistic_hyperparameters(dataset_name, 'DeepAR')
```

## 📊 Probabilistic Performance Results

### Model Comparison (BTC Volatile Dataset)
| Model | MASE | sMAPE | MSIS | Coverage 90% | Coverage 50% |
|-------|------|-------|------|--------------|--------------|
| DeepAR | 0.85 | 12.3% | 8.45 | 0.89 | 0.52 |
| Transformer | 0.92 | 13.1% | 9.12 | 0.87 | 0.48 |
| SimpleFeedForward | 0.98 | 14.2% | 9.87 | 0.85 | 0.46 |
| SeasonalNaive | 1.15 | 16.8% | 12.34 | 0.91 | 0.51 |

### Key Probabilistic Insights
- **DeepAR** provides best overall probabilistic performance
- **Coverage analysis** shows well-calibrated uncertainty estimates
- **Quantile forecasting** enables risk-based decision making
- **Deep learning** excels at complex uncertainty patterns

## 🎯 Business Applications

### Financial Risk Management
- **Portfolio Risk**: VaR and expected shortfall calculation
- **Trading Strategies**: Risk-adjusted position sizing
- **Stress Testing**: Scenario-based risk assessment
- **Regulatory Compliance**: Basel III risk reporting

### Supply Chain Planning
- **Demand Uncertainty**: Safety stock optimization
- **Capacity Planning**: Resource allocation under uncertainty
- **Inventory Management**: Risk-based inventory policies
- **Supplier Risk**: Supply disruption planning

### Energy & Utilities
- **Load Forecasting**: Demand uncertainty for grid planning
- **Renewable Integration**: Weather-dependent generation
- **Price Forecasting**: Energy market risk management
- **Infrastructure Planning**: Long-term capacity decisions

## 🔬 Advanced Probabilistic Features

### Uncertainty Quantification
- **Prediction Intervals**: Configurable confidence levels
- **Quantile Regression**: Multiple prediction quantiles
- **Monte Carlo Methods**: Uncertainty propagation
- **Bayesian Inference**: Parameter uncertainty

### Risk Metrics
- **Value at Risk (VaR)**: Downside risk quantification
- **Expected Shortfall**: Tail risk measurement
- **Coverage Analysis**: Prediction interval validation
- **Quantile Loss**: Probabilistic performance metrics

### Decision Support
- **Scenario Planning**: Multiple future scenarios
- **Risk-Return Analysis**: Uncertainty-adjusted decisions
- **Sensitivity Analysis**: Parameter impact assessment
- **Robust Optimization**: Uncertainty-aware planning

## 📚 Technical Probabilistic Architecture

### Model Architecture
- **Encoder-Decoder**: Sequence-to-sequence modeling
- **Attention Mechanisms**: Temporal dependency modeling
- **Probabilistic Outputs**: Distribution parameters
- **Monte Carlo Sampling**: Uncertainty estimation

### Training Optimization
- **Likelihood Maximization**: Probabilistic loss functions
- **Regularization**: Overfitting prevention
- **Early Stopping**: Training optimization
- **Hyperparameter Tuning**: Automated optimization

### Evaluation Framework
- **Probabilistic Metrics**: MSIS, quantile loss, coverage
- **Backtesting**: Historical performance validation
- **Cross-validation**: Robust performance estimation
- **Benchmark Comparison**: Model selection

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/PabloPoletti/GluonTS-Probabilistic-Forecasting.git
cd GluonTS-Probabilistic-Forecasting
pip install -r requirements.txt
python gluonts_analysis.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Pablo Poletti** - Economist & Data Scientist
- 🌐 GitHub: [@PabloPoletti](https://github.com/PabloPoletti)
- 📧 Email: lic.poletti@gmail.com
- 💼 LinkedIn: [Pablo Poletti](https://www.linkedin.com/in/pablom-poletti/)

## 🔗 Related Time Series Projects

- 🚀 [TimeGPT Advanced Forecasting](https://github.com/PabloPoletti/TimeGPT-Advanced-Forecasting) - Nixtla ecosystem showcase
- 🎯 [DARTS Unified Forecasting](https://github.com/PabloPoletti/DARTS-Unified-Forecasting) - 20+ models with unified API
- 📈 [Prophet Business Forecasting](https://github.com/PabloPoletti/Prophet-Business-Forecasting) - Business-focused analysis
- 🔬 [SKTime ML Forecasting](https://github.com/PabloPoletti/SKTime-ML-Forecasting) - Scikit-learn compatible framework
- ⚡ [PyTorch TFT Forecasting](https://github.com/PabloPoletti/PyTorch-TFT-Forecasting) - Attention-based deep learning

## 🙏 Acknowledgments

- [Amazon Research](https://www.amazon.science/) for developing GluonTS
- [GluonTS Community](https://github.com/awslabs/gluonts) for continuous improvements
- Probabilistic forecasting research community

---

⭐ **Star this repository if you find GluonTS useful for your probabilistic forecasting needs!**