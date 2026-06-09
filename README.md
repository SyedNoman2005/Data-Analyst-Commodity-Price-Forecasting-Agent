# Commodity Price Forecasting Agent

A silver commodity price forecasting project combining time series modeling, feature engineering, and hybrid forecasting models.

## Project Structure

- `run_modeling.py` - Main entrypoint script for data loading, preprocessing, feature engineering, and model execution.
- `generate_notebooks.py` - Script to generate project notebooks programmatically.
- `generate_reports.py` - Script to generate report markdown files.
- `data/` - Raw datasets for silver price, macroeconomic, sentiment, futures, and supply-demand.
- `models/` - Model output and serialized model artifacts.
- `notebooks/` - Jupyter notebooks for EDA, preprocessing, feature engineering, and model experiments.
- `reports/` - Markdown reports and presentation artifacts.
- `web/` - Simple dashboard files for visualizing model outputs.

## Recommended Environment

- Python: `3.10+` (tested with `3.14`)
- Virtual environment recommended: `python -m venv .venv`

## Recommended dependencies

```txt
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
xgboost>=1.7
tensorflow>=2.15
prophet>=1.1
pmdarima>=2.0
hmmlearn>=0.3
arch>=5.0
openpyxl>=3.1
matplotlib>=3.8
seaborn>=0.13
statsmodels>=0.14
