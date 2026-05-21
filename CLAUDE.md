# Ride Surge Pricing Predictor

**Project:** Ride surge pricing predictor
**Owner:** jatindubey9389-commits

## Stack
- Python, pandas, scikit-learn, xgboost, matplotlib, seaborn, jupyter

## Structure
- `notebooks/` — EDA and exploration
- `src/` — data pipeline, feature engineering, training
- `tests/` — pytest unit tests

## Goals
Predict surge multiplier by zone/time using NYC TLC yellow taxi data. Target: `surge_label` (1 if fare > 1.5× median).

## Conventions
- Type hints on all functions
- Docstrings on public functions
- Commit after every working feature
