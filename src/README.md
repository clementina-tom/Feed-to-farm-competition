# Source Code Overview

This directory contains the core modules for the Feed-to-Farm Machine Learning Pipeline. The architecture is intentionally decoupled into three primary stages to enforce separation of concerns and maintainability.

### Module Structure

- **`data/`**: Ingestion layer. Handles robust CSV parsing and downcasts numeric types (e.g. `float64` to `float32`) to drastically reduce memory overhead during large dataset operations.
- **`features/`**: Transformation layer. Implements complex, leak-proof aggregate features such as expanding mean buy-rates, rolling average product demand (Global Trends), and historical pair lags. It also generates the temporal target variables for the 1-week and 2-week prediction horizons.
- **`models/`**: Learning and Inference layer. 
  - `trainer.py` executes the *5-Seed Hybrid Ensemble* (LightGBM and CatBoost) and specifically targets positive-only samples for the Tweedie regressors.
  - `predictor.py` aggregates the multi-seed outputs and applies our proprietary *"Decoupled Calibration Strategy"*, isolating the AUC improvements from the MAE constraints.

### Usage
This code is executed via the root `main.py` entrypoint. However, individual classes like `FeatureEngineer` or `ModelTrainer` can be safely imported into standalone Jupyter Notebooks for experimental EDA or further hyperparameter tuning.
