# 🥕 Feed-to-Farm: Fresh Produce Purchase Predictor

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/clementina-tom/Feed-to-farm-competition)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)](https://catboost.ai/)
[![Score](https://img.shields.io/badge/Private%20Score-0.945-brightgreen)]()
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Zindi Competition](https://img.shields.io/badge/Zindi-Competition-20BEFF)

> **About this project**: A modular, production-ready Machine Learning pipeline that predicts future shopping baskets for surplus fresh produce. Built with a 5-seed Hybrid Ensemble (LGBM + CatBoost) to optimize both AUC and MAE.

---

## 🌱 The Problem

**Farm To Feed** connects farmers, businesses and consumers with "odd-looking" fruits and vegetables that would otherwise go to waste. By selling otherwise wasted fresh produce, Farm to Feed is reducing food waste, boosting farmer incomes, and making fresh nutritious produce more accessible.

The challenge was to build a **Recommender System** using historical transaction data that:
1. 📅 Generates a **likelihood of purchase in 7 and 14 days** for each product-customer pair.
2. 📦 Recommends the **exact quantity** of each product a customer will purchase.

This solution helps Farm to Feed move surplus produce efficiently, reduce food waste, and expand market access for smallholder farmers.

---

## 🏆 Results

| Metric | Score |
|---|---|
| **Public Score** | 0.945 |
| **Private Score** | 0.934 |
| Target Purchase 1 Week (AUC) | 0.912 |
| Target Purchase 2 Week (AUC) | 0.878 |
| Target Qty 1 Week (MAE) | 1.313 |
| Target Qty 2 Week (MAE) | 2.544 |

---

## 🧠 Strategy: "Hybrid Grandmaster Ensemble"

This pipeline merges two high-performing approaches from the research phase:

| Component | Technique |
|---|---|
| **AUC Optimization** | LightGBM + CatBoost classification with probability calibration |
| **MAE Optimization** | Tweedie regression trained on positive-quantity samples only |
| **Stability** | 5-seed ensembling (seeds: 42, 202, 777, 1337, 999) |
| **Decoupled Scaling** | Purchase probabilities scaled for AUC; quantity kept unsupervised with low-confidence thresholds |

### Evaluation
> This competition uses a **50/50 weighted blend** of AUC (binary purchase prediction) and MAE (quantity estimation). This dual objective is the reason for our "decoupled" strategy — treating rank-order and magnitude as distinct problems.

### Feature Engineering ("Grandmaster" Features)
- **Pair Lag Features**: Prior week quantities for each (customer, product) pair.
- **Customer Momentum**: Customer-level rolling averages and expanding purchase rates.
- **Global Product Trends**: Market-wide product demand trend with 1-week lag.
- **Recency**: Weeks since last purchase for each customer-product pair.
- **Seasonality**: Month and ISO week-of-year features.

---

## 🗂️ Project Structure

```plaintext
├── config/
│   └── config.yaml          # Hyperparameters, Paths, Ensemble Weights
├── src/
│   ├── data/
│   │   └── loader.py        # Data ingestion & memory-optimized downcasting
│   ├── features/
│   │   └── engineer.py      # Grandmaster feature engineering & target generation
│   └── models/
│       ├── trainer.py       # 5-Seed Hybrid LGBM + CatBoost training
│       └── predictor.py     # Blending, Decoupled Calibration & submission export
├── main.py                  # Pipeline execution entrypoint
├── requirements.txt         # Project dependencies
└── pipeline_redo.ipynb      # Original research notebook
```

---

## 🚀 Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Place competition dataset files in the root directory:**
```
Train.csv, Test.csv, customer_data.csv, sku_data.csv
```

**3. Run the full pipeline:**
```bash
python main.py
```

Output submission file will be saved to: `output/submission_hybrid_ensemble.csv`

---

## ⚙️ Configuration

All hyperparameters, scaling factors, and ensemble weights are managed in [`config/config.yaml`](config/config.yaml). Key settings:

```yaml
model:
  seeds: [42, 202, 777, 1337, 999]   # Seeds for stability
  n_estimators: 2000
  learning_rate: 0.02

scaling:
  purchase_1w_scale: 1.15            # Probability calibration for AUC
  qty_1w_threshold: 0.015            # Low-confidence noise elimination
```
