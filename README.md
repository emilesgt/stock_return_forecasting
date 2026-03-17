# Stock Return Forecasting

Machine learning pipeline for **short-term stock direction prediction** using technical indicators and non-linear classification models.

This project builds a clean end-to-end workflow: data collection, preprocessing, feature engineering, time-aware train/validation/test splitting, model comparison, hyperparameter tuning, and benchmark comparison against a simple buy-and-hold strategy.

---

## Project overview

The objective is to predict the **next-day market direction** (up/down) from historical price and volume data.

The notebook develops the full pipeline on equity index data downloaded with `yfinance`, with a primary modeling workflow built on the **S&P 500 dataset**, and additional downloaded datasets for the **Dow Jones Industrial Average** and **CAC 40**.

The repository includes:

- historical CSV datasets
- the main Jupyter notebook containing the full pipeline
- a final report in PDF format
- a presentation video

---

## Repository structure

```text
stock_return_forecasting/
├── CAC40_data.csv
├── DJ30_data.csv
├── SNP500_data.csv
├── projet_ml.ipynb
├── rapport_final.pdf
├── video_Sagot_Wyseur_Godet.mp4
├── readme.txt
└── github link.txt
```

---

## Methodology

### 1. Data collection

Market data is downloaded with `yfinance` for:

- **DJ30** (`^DJI`)
- **S&P 500** (`^GSPC`)
- **CAC 40** (`^FCHI`)

The notebook uses OHLCV fields:

- Open
- High
- Low
- Close
- Volume

The download window runs from **2010-01-01 to 2025-01-01**.

### 2. Target construction

The classification target is the **next-day return direction**:

- compute daily return
- shift it by one day forward
- assign class `1` if the next-day return is positive, `0` otherwise

This turns the problem into a **binary classification task**.

### 3. Feature engineering

The notebook creates several technical indicators and lagged predictors, including:

- **MACD**
  - MACD line
  - signal line
  - histogram
- **RSI (14)**
- **SMA 20**
- **SMA 50**
- **EMA 20**
- **Close − SMA20**
- **Close − SMA50**
- **10-day momentum**
- **20-day momentum**

To avoid look-ahead bias, engineered predictors are shifted before modeling.

### 4. Time-aware split

The modeling dataset is split chronologically:

- **Train:** 2015–2019
- **Validation:** 2020–2022
- **Test:** 2023–2024 / early 2023 test window in the notebook workflow

This preserves the temporal structure of financial data and avoids leakage from future observations.

### 5. Models compared

The notebook evaluates several classifiers:

- **Logistic Regression** — linear baseline
- **Decision Tree** — non-linear baseline
- **Random Forest** — ensemble model
- **Gradient Boosting** — boosted trees

A final tuning stage compares top Random Forest and Gradient Boosting candidates using a validation-based selection process.

### 6. Strategy evaluation

The final classifier is converted into a simple directional trading rule:

- invest only when the model predicts **up**
- otherwise stay out of the market

This **ML buy-only strategy** is then compared against a **buy-and-hold benchmark** on the test set using:

- cumulative return
- Sharpe ratio

---

## Main results

According to the final notebook outputs, the selected model is a **Random Forest** with tuned parameters:

- `max_depth = 3`
- `min_samples_leaf = 20`
- `n_estimators = 1000`

### Final test classification results

- **Accuracy:** 0.5565
- **Precision:** 0.5545
- **Recall:** 0.9104
- **F1-score:** 0.6893
- **ROC-AUC:** 0.5022

### Strategy comparison on the test set

| Metric | Buy & Hold | ML Strategy |
|---|---:|---:|
| Total return | 0.1651 | 0.1917 |
| Sharpe ratio | 2.2201 | 2.6622 |

These results suggest that while the classification edge remains modest, the trading rule derived from the model slightly outperforms the benchmark in this notebook’s test evaluation.

---

## Tech stack

- **Python**
- **Jupyter Notebook**
- **pandas**
- **numpy**
- **matplotlib**
- **scikit-learn**
- **scipy**
- **yfinance**

---

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/emilesgt/stock_return_forecasting.git
cd stock_return_forecasting
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn scipy yfinance jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
projet_ml.ipynb
```

---

## Possible improvements

A few natural extensions for the project would be:

- add more market features and macro variables
- test additional horizons beyond 1-day prediction
- introduce transaction costs and turnover penalties
- compare classification with direct return forecasting
- evaluate other models such as XGBoost, LightGBM, or LSTM-based approaches
- perform walk-forward backtesting with repeated re-training

---

## Authors

- **Emile SAGOT**
- **Capucine WYSEUR**
- **Chiara GODET**

---

## Disclaimer

This repository is an academic machine learning project for financial prediction research and should not be interpreted as investment advice.
