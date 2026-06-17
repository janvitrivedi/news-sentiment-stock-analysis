# AI-Driven Stock Price Prediction Using Financial News Sentiment

DAMO 699 – Capstone Project | Group 3 | University of Niagara Falls Canada | June 2026

Team: Rohit Singh Sidhu, Janvi Trivedi, Bidita Saha, Devansh Patel

Supervisor: Dr. William Pourmajidi

---

## Project Overview

This project builds a pipeline that scores financial news sentiment and predicts
the directional movement of NASDAQ stock prices. Trained on 68,043 headlines
from 47 companies (2009–2019) and validated on 979 live headlines (May–June 2026).

---

## Key Results

- Logistic Regression Accuracy: 66.1%
- ROC-AUC: 0.724
- Above random baseline: +16.1 percentage points
- Tickers with significant correlation: 42 out of 47
- Strongest correlation: ASML at r = 0.418
- Anomaly rate: 24.8%

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the full pipeline:
   python main.py

3. Launch the dashboard:
   streamlit run app.py

---

## Project Structure

- app.py — Streamlit dashboard
- main.py — Main pipeline runner
- src/data_api.py — Data collection (yfinance, NewsAPI)
- src/sentiment.py — TextBlob and VADER scoring
- src/merge_and_analyze.py — Merging, feature engineering, ML models
- requirements.txt — Dependencies

Data files:
- merged_dataset.csv — 27,803 rows, main modelling dataset
- kaggle_sentiment_scores.csv — 68,043 scored headlines
- correlation_results.csv — Pearson r per ticker
- lag_analysis_results.csv — Lag-1 and lag-2 correlations
- anomaly_results.csv — Isolation Forest output
- event_study_results.csv — Day 0 to Day 5 returns
- ml_results.csv — Model performance metrics
- predictions_2026.csv — May–June 2026 live predictions

---

## Tech Stack

- Language: Python 3.10+
- NLP: TextBlob, VADER
- ML: scikit-learn (Logistic Regression, Random Forest, Isolation Forest)
- Dashboard: Streamlit
- Data: pandas, numpy, yfinance, NewsAPI
- Version Control: GitHub
- Project Management: Taiga (CRISP-DM, 5 sprints)

---

## Data Sources

- Kaggle Financial News Dataset: 68,043 headlines (2009–2019)
- Yahoo Finance via yfinance: Daily stock prices (2009–2020)
- NewsAPI: 979 live headlines (May–June 2026)
