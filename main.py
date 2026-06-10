import matplotlib
matplotlib.use('Agg')
import os

from src.data_api import (
    fetch_news_data,
    fetch_and_save_stock_data,
    load_kaggle_news,
    TICKERS,
    KAGGLE_TICKERS
)
from src.sentiment import (
    apply_textblob_sentiment,
    apply_vader_sentiment,
    save_sentiment_dataset,
    save_vader_dataset,
)
from src.merge_and_analyze import (
    merge_news_stock,
    run_correlation_analysis,
    run_lag_analysis,
    run_event_study,
    run_anomaly_detection,
    run_ticker_sensitivity,
    run_ml_models,
)


def step1_stock_data():
    print("\n--- STEP 1: Stock Data ---")
    if os.path.exists("stock_prices_kaggle.csv"):
        print("Already exists. Skipping.")
    else:
        fetch_and_save_stock_data(
            KAGGLE_TICKERS,
            filename="stock_prices_kaggle.csv",
            start="2010-01-01",
            end="2026-05-31"
        )


def step2_news_sentiment():
    print("\n--- STEP 2: News + Sentiment ---")
    if os.path.exists("kaggle_sentiment_scores.csv"):
        print("Already exists. Skipping.")
    else:
        news = load_kaggle_news("raw_analyst_ratings.csv")
        print(f"Headlines loaded: {len(news)}")

        import pandas as pd
        df = apply_textblob_sentiment(news)
        df = apply_vader_sentiment(df)

        df.to_csv("kaggle_sentiment_scores.csv", index=False)
        print("Saved: kaggle_sentiment_scores.csv")


def step3_newsapi_recent():
    print("\n--- STEP 3: Recent NewsAPI Headlines ---")
    if os.path.exists("sentiment_scores.csv"):
        print("Already exists. Skipping.")
    else:
        all_news = []
        for ticker in TICKERS[:10]:
            news = fetch_news_data(ticker)
            all_news.extend(news)
        print(f"Headlines fetched: {len(all_news)}")

        df = apply_textblob_sentiment(all_news)
        df = apply_vader_sentiment(df)

        save_sentiment_dataset(df)
        save_vader_dataset(df)


def step4_merge():
    print("\n--- STEP 4: Merge ---")
    if os.path.exists("merged_dataset.csv"):
        print("Already exists. Skipping.")
    else:
        merge_news_stock()


def step5_analysis():
    print("\n--- STEP 5: Analysis ---")

    if os.path.exists("correlation_results.csv"):
        print("Correlation already exists. Skipping.")
    else:
        run_correlation_analysis()

    if os.path.exists("lag_analysis_results.csv"):
        print("Lag analysis already exists. Skipping.")
    else:
        run_lag_analysis()

    if os.path.exists("event_study_results.csv"):
        print("Event study already exists. Skipping.")
    else:
        run_event_study()

    if os.path.exists("anomaly_results.csv"):
        print("Anomaly detection already exists. Skipping.")
    else:
        run_anomaly_detection()

    if os.path.exists("ticker_sensitivity.csv"):
        print("Ticker sensitivity already exists. Skipping.")
    else:
        run_ticker_sensitivity()


def step6_ml():
    print("\n--- STEP 6: ML Models ---")
    if os.path.exists("ml_results.csv"):
        print("Already exists. Skipping.")
    else:
        run_ml_models()

# Step 7 - 2026 predictions
    if os.path.exists("predictions_2026.csv"):
        print("Step 7: 2026 predictions already exist. Skipping.")
    else:
        from src.merge_and_analyze import run_2026_predictions
        run_2026_predictions()
        
def main():
    print("\n" + "="*50)
    print("AI NEWS IMPACT ANALYZER — FULL PIPELINE")
    print("="*50)

    step1_stock_data()
    step2_news_sentiment()
    step3_newsapi_recent()
    step4_merge()
    step5_analysis()
    step6_ml()

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()