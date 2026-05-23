import pandas as pd

from src.data_api import fetch_news_data, TICKERS
from src.sentiment import (
    apply_textblob_sentiment,
    apply_vader_sentiment,
    apply_finbert_sentiment,
)
from src.model import (
    aggregate_daily_sentiment,
    create_sentiment_lag_features,
    compute_rolling_averages,
)


def collect_news_data(tickers):
    """
    Fetch news articles for selected tickers.
    """
    all_news = []

    for ticker in tickers:
        news = fetch_news_data(ticker)
        all_news.extend(news)

    return all_news


def run_sentiment_pipeline(news_articles):
    """
    Apply TextBlob, VADER, and FinBERT sentiment scoring.
    """
    sentiment_df = apply_textblob_sentiment(news_articles)
    sentiment_df = apply_vader_sentiment(sentiment_df)
    sentiment_df = apply_finbert_sentiment(sentiment_df)

    return sentiment_df


def run_feature_engineering_pipeline(sentiment_df):
    """
    Create daily sentiment aggregation, lag features,
    and rolling average features.
    """
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    lagged_df = create_sentiment_lag_features(daily_sentiment)
    rolling_df = compute_rolling_averages(lagged_df)

    return rolling_df


def run_full_pipeline(tickers=None, output_file="final_sentiment_features.csv"):
    """
    Run full sentiment and feature engineering workflow.
    """
    if tickers is None:
        tickers = TICKERS[:10]

    news_articles = collect_news_data(tickers)

    sentiment_df = run_sentiment_pipeline(news_articles)

    final_df = run_feature_engineering_pipeline(sentiment_df)

    if not final_df.empty:
        final_df.to_csv(output_file, index=False)
        print(f"\nFinal pipeline output saved as: {output_file}")
    else:
        print("\nPipeline completed, but output dataset is empty.")

    return final_df