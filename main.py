import matplotlib.pyplot as plt

from src.data_api import fetch_stock_data, fetch_news_data, TICKERS

from src.model import (
    plot_closing_price_histogram,
    plot_stock_time_series,
    summarize_stock_trends,
    prepare_news_volume,
    plot_daily_news_volume,
    identify_high_volume_periods,
    cross_reference_news_with_stock,
    plot_sentiment_score_distribution,
    identify_model_disagreements,
    inspect_disagreement_cases,
    generate_sentiment_comparison_summary,
    aggregate_daily_sentiment,
    save_daily_sentiment_dataset,
    create_sentiment_lag_features,
    validate_lag_alignment,
    save_lagged_dataset,
    compute_rolling_averages,
    validate_rolling_averages,
    save_rolling_average_dataset,
)

from src.sentiment import (
    apply_textblob_sentiment,
    validate_sentiment_scores,
    save_sentiment_dataset,
    apply_vader_sentiment,
    validate_vader_scores,
    save_vader_dataset,
    apply_finbert_sentiment,
    save_finbert_dataset,
)


def get_selected_tickers():
    return TICKERS[:10]


def collect_news(selected_tickers):
    all_news = []

    for ticker in selected_tickers:
        news = fetch_news_data(ticker)
        all_news.extend(news)

    return all_news


def build_full_sentiment_dataset():
    selected_tickers = get_selected_tickers()
    all_news = collect_news(selected_tickers)

    sentiment_df = apply_textblob_sentiment(all_news)
    sentiment_df = apply_vader_sentiment(sentiment_df)
    sentiment_df = apply_finbert_sentiment(sentiment_df)

    return sentiment_df


def analyze_stock_price_patterns():
    selected_tickers = get_selected_tickers()
    stock_data = fetch_stock_data(selected_tickers)

    for ticker in selected_tickers:
        print(f"\nAnalyzing stock price pattern for {ticker}...")

        plot_closing_price_histogram(stock_data, ticker)
        plot_stock_time_series(stock_data, ticker)

        summary = summarize_stock_trends(stock_data, ticker)
        print(summary)


def analyze_news_frequency():
    selected_tickers = get_selected_tickers()

    stock_data = fetch_stock_data(selected_tickers)
    all_news = collect_news(selected_tickers)

    daily_news = prepare_news_volume(all_news)

    print("\nDaily News Volume:")
    print(daily_news)

    plot_daily_news_volume(daily_news)

    high_volume_news = identify_high_volume_periods(daily_news)

    print("\nHigh-Volume News Periods:")
    print(high_volume_news)

    comparison = cross_reference_news_with_stock(high_volume_news, stock_data)

    print("\nHigh-Volume News Cross-Reference With Stock Movement:")

    if comparison.empty:
        print("No matching stock movement found for high-volume news dates.")
    else:
        print(comparison)


def analyze_headline_sentiment():
    selected_tickers = get_selected_tickers()
    all_news = collect_news(selected_tickers)

    sentiment_df = apply_textblob_sentiment(all_news)

    print("\nTextBlob Sentiment Results:")
    print(sentiment_df.head())

    validate_sentiment_scores(sentiment_df)
    save_sentiment_dataset(sentiment_df)


def analyze_vader_sentiment():
    selected_tickers = get_selected_tickers()
    all_news = collect_news(selected_tickers)

    sentiment_df = apply_textblob_sentiment(all_news)
    sentiment_df = apply_vader_sentiment(sentiment_df)

    print("\nVADER Sentiment Results:")
    print(sentiment_df.head())

    validate_sentiment_scores(sentiment_df)
    validate_vader_scores(sentiment_df)

    save_vader_dataset(sentiment_df)


def analyze_finbert_sentiment():
    sentiment_df = build_full_sentiment_dataset()

    print("\nFinBERT Sentiment Results:")

    columns = [
        "ticker",
        "headline",
        "polarity_score",
        "vader_score",
        "finbert_sentiment",
        "finbert_confidence",
    ]

    available_columns = [col for col in columns if col in sentiment_df.columns]

    print(sentiment_df[available_columns].head(20))

    save_finbert_dataset(sentiment_df)


def compare_sentiment_models():
    sentiment_df = build_full_sentiment_dataset()

    plot_sentiment_score_distribution(sentiment_df)

    disagreements_df = identify_model_disagreements(sentiment_df)

    print("\nModel Disagreements:")

    if disagreements_df.empty:
        print("No disagreement cases found.")
    else:
        columns = [
            "ticker",
            "headline",
            "sentiment",
            "vader_sentiment",
            "finbert_sentiment",
        ]

        available_columns = [
            col for col in columns if col in disagreements_df.columns
        ]

        print(disagreements_df[available_columns].head(20))

    inspect_disagreement_cases(disagreements_df)

    summary = generate_sentiment_comparison_summary(
        sentiment_df,
        disagreements_df
    )

    print(summary)


def build_daily_sentiment_dataset():
    sentiment_df = build_full_sentiment_dataset()

    daily_sentiment = aggregate_daily_sentiment(sentiment_df)

    print("\nDaily Aggregated Sentiment Scores:")
    print(daily_sentiment.head(20))

    save_daily_sentiment_dataset(daily_sentiment)

    return daily_sentiment


def build_lagged_sentiment_features():
    daily_sentiment = build_daily_sentiment_dataset()

    lagged_df = create_sentiment_lag_features(daily_sentiment)

    print("\nLagged Sentiment Features:")
    print(lagged_df.head(20))

    validate_lag_alignment(lagged_df)
    save_lagged_dataset(lagged_df)

    return lagged_df


def build_rolling_average_features():
    lagged_df = build_lagged_sentiment_features()

    rolling_df = compute_rolling_averages(lagged_df)

    print("\nRolling Average Features:")
    print(rolling_df.head(20))

    validate_rolling_averages(rolling_df)
    save_rolling_average_dataset(rolling_df)

    return rolling_df


def main():
    print("\nStarting full project pipeline...\n")

    analyze_stock_price_patterns()
    analyze_news_frequency()
    analyze_headline_sentiment()
    analyze_vader_sentiment()
    analyze_finbert_sentiment()
    compare_sentiment_models()
    build_daily_sentiment_dataset()
    build_lagged_sentiment_features()
    build_rolling_average_features()

    plt.show()

    print("\nProject pipeline completed successfully.")


if __name__ == "__main__":
    main()