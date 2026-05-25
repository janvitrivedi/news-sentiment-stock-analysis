import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# STOCK PRICE DISTRIBUTION
# -------------------------------
def plot_closing_price_histogram(stock_data, ticker):
    data = stock_data[ticker].copy()

    if data.empty:
        print(f"No stock data available for {ticker}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(data["Close"].dropna(), bins=30)
    plt.title(f"{ticker} Daily Closing Price Distribution")
    plt.xlabel("Closing Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# -------------------------------
# STOCK TIME SERIES
# -------------------------------
def plot_stock_time_series(stock_data, ticker):
    data = stock_data[ticker].copy()

    if data.empty:
        print(f"No stock data available for {ticker}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"])
    plt.title(f"{ticker} Stock Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------
# STOCK TREND SUMMARY + OUTLIERS
# -------------------------------
def summarize_stock_trends(stock_data, ticker):
    data = stock_data[ticker].copy()

    if data.empty:
        return {"Ticker": ticker, "Message": "No data available"}

    data["daily_return"] = data["Close"].pct_change(fill_method=None)

    first_close = data["Close"].iloc[0]
    last_close = data["Close"].iloc[-1]

    mean_return = data["daily_return"].mean()
    std_return = data["daily_return"].std()

    data["z_score"] = (data["daily_return"] - mean_return) / std_return
    data["is_outlier"] = data["z_score"].abs() > 3

    return {
        "Ticker": ticker,
        "Start Price": round(first_close, 2),
        "End Price": round(last_close, 2),
        "Price Change %": round(((last_close - first_close) / first_close) * 100, 2),
        "Highest Close": round(data["Close"].max(), 2),
        "Lowest Close": round(data["Close"].min(), 2),
        "Outlier Days": int(data["is_outlier"].sum())
    }


# -------------------------------
# NEWS VOLUME PREPARATION
# -------------------------------
def prepare_news_volume(news_articles):
    news_df = pd.DataFrame(news_articles)

    if news_df.empty:
        return pd.DataFrame(columns=["date", "article_count"])

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce").dt.date

    daily_news = (
        news_df.groupby("date")
        .size()
        .reset_index(name="article_count")
    )

    return daily_news


# -------------------------------
# NEWS VOLUME CHART
# -------------------------------
def plot_daily_news_volume(daily_news):
    if daily_news.empty:
        print("No news data available to plot.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(daily_news["date"], daily_news["article_count"], marker="o")
    plt.title("Daily News Article Volume")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------
# HIGH-VOLUME NEWS DETECTION
# -------------------------------
def identify_high_volume_periods(daily_news, threshold=0):

    if daily_news.empty:
        return pd.DataFrame()

    daily_news = daily_news.copy()

    mean_count = daily_news["article_count"].mean()
    std_count = daily_news["article_count"].std()

    if pd.isna(std_count):
        std_count = 0

    daily_news["is_high_volume"] = (
        daily_news["article_count"] >= mean_count + threshold
    )

    return daily_news[daily_news["is_high_volume"] == True]
# -------------------------------
# NEWS + STOCK CROSS REFERENCE
# -------------------------------
def cross_reference_news_with_stock(high_volume_news, stock_data):
    import pandas as pd

    if high_volume_news.empty:
        return pd.DataFrame()

    results = []

    high_volume_news = high_volume_news.copy()
    high_volume_news["date"] = pd.to_datetime(high_volume_news["date"]).dt.strftime("%Y-%m-%d")

    for ticker, data in stock_data.items():
        if data.empty:
            continue

        temp = data.copy()

        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)

        temp = temp.reset_index()

        temp["date"] = pd.to_datetime(temp["Date"]).dt.strftime("%Y-%m-%d")
        temp["Ticker"] = ticker
        temp["daily_return"] = temp["Close"].pct_change(fill_method=None)

        stock_part = temp[["date", "Ticker", "Close", "daily_return"]].copy()
        news_part = high_volume_news[["date", "article_count"]].copy()

        merged = pd.merge(stock_part, news_part, on="date", how="inner")

        if not merged.empty:
            results.append(merged)

    if len(results) == 0:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)

def plot_sentiment_score_distribution(sentiment_df):
    if sentiment_df.empty:
        print("No sentiment data available to plot.")
        return

    plt.figure(figsize=(10, 5))

    if "polarity_score" in sentiment_df.columns:
        plt.hist(sentiment_df["polarity_score"].dropna(), bins=20, alpha=0.5, label="TextBlob")

    if "vader_score" in sentiment_df.columns:
        plt.hist(sentiment_df["vader_score"].dropna(), bins=20, alpha=0.5, label="VADER")

    if "finbert_confidence" in sentiment_df.columns:
        plt.hist(sentiment_df["finbert_confidence"].dropna(), bins=20, alpha=0.5, label="FinBERT Confidence")

    plt.title("Sentiment Score Distribution Across Models")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def identify_model_disagreements(sentiment_df):
    if sentiment_df.empty:
        return pd.DataFrame()

    df = sentiment_df.copy()

    required_cols = ["sentiment", "vader_sentiment", "finbert_sentiment"]

    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return pd.DataFrame()

    df["finbert_sentiment_clean"] = df["finbert_sentiment"].str.capitalize()

    df["models_disagree"] = (
        (df["sentiment"] != df["vader_sentiment"]) |
        (df["sentiment"] != df["finbert_sentiment_clean"]) |
        (df["vader_sentiment"] != df["finbert_sentiment_clean"])
    )

    return df[df["models_disagree"] == True]


def inspect_disagreement_cases(disagreements_df, sample_size=10):
    if disagreements_df.empty:
        print("No disagreement cases found.")
        return

    sample = disagreements_df.head(sample_size)

    print("\nManual Inspection of Disagreement Cases:\n")

    for _, row in sample.iterrows():
        print("Headline:", row["headline"])
        print("TextBlob:", row["sentiment"], "| Score:", row.get("polarity_score"))
        print("VADER:", row["vader_sentiment"], "| Score:", row.get("vader_score"))
        print("FinBERT:", row["finbert_sentiment"], "| Confidence:", row.get("finbert_confidence"))
        print("-" * 70)


def generate_sentiment_comparison_summary(sentiment_df, disagreements_df):
    if sentiment_df.empty:
        return "No sentiment data available for comparison."

    total = len(sentiment_df)
    disagreement_count = len(disagreements_df)
    disagreement_rate = round((disagreement_count / total) * 100, 2)

    summary = f"""
Sentiment Model Comparison Summary

Total headlines analyzed: {total}
Total disagreement cases: {disagreement_count}
Disagreement rate: {disagreement_rate}%

TextBlob is useful as a simple baseline because it is fast and easy to interpret, but it may miss financial context.
VADER handles short headline-style text better and is useful for polarity comparison.
FinBERT is the strongest finance-specific model because it understands financial language better, but it requires more processing time.

Overall, FinBERT is preferred for final sentiment scoring, while TextBlob and VADER are useful as baseline comparison models.
"""

    return summary

def aggregate_daily_sentiment(sentiment_df):
    if sentiment_df.empty:
        print("No sentiment data available for aggregation.")
        return pd.DataFrame()

    df = sentiment_df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    aggregation = {
        "headline": "count"
    }

    if "polarity_score" in df.columns:
        aggregation["polarity_score"] = "mean"

    if "vader_score" in df.columns:
        aggregation["vader_score"] = "mean"

    if "finbert_confidence" in df.columns:
        aggregation["finbert_confidence"] = "mean"

    daily_sentiment = (
        df.groupby(["date", "ticker"])
        .agg(aggregation)
        .reset_index()
    )

    daily_sentiment = daily_sentiment.rename(columns={
        "headline": "daily_article_count",
        "polarity_score": "avg_textblob_score",
        "vader_score": "avg_vader_score",
        "finbert_confidence": "avg_finbert_confidence"
    })

    return daily_sentiment


def save_daily_sentiment_dataset(daily_sentiment, filename="daily_sentiment_scores.csv"):
    if daily_sentiment.empty:
        print("No daily sentiment data available to save.")
        return

    daily_sentiment.to_csv(filename, index=False)

    print(f"\nDaily aggregated sentiment dataset saved as: {filename}")

def create_sentiment_lag_features(daily_sentiment):
    if daily_sentiment.empty:
        print("No daily sentiment data available.")
        return pd.DataFrame()

    df = daily_sentiment.copy()

    df = df.sort_values(by=["ticker", "date"])

    # -------------------------------
    # TEXTBLOB LAG FEATURES
    # -------------------------------
    if "avg_textblob_score" in df.columns:

        df["textblob_lag_1"] = (
            df.groupby("ticker")["avg_textblob_score"]
            .shift(1)
        )

        df["textblob_lag_2"] = (
            df.groupby("ticker")["avg_textblob_score"]
            .shift(2)
        )

    # -------------------------------
    # VADER LAG FEATURES
    # -------------------------------
    if "avg_vader_score" in df.columns:

        df["vader_lag_1"] = (
            df.groupby("ticker")["avg_vader_score"]
            .shift(1)
        )

        df["vader_lag_2"] = (
            df.groupby("ticker")["avg_vader_score"]
            .shift(2)
        )

    # -------------------------------
    # FINBERT LAG FEATURES
    # -------------------------------
    if "avg_finbert_confidence" in df.columns:

        df["finbert_lag_1"] = (
            df.groupby("ticker")["avg_finbert_confidence"]
            .shift(1)
        )

        df["finbert_lag_2"] = (
            df.groupby("ticker")["avg_finbert_confidence"]
            .shift(2)
        )

    return df


def validate_lag_alignment(lagged_df, sample_size=10):
    if lagged_df.empty:
        print("No lagged data available.")
        return

    print("\nLag Feature Validation:\n")

    sample = lagged_df.head(sample_size)

    columns_to_show = [
        "date",
        "ticker"
    ]

    optional_cols = [
        "avg_textblob_score",
        "textblob_lag_1",
        "textblob_lag_2",
        "avg_vader_score",
        "vader_lag_1",
        "vader_lag_2",
        "avg_finbert_confidence",
        "finbert_lag_1",
        "finbert_lag_2"
    ]

    for col in optional_cols:
        if col in sample.columns:
            columns_to_show.append(col)

    print(sample[columns_to_show])


def save_lagged_dataset(lagged_df, filename="lagged_sentiment_dataset.csv"):
    if lagged_df.empty:
        print("No lagged dataset available to save.")
        return

    lagged_df.to_csv(filename, index=False)

    print(f"\nLagged sentiment dataset saved as: {filename}")

def compute_rolling_averages(master_df):
    if master_df.empty:
        print("No master dataset available.")
        return pd.DataFrame()

    df = master_df.copy()

    df = df.sort_values(by=["ticker", "date"])

    # -------------------------------
    # TEXTBLOB ROLLING MEANS
    # -------------------------------
    if "avg_textblob_score" in df.columns:

        df["textblob_roll_3"] = (
            df.groupby("ticker")["avg_textblob_score"]
            .transform(lambda x: x.rolling(window=3).mean())
        )

        df["textblob_roll_5"] = (
            df.groupby("ticker")["avg_textblob_score"]
            .transform(lambda x: x.rolling(window=5).mean())
        )

    # -------------------------------
    # VADER ROLLING MEANS
    # -------------------------------
    if "avg_vader_score" in df.columns:

        df["vader_roll_3"] = (
            df.groupby("ticker")["avg_vader_score"]
            .transform(lambda x: x.rolling(window=3).mean())
        )

        df["vader_roll_5"] = (
            df.groupby("ticker")["avg_vader_score"]
            .transform(lambda x: x.rolling(window=5).mean())
        )

    # -------------------------------
    # FINBERT ROLLING MEANS
    # -------------------------------
    if "avg_finbert_confidence" in df.columns:

        df["finbert_roll_3"] = (
            df.groupby("ticker")["avg_finbert_confidence"]
            .transform(lambda x: x.rolling(window=3).mean())
        )

        df["finbert_roll_5"] = (
            df.groupby("ticker")["avg_finbert_confidence"]
            .transform(lambda x: x.rolling(window=5).mean())
        )

    # -------------------------------
    # STOCK RETURN ROLLING MEANS
    # -------------------------------
    if "daily_return" in df.columns:

        df["return_roll_3"] = (
            df.groupby("ticker")["daily_return"]
            .transform(lambda x: x.rolling(window=3).mean())
        )

        df["return_roll_5"] = (
            df.groupby("ticker")["daily_return"]
            .transform(lambda x: x.rolling(window=5).mean())
        )

    return df


def validate_rolling_averages(rolling_df, sample_size=10):
    if rolling_df.empty:
        print("No rolling average data available.")
        return

    print("\nRolling Average Validation:\n")

    sample = rolling_df.head(sample_size)

    columns_to_show = [
        "date",
        "ticker"
    ]

    optional_cols = [
        "textblob_roll_3",
        "textblob_roll_5",
        "vader_roll_3",
        "vader_roll_5",
        "finbert_roll_3",
        "finbert_roll_5",
        "return_roll_3",
        "return_roll_5"
    ]

    for col in optional_cols:
        if col in sample.columns:
            columns_to_show.append(col)

    print(sample[columns_to_show])


def save_rolling_average_dataset(
    rolling_df,
    filename="rolling_average_dataset.csv"
):
    if rolling_df.empty:
        print("No rolling average dataset available.")
        return

    rolling_df.to_csv(filename, index=False)

    print(f"\nRolling average dataset saved as: {filename}")