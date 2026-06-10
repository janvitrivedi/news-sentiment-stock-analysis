import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------
# STEP 1 - MERGE NEWS + STOCK DATA
# -----------------------------------
def merge_news_stock():
    print("\nMerging news sentiment with stock prices...")

    news_df = pd.read_csv("kaggle_sentiment_scores.csv")
    stock_df = pd.read_csv("stock_prices_kaggle.csv")

    news_df["date"] = pd.to_datetime(
        news_df["date"], errors="coerce"
    ).dt.date.astype(str)

    stock_df["date"] = pd.to_datetime(
        stock_df["date"], errors="coerce"
    ).dt.date.astype(str)

    # Aggregate daily sentiment
    daily = news_df.groupby(["ticker", "date"]).agg(
        avg_textblob_score=("polarity_score", "mean"),
        avg_vader_score=("vader_score", "mean"),
        daily_article_count=("headline", "count"),
        positive_count=(
            "sentiment", lambda x: (x == "Positive").sum()
        ),
        negative_count=(
            "sentiment", lambda x: (x == "Negative").sum()
        ),
        neutral_count=(
            "sentiment", lambda x: (x == "Neutral").sum()
        ),
    ).reset_index()

    # Merge with stock prices
    merged = pd.merge(
        daily, stock_df,
        on=["ticker", "date"],
        how="inner"
    )

    # Sort
    merged = merged.sort_values(["ticker", "date"])

    # Lag features
    merged["textblob_lag_1"] = merged.groupby(
        "ticker"
    )["avg_textblob_score"].shift(1)

    merged["vader_lag_1"] = merged.groupby(
        "ticker"
    )["avg_vader_score"].shift(1)

    merged["textblob_lag_2"] = merged.groupby(
        "ticker"
    )["avg_textblob_score"].shift(2)

    merged["vader_lag_2"] = merged.groupby(
        "ticker"
    )["avg_vader_score"].shift(2)

    # Rolling averages
    merged["textblob_roll_5"] = merged.groupby(
        "ticker"
    )["avg_textblob_score"].transform(
        lambda x: x.rolling(5).mean()
    )

    merged["vader_roll_5"] = merged.groupby(
        "ticker"
    )["avg_vader_score"].transform(
        lambda x: x.rolling(5).mean()
    )

    # Price direction
    merged["price_direction"] = (
        merged["daily_return"] > 0
    ).astype(int)

    print(f"Merged dataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Tickers: {merged['ticker'].unique()}")

    merged.to_csv("merged_dataset.csv", index=False)
    print("Saved: merged_dataset.csv")

    return merged


# -----------------------------------
# STEP 2 - CORRELATION ANALYSIS
# -----------------------------------
def run_correlation_analysis():
    print("\nRunning correlation analysis...")

    df = pd.read_csv("merged_dataset.csv")
    results = []

    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].dropna(
            subset=["avg_textblob_score", "daily_return"]
        )

        if len(t) < 5:
            continue

        r_tb, p_tb = stats.pearsonr(
            t["avg_textblob_score"],
            t["daily_return"]
        )
        r_vd, p_vd = stats.pearsonr(
            t["avg_vader_score"],
            t["daily_return"]
        )

        results.append({
            "ticker": ticker,
            "textblob_correlation": round(r_tb, 4),
            "textblob_pvalue": round(p_tb, 4),
            "vader_correlation": round(r_vd, 4),
            "vader_pvalue": round(p_vd, 4),
            "sample_size": len(t),
            "significant": "Yes" if p_tb < 0.05 else "No"
        })

    corr_df = pd.DataFrame(results)
    corr_df.to_csv("correlation_results.csv", index=False)

    print("\nCorrelation Results:")
    print(corr_df.to_string())

    return corr_df


# -----------------------------------
# STEP 3 - LAG ANALYSIS
# -----------------------------------
def run_lag_analysis():
    print("\nRunning lag analysis...")

    df = pd.read_csv("merged_dataset.csv")
    results = []

    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].dropna(
            subset=["textblob_lag_1", "daily_return"]
        )

        if len(t) < 5:
            continue

        r_lag1, p_lag1 = stats.pearsonr(
            t["textblob_lag_1"],
            t["daily_return"]
        )

        t2 = df[df["ticker"] == ticker].dropna(
            subset=["textblob_lag_2", "daily_return"]
        )

        if len(t2) >= 5:
            r_lag2, p_lag2 = stats.pearsonr(
                t2["textblob_lag_2"],
                t2["daily_return"]
            )
        else:
            r_lag2, p_lag2 = 0, 1

        results.append({
            "ticker": ticker,
            "lag1_textblob_correlation": round(r_lag1, 4),
            "lag1_pvalue": round(p_lag1, 4),
            "lag2_textblob_correlation": round(r_lag2, 4),
            "lag2_pvalue": round(p_lag2, 4),
            "sample_size": len(t)
        })

    lag_df = pd.DataFrame(results)
    lag_df.to_csv("lag_analysis_results.csv", index=False)

    print("\nLag Analysis Results:")
    print(lag_df.to_string())

    return lag_df


# -----------------------------------
# STEP 4 - EVENT STUDY
# -----------------------------------
def run_event_study():
    print("\nRunning event study...")

    df = pd.read_csv("merged_dataset.csv")
    results = []

    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].copy()
        t = t.sort_values("date").reset_index(drop=True)

        for i in range(len(t) - 5):
            sentiment = t.loc[i, "avg_vader_score"]

            if abs(sentiment) < 0.1:
                continue

            label = "Positive" if sentiment > 0 else "Negative"

            for lag in range(6):
                if i + lag < len(t):
                    results.append({
                        "ticker": ticker,
                        "event_type": label,
                        "lag_day": lag,
                        "return": t.loc[
                            i + lag, "daily_return"
                        ]
                    })

    event_df = pd.DataFrame(results)

    summary = event_df.groupby(
        ["event_type", "lag_day"]
    )["return"].mean().reset_index()

    summary.columns = [
        "event_type", "lag_day", "avg_return"
    ]

    summary.to_csv("event_study_results.csv", index=False)

    print("Saved: event_study_results.csv")
    print(summary)

    return summary


# -----------------------------------
# STEP 5 - ANOMALY DETECTION
# -----------------------------------
def run_anomaly_detection():
    print("\nRunning anomaly detection...")

    df = pd.read_csv("merged_dataset.csv")
    df = df.dropna(
        subset=["avg_vader_score", "daily_return"]
    ).copy()

    df["sentiment_signal"] = df["avg_vader_score"].apply(
        lambda x: 1 if x > 0.05
        else -1 if x < -0.05
        else 0
    )

    df["return_signal"] = df["daily_return"].apply(
        lambda x: 1 if x > 0 else -1
    )

    df["is_anomaly"] = (
        (df["sentiment_signal"] != 0) &
        (df["sentiment_signal"] != df["return_signal"])
    )

    # Save FULL dataset with anomaly flag
    # not just anomaly rows
    df.to_csv("anomaly_results.csv", index=False)

    total = len(df)
    anomaly_count = len(df[df["is_anomaly"] == True])
    rate = round(anomaly_count / total * 100, 2)

    print(f"Total records: {total}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly rate: {rate}%")

    return df


# -----------------------------------
# STEP 6 - TICKER SENSITIVITY
# -----------------------------------
def run_ticker_sensitivity():
    print("\nRunning ticker sensitivity analysis...")

    df = pd.read_csv("merged_dataset.csv")
    results = []

    for ticker in df["ticker"].unique():
        t = df[df["ticker"] == ticker].dropna(
            subset=["avg_vader_score", "daily_return"]
        )

        if len(t) < 5:
            continue

        r, p = stats.pearsonr(
            t["avg_vader_score"],
            t["daily_return"]
        )

        results.append({
            "ticker": ticker,
            "sensitivity": round(abs(r), 4),
            "correlation": round(r, 4),
            "sample_size": len(t),
            "avg_daily_articles": round(
                t["daily_article_count"].mean(), 1
            )
        })

    sensitivity_df = pd.DataFrame(results).sort_values(
        "sensitivity", ascending=False
    )

    sensitivity_df.to_csv(
        "ticker_sensitivity.csv", index=False
    )

    print("\nTicker Sensitivity Ranking:")
    print(sensitivity_df.to_string())

    return sensitivity_df


# -----------------------------------
# STEP 7 - ML MODELS
# -----------------------------------
def run_ml_models():
    print("\nRunning ML models...")

    df = pd.read_csv("merged_dataset.csv")

    features = [
        "avg_textblob_score",
        "avg_vader_score",
        "daily_article_count",
        "textblob_lag_1",
        "vader_lag_1",
        "textblob_roll_5",
        "vader_roll_5",
        "positive_count",
        "negative_count"
    ]

    available = [
        f for f in features if f in df.columns
    ]

    df_model = df[
        available + ["price_direction", "date"]
    ].dropna()

    if len(df_model) < 20:
        print("Not enough data for ML models.")
        return

    # Train on pre-2020, test on 2020
    train = df_model[df_model["date"] < "2020-01-01"]
    test = df_model[df_model["date"] >= "2020-01-01"]

    X_train = train[available]
    y_train = train["price_direction"]
    X_test = test[available]
    y_test = test["price_direction"]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    results = []

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results.append({
        "model": "Logistic Regression",
        "accuracy": round(
            accuracy_score(y_test, y_pred_lr), 4
        ),
        "precision": round(
            precision_score(
                y_test, y_pred_lr, zero_division=0
            ), 4
        ),
        "recall": round(
            recall_score(
                y_test, y_pred_lr, zero_division=0
            ), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, y_prob_lr), 4
        ),
        "train_size": len(X_train),
        "test_size": len(X_test)
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    results.append({
        "model": "Random Forest",
        "accuracy": round(
            accuracy_score(y_test, y_pred_rf), 4
        ),
        "precision": round(
            precision_score(
                y_test, y_pred_rf, zero_division=0
            ), 4
        ),
        "recall": round(
            recall_score(
                y_test, y_pred_rf, zero_division=0
            ), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, y_prob_rf), 4
        ),
        "train_size": len(X_train),
        "test_size": len(X_test)
    })

    ml_df = pd.DataFrame(results)
    ml_df.to_csv("ml_results.csv", index=False)

    print("\nML Results:")
    print(ml_df.to_string())

    return ml_df

def run_2026_predictions():
    print("\nRunning 2026 predictions on NewsAPI data...")

    import joblib
    from sklearn.linear_model import LogisticRegression

    # Load merged training data
    df = pd.read_csv("merged_dataset.csv")
    features = [
        "avg_textblob_score", "avg_vader_score",
        "daily_article_count", "textblob_lag_1",
        "vader_lag_1", "textblob_roll_5", "vader_roll_5",
        "positive_count", "negative_count"
    ]
    avail = [f for f in features if f in df.columns]
    df_m = df[avail + ["price_direction", "date"]].dropna()

    # Train on full Kaggle data
    X_train = df_m[avail]
    y_train = df_m["price_direction"]
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)

    # Load recent NewsAPI sentiment
    recent = pd.read_csv("sentiment_scores.csv")
    recent["date"] = pd.to_datetime(
        recent["date"], utc=True, errors="coerce"
    ).dt.tz_localize(None).dt.date.astype(str)

    # Aggregate daily
    daily = recent.groupby(["ticker", "date"]).agg(
        avg_textblob_score=("polarity_score", "mean"),
        avg_vader_score=("vader_score", "mean"),
        daily_article_count=("headline", "count"),
        positive_count=(
            "sentiment", lambda x: (x == "Positive").sum()
        ),
        negative_count=(
            "sentiment", lambda x: (x == "Negative").sum()
        ),
    ).reset_index()

    # Add lag and rolling features
    daily = daily.sort_values(["ticker", "date"])
    daily["textblob_lag_1"] = daily.groupby(
        "ticker"
    )["avg_textblob_score"].shift(1)
    daily["vader_lag_1"] = daily.groupby(
        "ticker"
    )["avg_vader_score"].shift(1)
    daily["textblob_roll_5"] = daily.groupby(
        "ticker"
    )["avg_textblob_score"].transform(
        lambda x: x.rolling(5).mean()
    )
    daily["vader_roll_5"] = daily.groupby(
        "ticker"
    )["avg_vader_score"].transform(
        lambda x: x.rolling(5).mean()
    )

    # Fill missing with 0
    daily[avail] = daily[avail].fillna(0)

    # Predict
    X_pred = daily[avail]
    daily["prediction"] = lr.predict(X_pred)
    daily["confidence"] = lr.predict_proba(X_pred)[:, 1]
    daily["signal"] = daily["prediction"].map(
        {1: "UP", 0: "DOWN"}
    )

    daily.to_csv("predictions_2026.csv", index=False)
    print(f"Saved: predictions_2026.csv — {len(daily)} rows")
    print(daily[["ticker", "date", "signal", "confidence"]].head(20))

    return daily

# -----------------------------------
# SAVE FINAL DATASET
# -----------------------------------
def save_final_dataset():
    df = pd.read_csv("merged_dataset.csv")
    df.to_csv("final_dataset.csv", index=False)
    print("Saved: final_dataset.csv")