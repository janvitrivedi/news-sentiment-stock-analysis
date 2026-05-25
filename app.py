import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.data_api import TICKERS

st.set_page_config(
    page_title="NASDAQ-100 Sentiment Dashboard",
    layout="wide"
)

st.title("NASDAQ-100 News Sentiment Analysis Dashboard")

st.sidebar.header("Dashboard Controls")

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    TICKERS,
    default=TICKERS[:5]
)

# -------------------------------
# LOAD CSV FILES
# -------------------------------

def load_csv(file_name):
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    return pd.DataFrame()


daily_sentiment_df = load_csv("daily_sentiment_scores.csv")
finbert_df = load_csv("finbert_sentiment_scores.csv")
lagged_df = load_csv("lagged_sentiment_dataset.csv")
rolling_df = load_csv("rolling_average_dataset.csv")
sentiment_df = load_csv("sentiment_scores.csv")
vader_df = load_csv("vader_sentiment_scores.csv")

# -------------------------------
# OVERVIEW
# -------------------------------

st.header("Project Overview")

st.markdown("""
This dashboard analyzes NASDAQ-100 stock news sentiment using:

- TextBlob
- VADER
- FinBERT

The project includes:

- Sentiment Analysis
- Daily Aggregation
- Lag Features
- Rolling Average Features
- Feature Engineering
""")

# -------------------------------
# TEXTBLOB SENTIMENT
# -------------------------------

st.header("TextBlob Sentiment Analysis")

if not sentiment_df.empty:

    st.subheader("Sentiment Dataset")

    st.dataframe(sentiment_df.head(100))

    if "sentiment" in sentiment_df.columns:

        positive = (
            sentiment_df["sentiment"] == "Positive"
        ).sum()

        neutral = (
            sentiment_df["sentiment"] == "Neutral"
        ).sum()

        negative = (
            sentiment_df["sentiment"] == "Negative"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", positive)
        col2.metric("Neutral", neutral)
        col3.metric("Negative", negative)

    if "polarity_score" in sentiment_df.columns:

        st.subheader("TextBlob Polarity Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            sentiment_df["polarity_score"].dropna(),
            bins=20
        )

        ax.set_xlabel("Polarity Score")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

else:
    st.warning("sentiment_scores.csv not found.")

# -------------------------------
# VADER SENTIMENT
# -------------------------------

st.header("VADER Sentiment Analysis")

if not vader_df.empty:

    st.dataframe(vader_df.head(100))

    if "vader_score" in vader_df.columns:

        st.subheader("VADER Score Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            vader_df["vader_score"].dropna(),
            bins=20
        )

        ax.set_xlabel("VADER Score")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

else:
    st.warning("vader_sentiment_scores.csv not found.")

# -------------------------------
# FINBERT SENTIMENT
# -------------------------------

st.header("FinBERT Sentiment Analysis")

if not finbert_df.empty:

    st.dataframe(finbert_df.head(100))

    if "finbert_sentiment" in finbert_df.columns:

        finbert_counts = (
            finbert_df["finbert_sentiment"]
            .value_counts()
        )

        st.subheader("FinBERT Sentiment Counts")

        st.bar_chart(finbert_counts)

else:
    st.warning("finbert_sentiment_scores.csv not found.")

# -------------------------------
# DAILY AGGREGATED SENTIMENT
# -------------------------------

st.header("Daily Aggregated Sentiment")

if not daily_sentiment_df.empty:

    st.dataframe(daily_sentiment_df.head(100))

    ticker_option = st.selectbox(
        "Select Ticker for Daily Analysis",
        daily_sentiment_df["ticker"].unique()
    )

    ticker_data = daily_sentiment_df[
        daily_sentiment_df["ticker"] == ticker_option
    ]

    if "avg_textblob_score" in ticker_data.columns:

        st.subheader("Average TextBlob Sentiment")

        chart_data = ticker_data[
            ["date", "avg_textblob_score"]
        ].set_index("date")

        st.line_chart(chart_data)

    if "daily_article_count" in ticker_data.columns:

        st.subheader("Daily Article Count")

        chart_data = ticker_data[
            ["date", "daily_article_count"]
        ].set_index("date")

        st.bar_chart(chart_data)

else:
    st.warning("daily_sentiment_scores.csv not found.")

# -------------------------------
# LAG FEATURES
# -------------------------------

st.header("Lag Features")

if not lagged_df.empty:

    st.dataframe(lagged_df.head(100))

    st.subheader("Lag Feature Columns")

    lag_columns = [
        col for col in lagged_df.columns
        if "lag" in col.lower()
    ]

    st.write(lag_columns)

else:
    st.warning("lagged_sentiment_dataset.csv not found.")

# -------------------------------
# ROLLING AVERAGE FEATURES
# -------------------------------

st.header("Rolling Average Features")

if not rolling_df.empty:

    st.dataframe(rolling_df.head(100))

    rolling_columns = [
        col for col in rolling_df.columns
        if "roll" in col.lower()
    ]

    selected_roll = st.selectbox(
        "Select Rolling Feature",
        rolling_columns
    )

    if "ticker" in rolling_df.columns:

        ticker_roll = st.selectbox(
            "Select Ticker",
            rolling_df["ticker"].unique()
        )

        temp = rolling_df[
            rolling_df["ticker"] == ticker_roll
        ]

        if "date" in temp.columns:

            chart_data = temp[
                ["date", selected_roll]
            ].set_index("date")

            st.line_chart(chart_data)

else:
    st.warning("rolling_average_dataset.csv not found.")

# -------------------------------
# DOWNLOAD DATASETS
# -------------------------------

st.header("Download CSV Files")

csv_files = [
    "daily_sentiment_scores.csv",
    "finbert_sentiment_scores.csv",
    "lagged_sentiment_dataset.csv",
    "rolling_average_dataset.csv",
    "sentiment_scores.csv",
    "vader_sentiment_scores.csv"
]

for file in csv_files:

    if os.path.exists(file):

        with open(file, "rb") as f:

            st.download_button(
                label=f"Download {file}",
                data=f,
                file_name=file,
                mime="text/csv"
            )

# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")

st.markdown(
    "NASDAQ-100 News Sentiment Analysis Dashboard using Streamlit"
)