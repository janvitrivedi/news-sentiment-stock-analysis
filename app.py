import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="News Impact Analyzer",
    page_icon="📈",
    layout="wide"
)


# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #111827;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #111827 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="input"] * {
    color: #111827 !important;
}

h1, h2, h3, h4 {
    color: white !important;
}

p, label, div {
    color: #d1d5db;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background: #1f2937;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #374151;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}

.metric-title {
    font-size: 15px;
    color: #9ca3af;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: white;
}

.metric-sub {
    color: #22c55e;
    font-size: 15px;
}

.news-card {
    background: #1f2937;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    border: 1px solid #374151;
}

.sentiment-positive {
    background: #dcfce7;
    color: #14532d;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
}

.sentiment-negative {
    background: #fee2e2;
    color: #7f1d1d;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
}

.sentiment-neutral {
    background: #fef9c3;
    color: #713f12;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
}

.stDownloadButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}

.stDownloadButton button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD DATA
# -----------------------------------
rolling_df = pd.read_csv("rolling_average_dataset.csv")
finbert_df = pd.read_csv("finbert_sentiment_scores.csv")
sentiment_df = pd.read_csv("sentiment_scores.csv")

rolling_df["date"] = pd.to_datetime(rolling_df["date"], errors="coerce")

if "date" in sentiment_df.columns:
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce")


# -----------------------------------
# CLEAN DISPLAY NAMES
# -----------------------------------
display_names = {
    "ticker": "Stock",
    "date": "Date",
    "headline": "Headline",
    "source": "News Source",
    "polarity_score": "TextBlob Score",
    "sentiment": "TextBlob Sentiment",
    "avg_textblob_score": "TextBlob Sentiment Score",
    "avg_vader_score": "VADER Sentiment Score",
    "avg_finbert_confidence": "FinBERT Confidence",
    "daily_article_count": "Daily News Volume",
    "textblob_roll_3": "3-Day TextBlob Trend",
    "textblob_roll_5": "5-Day TextBlob Trend",
    "vader_roll_3": "3-Day VADER Trend",
    "vader_roll_5": "5-Day VADER Trend",
    "finbert_roll_3": "3-Day FinBERT Trend",
    "finbert_roll_5": "5-Day FinBERT Trend",
    "textblob_lag_1": "Previous Day TextBlob Impact",
    "textblob_lag_2": "Two-Day TextBlob Impact",
    "vader_lag_1": "Previous Day VADER Impact",
    "vader_lag_2": "Two-Day VADER Impact",
    "finbert_lag_1": "Previous Day FinBERT Impact",
    "finbert_lag_2": "Two-Day FinBERT Impact",
}


# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("News Impact Analyzer")
st.sidebar.caption("Interactive financial sentiment dashboard")

ticker_list = sorted(rolling_df["ticker"].dropna().unique())

selected_ticker = st.sidebar.selectbox(
    "Stock Ticker",
    ticker_list
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["7 Days", "30 Days", "90 Days", "All Available Data"]
)

sentiment_view = st.sidebar.selectbox(
    "Sentiment Indicator",
    [
        "TextBlob Sentiment Score",
        "VADER Sentiment Score",
        "FinBERT Confidence"
    ]
)

impact_threshold = st.sidebar.slider(
    "Impact Threshold",
    0.0,
    1.0,
    0.3
)

show_dataset = st.sidebar.checkbox(
    "Show Dataset",
    value=False
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: CSV sentiment datasets")
st.sidebar.caption("NLP: TextBlob, VADER, FinBERT")


# -----------------------------------
# FILTER DATA
# -----------------------------------
filtered_df = rolling_df[rolling_df["ticker"] == selected_ticker].copy()
filtered_df = filtered_df.sort_values("date")

if time_range == "7 Days":
    filtered_df = filtered_df.tail(7)
elif time_range == "30 Days":
    filtered_df = filtered_df.tail(30)
elif time_range == "90 Days":
    filtered_df = filtered_df.tail(90)

sentiment_feature_map = {
    "TextBlob Sentiment Score": "avg_textblob_score",
    "VADER Sentiment Score": "avg_vader_score",
    "FinBERT Confidence": "avg_finbert_confidence",
}

selected_sentiment_column = sentiment_feature_map[sentiment_view]

headline_df = sentiment_df[sentiment_df["ticker"] == selected_ticker].copy()

if "date" in headline_df.columns:
    headline_df = headline_df.sort_values("date", ascending=False)


# -----------------------------------
# KPI VALUES
# -----------------------------------
article_count = (
    int(filtered_df["daily_article_count"].sum())
    if "daily_article_count" in filtered_df.columns
    else len(filtered_df)
)

avg_textblob = filtered_df["avg_textblob_score"].mean()
avg_vader = filtered_df["avg_vader_score"].mean()
avg_finbert = filtered_df["avg_finbert_confidence"].mean()

overall_sentiment = (
    "Positive"
    if avg_textblob > 0
    else "Negative"
    if avg_textblob < 0
    else "Neutral"
)

positive_articles = (
    len(headline_df[headline_df["sentiment"] == "Positive"])
    if "sentiment" in headline_df.columns
    else 0
)

total_headlines = len(headline_df)
matched_ratio = f"{positive_articles} / {total_headlines}" if total_headlines else "0 / 0"


# -----------------------------------
# HEADER
# -----------------------------------
st.title(f"{selected_ticker} News Impact Analyzer")
st.caption(
    f"Analyzing financial news sentiment and stock-related signal patterns for {selected_ticker}."
)


# -----------------------------------
# KPI CARDS
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Articles Processed</div>
        <div class="metric-value">{article_count}</div>
        <div class="metric-sub">news records analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Overall Sentiment</div>
        <div class="metric-value">{overall_sentiment}</div>
        <div class="metric-sub">based on TextBlob score</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">VADER Score</div>
        <div class="metric-value">{round(avg_vader, 2)}</div>
        <div class="metric-sub">market emotion signal</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Positive Headlines</div>
        <div class="metric-value">{matched_ratio}</div>
        <div class="metric-sub">positive news detected</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------
# MAIN LAYOUT
# -----------------------------------
left, right = st.columns([2.2, 1])


# -----------------------------------
# LEFT PANEL
# -----------------------------------
with left:
    st.subheader("Sentiment Trend Over Time")

    fig_sentiment = px.line(
        filtered_df,
        x="date",
        y=selected_sentiment_column,
        markers=True,
        labels={
            "date": "Date",
            selected_sentiment_column: sentiment_view
        },
        title=sentiment_view
    )

    fig_sentiment.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=390,
        xaxis_title="Date",
        yaxis_title=sentiment_view
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.subheader("News Volume vs Sentiment Signal")

    fig_scatter = px.scatter(
        filtered_df,
        x="daily_article_count",
        y="avg_vader_score",
        size="daily_article_count",
        color="avg_vader_score",
        labels={
            "daily_article_count": "Daily News Volume",
            "avg_vader_score": "VADER Sentiment Score"
        },
        title="Relationship Between News Volume and Sentiment"
    )

    fig_scatter.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=390,
        xaxis_title="Daily News Volume",
        yaxis_title="VADER Sentiment Score"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Sentiment Intelligence")

    fig_roll = px.line(
        filtered_df,
        x="date",
        y="textblob_roll_3",
        markers=True,
        labels={
            "date": "Date",
            "textblob_roll_3": "3-Day TextBlob Trend"
        },
        title="3-Day TextBlob Trend"
    )

    fig_roll.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350,
        xaxis_title="Date",
        yaxis_title="3-Day TextBlob Trend"
    )

    st.plotly_chart(fig_roll, use_container_width=True)


# -----------------------------------
# RIGHT PANEL
# -----------------------------------
with right:
    st.subheader("News & Sentiment")

    if headline_df.empty:
        st.info("No headline-level sentiment data available.")
    else:
        display_news = headline_df.head(6)

        for _, row in display_news.iterrows():
            headline = row.get("headline", "No headline available")
            source = row.get("source", "Unknown Source")
            score = row.get("polarity_score", 0)
            sentiment = row.get("sentiment", "Neutral")

            if sentiment == "Positive":
                badge_class = "sentiment-positive"
            elif sentiment == "Negative":
                badge_class = "sentiment-negative"
            else:
                badge_class = "sentiment-neutral"

            st.markdown(f"""
            <div class="news-card">
                <h4>{headline}</h4>
                <p>{source}</p>
                <span class="{badge_class}">
                    {sentiment} {round(score, 2)}
                </span>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------
# MODEL COMPARISON
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Model Comparison Overview")

comparison_data = pd.DataFrame({
    "Model": ["TextBlob", "VADER", "FinBERT"],
    "Average Score": [
        avg_textblob,
        avg_vader,
        avg_finbert
    ]
})

fig_compare = px.bar(
    comparison_data,
    x="Model",
    y="Average Score",
    color="Model",
    title="Average Sentiment Scores Across NLP Models",
    labels={
        "Model": "Sentiment Model",
        "Average Score": "Average Sentiment Score"
    }
)

fig_compare.update_layout(
    paper_bgcolor="#1f2937",
    plot_bgcolor="#1f2937",
    font_color="white",
    height=390,
    showlegend=False
)

st.plotly_chart(fig_compare, use_container_width=True)


# -----------------------------------
# ADDITIONAL INSIGHT CHARTS
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Additional Market Sentiment Insights")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_news_bar = px.bar(
        filtered_df,
        x="date",
        y="daily_article_count",
        title="Daily News Volume",
        labels={
            "date": "Date",
            "daily_article_count": "Daily News Volume"
        }
    )

    fig_news_bar.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350,
        xaxis_title="Date",
        yaxis_title="Daily News Volume"
    )

    st.plotly_chart(fig_news_bar, use_container_width=True)

with chart_col2:
    fig_finbert = px.line(
        filtered_df,
        x="date",
        y="avg_finbert_confidence",
        markers=True,
        title="FinBERT Confidence Trend",
        labels={
            "date": "Date",
            "avg_finbert_confidence": "FinBERT Confidence"
        }
    )

    fig_finbert.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350,
        xaxis_title="Date",
        yaxis_title="FinBERT Confidence"
    )

    st.plotly_chart(fig_finbert, use_container_width=True)


chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_lag = px.line(
        filtered_df,
        x="date",
        y="textblob_lag_1",
        markers=True,
        title="Previous Day TextBlob Impact",
        labels={
            "date": "Date",
            "textblob_lag_1": "Previous Day TextBlob Impact"
        }
    )

    fig_lag.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350,
        xaxis_title="Date",
        yaxis_title="Previous Day TextBlob Impact"
    )

    st.plotly_chart(fig_lag, use_container_width=True)

with chart_col4:
    fig_vader_roll = px.line(
        filtered_df,
        x="date",
        y="vader_roll_3",
        markers=True,
        title="3-Day VADER Sentiment Trend",
        labels={
            "date": "Date",
            "vader_roll_3": "3-Day VADER Sentiment Trend"
        }
    )

    fig_vader_roll.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350,
        xaxis_title="Date",
        yaxis_title="3-Day VADER Sentiment Trend"
    )

    st.plotly_chart(fig_vader_roll, use_container_width=True)


# -----------------------------------
# CORRELATION MATRIX
# -----------------------------------
st.subheader("Sentiment Feature Correlation")

numeric_df = filtered_df.select_dtypes(include="number")

if not numeric_df.empty:
    corr = numeric_df.corr()
    corr = corr.rename(index=display_names, columns=display_names)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Sentiment Features"
    )

    fig_corr.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=600
    )

    st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------
# DATASET SECTION
# -----------------------------------
with st.expander("Dataset Explorer"):
    clean_df = filtered_df.rename(columns=display_names)

    if show_dataset:
        st.dataframe(clean_df, use_container_width=True)
    else:
        st.info("Enable 'Show Dataset' from the sidebar to view the dataset.")

    csv = clean_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Dataset",
        data=csv,
        file_name="filtered_news_impact_data.csv",
        mime="text/csv"
    )