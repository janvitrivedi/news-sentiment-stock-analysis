import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="News Impact Analyzer",
    page_icon="📈",
    layout="wide"
)

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

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: black !important;
}

h1, h2, h3, h4 {
    color: white !important;
}

p, label, div {
    color: #d1d5db;
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
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD DATA
# -----------------------------------
rolling_df = pd.read_csv("rolling_average_dataset.csv")
sentiment_df = pd.read_csv("sentiment_scores.csv")

rolling_df["date"] = pd.to_datetime(rolling_df["date"], errors="coerce")
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce")


# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("News Impact Analyzer")
st.sidebar.caption("Interactive financial sentiment dashboard")

ticker_list = sorted(rolling_df["ticker"].dropna().unique())

selected_ticker = st.sidebar.selectbox(
    "Stock Ticker",
    ticker_list,
    index=0
)

data_window = st.sidebar.radio(
    "Data Window",
    [
        "Last 1 Record",
        "Last 3 Records",
        "Last 5 Records",
        "All Records"
    ]
)

sentiment_view = st.sidebar.selectbox(
    "Sentiment Indicator",
    [
        "TextBlob Sentiment Score",
        "VADER Sentiment Score",
        "FinBERT Confidence"
    ],
    index=0
)

impact_threshold = st.sidebar.slider(
    "Impact Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

show_dataset = st.sidebar.checkbox("Show Dataset", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Data: CSV sentiment datasets")
st.sidebar.caption("NLP: TextBlob, VADER, FinBERT")


# -----------------------------------
# COLUMN MAPS
# -----------------------------------
sentiment_feature_map = {
    "TextBlob Sentiment Score": "avg_textblob_score",
    "VADER Sentiment Score": "avg_vader_score",
    "FinBERT Confidence": "avg_finbert_confidence",
}

selected_sentiment_column = sentiment_feature_map[sentiment_view]

display_names = {
    "ticker": "Stock",
    "date": "Date",
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
    "headline": "Headline",
    "source": "News Source",
    "polarity_score": "TextBlob Score",
    "sentiment": "Sentiment"
}


# -----------------------------------
# FILTER DATA
# -----------------------------------
ticker_df = rolling_df[
    rolling_df["ticker"] == selected_ticker
].copy()

ticker_df = ticker_df.sort_values("date")

filtered_df = ticker_df.copy()

if data_window == "Last 1 Record":
    filtered_df = filtered_df.tail(1)

elif data_window == "Last 3 Records":
    filtered_df = filtered_df.tail(3)

elif data_window == "Last 5 Records":
    filtered_df = filtered_df.tail(5)

elif data_window == "All Records":
    pass

if impact_threshold > 0 and selected_sentiment_column in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df[selected_sentiment_column].abs() >= impact_threshold
    ]

if not filtered_df.empty:
    st.sidebar.success(
        f"{filtered_df['date'].min().strftime('%Y-%m-%d')} to "
        f"{filtered_df['date'].max().strftime('%Y-%m-%d')}"
    )

st.sidebar.info(
    f"Rows for {selected_ticker}: {len(ticker_df)}\n\n"
    f"Rows after filter: {len(filtered_df)}"
)


headline_df = sentiment_df[
    sentiment_df["ticker"] == selected_ticker
].copy()

headline_df = headline_df.sort_values("date", ascending=False)


# -----------------------------------
# KPI VALUES
# -----------------------------------
article_count = (
    int(filtered_df["daily_article_count"].sum())
    if "daily_article_count" in filtered_df.columns and not filtered_df.empty
    else 0
)

avg_textblob = filtered_df["avg_textblob_score"].mean() if not filtered_df.empty else 0
avg_vader = filtered_df["avg_vader_score"].mean() if not filtered_df.empty else 0
avg_finbert = filtered_df["avg_finbert_confidence"].mean() if not filtered_df.empty else 0

overall_sentiment = (
    "Positive" if avg_textblob > 0 else
    "Negative" if avg_textblob < 0 else
    "Neutral"
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
    f"Data Window: {data_window} | Impact Threshold: {impact_threshold}"
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
        <div class="metric-sub">filtered news volume</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Overall Sentiment</div>
        <div class="metric-value">{overall_sentiment}</div>
        <div class="metric-sub">based on TextBlob</div>
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

with left:
    st.subheader("Sentiment Trend Over Time")

    fig_sentiment = px.line(
        filtered_df,
        x="date",
        y=selected_sentiment_column,
        markers=True,
        title=sentiment_view,
        labels={
            "date": "Date",
            selected_sentiment_column: sentiment_view
        }
    )

    fig_sentiment.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=390
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.subheader("News Volume vs Sentiment Signal")

    fig_scatter = px.scatter(
        filtered_df,
        x="daily_article_count",
        y="avg_vader_score",
        size="daily_article_count",
        color="avg_vader_score",
        title="Relationship Between News Volume and Sentiment",
        labels={
            "daily_article_count": "Daily News Volume",
            "avg_vader_score": "VADER Sentiment Score"
        }
    )

    fig_scatter.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=390
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Sentiment Intelligence")

    fig_roll = px.line(
        filtered_df,
        x="date",
        y="textblob_roll_3",
        markers=True,
        title="3-Day TextBlob Trend",
        labels={
            "date": "Date",
            "textblob_roll_3": "3-Day TextBlob Trend"
        }
    )

    fig_roll.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350
    )

    st.plotly_chart(fig_roll, use_container_width=True)


with right:
    st.subheader("News & Sentiment")

    if headline_df.empty:
        st.info("No headline-level sentiment data available.")
    else:
        for _, row in headline_df.head(6).iterrows():
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
# ADDITIONAL CHARTS
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Additional Market Sentiment Insights")

col5, col6 = st.columns(2)

with col5:
    fig_news = px.bar(
        filtered_df,
        x="date",
        y="daily_article_count",
        title="Daily News Volume",
        labels={
            "date": "Date",
            "daily_article_count": "Daily News Volume"
        }
    )

    fig_news.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=350
    )

    st.plotly_chart(fig_news, use_container_width=True)

with col6:
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
        height=350
    )

    st.plotly_chart(fig_finbert, use_container_width=True)


# -----------------------------------
# HEATMAP
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Sentiment Feature Heatmap")

heatmap_columns = [
    "avg_textblob_score",
    "avg_vader_score",
    "avg_finbert_confidence",
    "daily_article_count",
    "textblob_roll_3",
    "vader_roll_3",
    "textblob_lag_1"
]

available_heatmap_columns = [
    col for col in heatmap_columns
    if col in filtered_df.columns
]

if len(available_heatmap_columns) >= 2 and len(filtered_df) >= 2:
    heatmap_df = filtered_df[available_heatmap_columns].rename(
        columns=display_names
    )

    corr = heatmap_df.corr()

    fig_heatmap = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap of Sentiment Features",
        color_continuous_scale="RdBu"
    )

    fig_heatmap.update_layout(
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font_color="white",
        height=600
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("Select at least 2 records to generate heatmap.")


# -----------------------------------
# DATASET
# -----------------------------------
with st.expander("Dataset Explorer"):
    clean_df = filtered_df.rename(columns=display_names)

    if show_dataset:
        st.dataframe(clean_df, use_container_width=True)
    else:
        st.info("Enable Show Dataset from sidebar.")

    csv = clean_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Dataset",
        csv,
        "filtered_news_impact_data.csv",
        "text/csv"
    )