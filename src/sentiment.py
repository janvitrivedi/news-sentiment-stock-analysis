from textblob import TextBlob
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline


# -------------------------------
# GET POLARITY SCORE
# -------------------------------
def get_polarity_score(text):

    if pd.isna(text) or text == "":
        return 0

    analysis = TextBlob(text)

    return analysis.sentiment.polarity


# -------------------------------
# CLASSIFY SENTIMENT
# -------------------------------
def classify_sentiment(score):

    if score > 0:
        return "Positive"

    elif score < 0:
        return "Negative"

    else:
        return "Neutral"


# -------------------------------
# APPLY SENTIMENT ANALYSIS
# -------------------------------
def apply_textblob_sentiment(news_articles):

    news_df = pd.DataFrame(news_articles)

    if news_df.empty:
        print("No news data available.")
        return pd.DataFrame()

    # Apply polarity score
    news_df["polarity_score"] = news_df["headline"].apply(
        get_polarity_score
    )

    # Apply sentiment label
    news_df["sentiment"] = news_df["polarity_score"].apply(
        classify_sentiment
    )

    return news_df


# -------------------------------
# VALIDATE SAMPLE HEADLINES
# -------------------------------
def validate_sentiment_scores(news_df, sample_size=20):

    if news_df.empty:
        print("No data available for validation.")
        return

    print("\nSample Sentiment Validation:\n")

    sample = news_df.head(sample_size)

    for _, row in sample.iterrows():

        print("Headline:", row["headline"])
        print("Polarity Score:", row["polarity_score"])
        print("Sentiment:", row["sentiment"])
        print("-" * 50)


# -------------------------------
# SAVE DATASET
# -------------------------------
def save_sentiment_dataset(news_df, filename="sentiment_scores.csv"):

    if news_df.empty:
        print("No sentiment data available to save.")
        return

    news_df.to_csv(filename, index=False)

    print(f"\nSentiment dataset saved as: {filename}")

nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


def get_vader_score(text):
    if pd.isna(text) or text == "":
        return 0

    score = sia.polarity_scores(text)

    return score["compound"]


def classify_vader_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def apply_vader_sentiment(news_df):
    if news_df.empty:
        print("No news data available for VADER.")
        return pd.DataFrame()

    news_df = news_df.copy()

    news_df["vader_score"] = news_df["headline"].apply(get_vader_score)

    news_df["vader_sentiment"] = news_df["vader_score"].apply(
        classify_vader_sentiment
    )

    return news_df


def validate_vader_scores(news_df, sample_size=20):
    if news_df.empty:
        print("No VADER data available for validation.")
        return

    print("\nVADER Sentiment Validation:\n")

    sample = news_df.head(sample_size)

    for _, row in sample.iterrows():
        print("Headline:", row["headline"])
        print("VADER Score:", row["vader_score"])
        print("VADER Sentiment:", row["vader_sentiment"])
        print("-" * 50)


def save_vader_dataset(news_df, filename="vader_sentiment_scores.csv"):
    if news_df.empty:
        print("No VADER data available to save.")
        return

    news_df.to_csv(filename, index=False)

    print(f"\nVADER sentiment dataset saved as: {filename}")

finbert_pipeline = None

def load_finbert_model():
    global finbert_pipeline

    if finbert_pipeline is None:
        print("Loading FinBERT model...")

        finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

    return finbert_pipeline


def apply_finbert_sentiment(news_df, batch_size=8):
    if news_df.empty:
        print("No news data available for FinBERT.")
        return pd.DataFrame()

    news_df = news_df.copy()

    model = load_finbert_model()

    headlines = news_df["headline"].fillna("").tolist()

    results = model(headlines, batch_size=batch_size)

    news_df["finbert_sentiment"] = [item["label"] for item in results]
    news_df["finbert_confidence"] = [round(item["score"], 4) for item in results]

    return news_df


def save_finbert_dataset(news_df, filename="finbert_sentiment_scores.csv"):
    if news_df.empty:
        print("No FinBERT data available to save.")
        return

    news_df.to_csv(filename, index=False)

    print(f"\nFinBERT sentiment dataset saved as: {filename}")