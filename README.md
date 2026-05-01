
News Sentiment Analysis on NASDAQ-100 Stocks

Overview

This project analyzes how financial news sentiment impacts stock price movements for companies in the NASDAQ-100 index.
It uses real-time data from APIs to extract sentiment from news headlines and predict stock trends.

Features

Fetches NASDAQ-100 stock data using yfinance API
Retrieves financial news headlines using News API
Performs sentiment analysis using TextBlob
Predicts stock movement (up/down) based on sentiment
Interactive dashboard using Streamlit

Tech Stack

Python
TextBlob (NLP)
yfinance
News API
Streamlit

Project Structure

main.py → Runs full pipeline
data_api.py → Fetches NASDAQ-100 stock + news data
sentiment.py → Performs sentiment analysis
model.py → Prediction logic
app.py → Dashboard

Data Source

Stock Data: Yahoo Finance (NASDAQ-100 companies)
News Data: News API

How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Run the project
   python main.py

3. (Optional) Run dashboard
   streamlit run app.py

Future Improvements

Add machine learning models (Logistic Regression, Random Forest)
Add time-lag analysis
Use transformer-based NLP models
Improve prediction accuracy
