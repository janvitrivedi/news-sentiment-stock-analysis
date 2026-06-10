import yfinance as yf
import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# -------------------------------
# NASDAQ-100 TICKERS
# -------------------------------
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","COST",
    "PEP","ADBE","CSCO","CMCSA","NFLX","INTC","AMD","TXN","QCOM","INTU",
    "HON","AMGN","SBUX","AMAT","ISRG","BKNG","GILD","ADI","LRCX","MDLZ",
    "MU","VRTX","REGN","KLAC","SNPS","PANW","MELI","CDNS","ASML","ADP",
    "ORLY","FTNT","CRWD","NXPI","CHTR","MNST","AEP","CTAS","MRVL","WDAY",
    "PCAR","ROST","ODFL","KDP","FAST","EXC","VRSK","XEL","IDXX","EA",
    "TEAM","CSGP","CPRT","ANSS","PAYX","BIIB","WBA","DLTR","ON","GEHC",
    "ZS","ILMN","MCHP","ABNB","LULU","FANG","SIRI","PYPL","LCID","RIVN",
    "DOCU","OKTA","JD","BIDU","PDD","NTES","VRSN","SWKS","TTWO","CTSH",
    "EBAY","ALGN","MTCH","CDW","SPLK","ZM","ENPH","MRNA","DDOG","NET"
]

KAGGLE_TICKERS = [
    'NVDA', 'MU', 'NFLX', 'EBAY', 'GILD', 'QCOM', 'ORCL', 'EA',
    'ADBE', 'BIIB', 'BIDU', 'TSLA', 'CMCSA', 'TXN', 'AVGO', 'GOOGL',
    'MRVL', 'REGN', 'DLTR', 'VRTX', 'PANW', 'ILMN', 'SIRI', 'JD',
    'GOOG', 'WBA', 'INTU', 'NXPI', 'CSCO', 'ADI', 'MNST', 'WDAY',
    'FTNT', 'CTSH', 'AMAT', 'SBUX', 'MCHP', 'ADP', 'LRCX', 'KLAC',
    'PAYX', 'ASML', 'NTES', 'ORLY', 'CTAS', 'ALGN', 'ENPH', 'FAST'
]

# -------------------------------
# FETCH STOCK DATA (DICT FORMAT)
# -------------------------------
def fetch_stock_data(tickers):
    stock_data = {}

    for ticker in tickers:
        print(f"Fetching stock data for {ticker}...")
        try:
            data = yf.download(
                ticker,
                start="2021-04-01",
                end="2026-05-31",
                progress=False
            )
            stock_data[ticker] = data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return stock_data


# -------------------------------
# FETCH STOCK DATA (CSV FORMAT)
# -------------------------------
def fetch_and_save_stock_data(
    tickers,
    filename="stock_prices.csv",
    start="2021-01-01",
    end="2026-05-31"
):
    all_data = []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False
            )
            df["ticker"] = ticker
            df = df.reset_index()
            df.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in df.columns
            ]
            df = df[[
                "Date", "ticker", "Close",
                "Open", "High", "Low", "Volume"
            ]]
            df.columns = [
                "date", "ticker", "close",
                "open", "high", "low", "volume"
            ]
            df["daily_return"] = df["close"].pct_change()
            all_data.append(df)

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if all_data:
        stock_df = pd.concat(all_data, ignore_index=True)
        stock_df.to_csv(filename, index=False)
        print(f"\nStock data saved: {stock_df.shape}")
        return stock_df

    return pd.DataFrame()


# -------------------------------
# FETCH NEWS DATA (NEWSAPI)
# -------------------------------
def fetch_news_data(ticker):
    print(f"Fetching news for {ticker}...")

    url = "https://newsapi.org/v2/everything"

    # Use company name for better results
    query_map = {
        "AAPL": "Apple stock",
        "MSFT": "Microsoft stock",
        "NVDA": "NVIDIA stock",
        "AMZN": "Amazon stock",
        "GOOGL": "Google Alphabet stock",
        "GOOG": "Google Alphabet stock",
        "META": "Meta Facebook stock",
        "TSLA": "Tesla stock",
        "AVGO": "Broadcom stock",
        "COST": "Costco stock",
    }

    query = query_map.get(ticker, f"{ticker} stock")

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("Error fetching news:", e)
        return []

    # Filter out irrelevant headlines
    exclude_keywords = [
        "720p", "1080p", "WEB-DL", "BluRay",
        "torrent", "download", "mkv", "x264",
        "HDTV", "BDRip", "DVDRip"
    ]

    articles = []
    for article in data.get("articles", []):
        headline = article.get("title", "") or ""

        # Skip if headline contains file/movie keywords
        if any(kw.lower() in headline.lower() for kw in exclude_keywords):
            continue

        articles.append({
            "ticker": ticker,
            "headline": headline,
            "source": article.get("source", {}).get("name"),
            "date": article.get("publishedAt")
        })

    return articles


# -------------------------------
# LOAD KAGGLE NEWS DATA
# -------------------------------
def load_kaggle_news(filename="raw_analyst_ratings.csv"):
    print("Loading Kaggle news data...")

    df = pd.read_csv(filename)

    df = df[df["stock"].isin(KAGGLE_TICKERS)].copy()

    df = df.rename(columns={
        "stock": "ticker",
        "publisher": "source"
    })

    # Fix date parsing — handles timezone offset format
    df["date"] = df["date"].str[:10]

    df = df[["ticker", "headline", "source", "date"]].dropna()

    print(f"Total headlines loaded: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df.to_dict("records")