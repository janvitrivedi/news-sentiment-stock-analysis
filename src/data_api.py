import yfinance as yf
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ✅ Correct way to load API key
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


# -------------------------------
# FETCH STOCK DATA (2021–2026)
# -------------------------------
def fetch_stock_data(tickers):
    stock_data = {}

    for ticker in tickers:
        print(f"Fetching stock data for {ticker}...")

        try:
            data = yf.download(
                ticker,
                start="2021-04-01",
                end="2026-04-01",
                progress=False
            )
            stock_data[ticker] = data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return stock_data


# -------------------------------
# FETCH NEWS DATA
# -------------------------------
def fetch_news_data(ticker):
    print(f"Fetching news for {ticker}...")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 10   
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("Error fetching news:", e)
        return []

    articles = []

    for article in data.get("articles", []):
        articles.append({
            "ticker": ticker,
            "headline": article.get("title"),
            "source": article.get("source", {}).get("name"),
            "date": article.get("publishedAt")
        })

    return articles


# -------------------------------
# MAIN FUNCTION
# -------------------------------
if __name__ == "__main__":
    print("Starting data collection...\n")

    # Fetch stock data
    stock_data = fetch_stock_data(TICKERS)

    # Fetch news data
    all_news = []

    for ticker in TICKERS[:10]:  
        news = fetch_news_data(ticker)
        all_news.extend(news)

    print("\nSample News Output:")
    for item in all_news[:5]:
        print(item)

    print("\nData collection completed.")