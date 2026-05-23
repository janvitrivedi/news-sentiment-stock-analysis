import pandas as pd


def calculate_daily_returns(stock_df):
    stock_df = stock_df.copy()

    if stock_df.empty:
        return stock_df

    stock_df["close"] = pd.to_numeric(stock_df["close"], errors="coerce")

    stock_df = stock_df.sort_values(["Ticker", "date"])

    stock_df["daily_return"] = (
        stock_df.groupby("Ticker")["close"]
        .pct_change(fill_method=None)
    )

    return stock_df


def detect_outliers(stock_df, threshold=3):
    stock_df = stock_df.copy()

    if stock_df.empty:
        return stock_df

    stock_df["mean_return"] = (
        stock_df.groupby("Ticker")["daily_return"]
        .transform("mean")
    )

    stock_df["std_return"] = (
        stock_df.groupby("Ticker")["daily_return"]
        .transform("std")
    )

    stock_df["z_score"] = (
        (stock_df["daily_return"] - stock_df["mean_return"])
        / stock_df["std_return"]
    )

    stock_df["is_outlier"] = stock_df["z_score"].abs() > threshold

    return stock_df