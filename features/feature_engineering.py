import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to download and prepare data
def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df.drop(columns=['adj close'], inplace=True)
    return df

# Function to compute Bollinger Bands
def compute_bollinger_bands(series, window):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

# Function to calculate technical features
def calculate_features(df):
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    upper_band, lower_band = compute_bollinger_bands(df['close'], 21)
    df['bollinger_upper'] = upper_band
    df['bollinger_lower'] = lower_band
    df['lag7'] = df['close'].shift(7)
    df['volatility'] = df['close'].rolling(window=7).std()
    df.dropna(inplace=True)
    return df

# Function to fetch and prepare USD Index data
def fetch_usd_index_data(start_date, end_date):
    ticker = 'DX-Y.NYB'
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df.drop(columns=['adj close', 'volume', 'open', 'high', 'low'], inplace=True)
    return df

# Function to merge dataframes
def merge_dataframes(df1, df2, column_suffix):
    df_merged = pd.merge(df1, df2[['date', 'close']], on='date', how='left', suffixes=('', column_suffix))
    df_merged[f'close{column_suffix}'].fillna(method='ffill', inplace=True)
    return df_merged

# Function to fetch Bitcoin hash rate data
def fetch_hash_rate_data(url):
    df = pd.read_csv(url, header=None, names=['date', 'hash_rate'], index_col=0)
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Main function to execute the complete data preparation pipeline
def prepare_data(start_date, end_date):
    # Download and prepare Bitcoin data
    pd.options.display.float_format = '{:.4f}'.format
    symbol = 'BTC-USD'
    df_btc = download_data(symbol, start_date, end_date)
    df_btc = calculate_features(df_btc)

    # Fetch and merge USD Index data
    df_usd_index = fetch_usd_index_data(start_date, end_date)
    df_merged = merge_dataframes(df_btc, df_usd_index, '_usd_index')

    # Define and fetch oil and gold data
    oil_ticker = 'CL=F'
    gold_ticker = 'GC=F'
    df_oil = download_data(oil_ticker, start_date, end_date)
    df_gold = download_data(gold_ticker, start_date, end_date)

    # Merge oil and gold data
    df_merged = merge_dataframes(df_merged, df_oil, '_oil')
    df_merged = merge_dataframes(df_merged, df_gold, '_gold')

    # Fetch and merge hash rate data
    hash_rate_url = 'https://api.blockchain.info/charts/hash-rate?timespan=all&format=csv'
    df_hash_rate = fetch_hash_rate_data(hash_rate_url)
    df_merged = pd.merge(df_merged, df_hash_rate, on='date', how='left')
    df_merged['hash_rate'].fillna(method='ffill', inplace=True)

    # Drop rows with any remaining NaN values
    df_merged.dropna(inplace=True)
    df_merged['id'] = range(1, len(df_merged) + 1)

    return df_merged