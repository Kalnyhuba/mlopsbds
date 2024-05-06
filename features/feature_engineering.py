import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

def process_bitcoin_data(start, end) -> pd.DataFrame:
    symbol = 'BTC-USD'
    df = yf.download(symbol, start=start, end=end)

    # Lowercase column names
    df.columns = df.columns.str.lower()

    df.drop(columns=['adj close'], inplace=True)

    df['date'] = df.index.strftime("%Y-%m-%d")

    # Calculate timestamp 
    df["timestamp"] = df["date"].apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp() * 1000))

    df.reset_index(drop=True, inplace=True)

    df['tomorrow'] = df['close'].shift(-1)

    #df.dropna(inplace=True)

    # Pivot DataFrame
    btc = df.pivot_table(index=['timestamp', 'date'], columns=None, values=None).reset_index()
    btc = btc[['timestamp', 'date', 'open', 'high', 'low', 'volume', 'close', 'tomorrow']]

    return btc
