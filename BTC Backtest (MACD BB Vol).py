from hyperliquid.info import Info
from datetime import datetime
import pandas as pd
import numpy as np

############################################################ Getting data from HyperLiquid ########################################
# Set up the connection
info = Info(base_url="https://api.hyperliquid.xyz")

# Current date
current_date = datetime.now()

# List of tickers
tickers = ['BTC']
ohlcv_data = {}


# Get candles for each ticker
for ticker in tickers:
    start_date = datetime(2025, 1, 1).timestamp() * 1000
    end_date = current_date.timestamp() * 1000
    ohlcv_data[ticker] = info.candles_snapshot(ticker, '5m', int(start_date), int(end_date))

ticker_data = {}
for ticker in tickers:
    df = pd.DataFrame(ohlcv_data[ticker], columns=['T', 'c', 'h', 'i', 'l', 'n', 'o', 's', 't', 'v'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    ticker_data[ticker] = df
    
    # Rename columns for clarity
    df.columns = ['close_timestamp', 'close', 'high', 'interval', 'low', 'number_of_trades', 'open', 'symbol', 'volume']
    
    # Convert string values to appropriate types
    df[['close', 'high', 'low', 'open', 'volume']] = df[['close', 'high', 'low', 'open', 'volume']].astype(float)
    df['number_of_trades'] = df['number_of_trades'].astype(int)
    df['close_timestamp'] = pd.to_datetime(df['close_timestamp'], unit='ms')

################################################################### Listing Indicators #################################################

def MACD(DF, a=12 ,b=26, c=9):
    """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
    df = DF.copy()
    df["ma_fast"] = df["close"].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df["close"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    return df.loc[:,["macd","signal"]]

def Boll_Band(DF, n=20):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MB"] = df["close"].rolling(n).mean()
    df["UB"] = df["MB"] + 2*df["close"].rolling(n).std(ddof=0)
    df["LB"] = df["MB"] - 2*df["close"].rolling(n).std(ddof=0)
    df["BB_Width"] = df["UB"] - df["LB"]
    return df[["MB","UB","LB","BB_Width"]]

def RSI(DF, n=14):
    df = DF.copy()
    df["change"] = df["close"] - df["close"].shift(1)
    df["gain"] = np.where(df["change"]>=0,df["change"],0)
    df["loss"] = np.where(df["change"]<0,-1*df["change"],0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/(1+df["rs"]))
    return df["rsi"]

################################################################### Performance Metrix ###################################################

def CAGR(DF, initial_value=1000):
    df = DF.copy()
    n = len(df) / (252 * 78)  # 252 trading days, 78 5-min candles per day
    total_return = df['cum_ret'].iloc[-1]
    CAGR = (((initial_value + total_return) / initial_value) ** (1/n)) - 1
    return CAGR

def volatility(DF, initial_value=1000):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    # Convert dollar returns to percentage returns
    pct_returns = df["strategy_ret"] / initial_value
    # Calculate annualized volatility
    vol = pct_returns.std() * np.sqrt(252 * 78)
    return vol

def sharpe(DF, rf, initial_value=1000):
    df = DF.copy()
    pct_returns = df['strategy_ret'] / initial_value
    excess_returns = pct_returns - rf / (252 * 78)
    sharpe_ratio = np.sqrt(252 * 78) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def max_dd(DF, initial_value=1000):
    df = DF.copy()
    df["cum_return"] = df["strategy_ret"].cumsum()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / initial_value
    max_dd = df["drawdown_pct"].max()
    return max_dd

############################################################### Backtest begins ############################################################
# Adding indicators into DF
ohlc_dict = {ticker: df.copy() for ticker in tickers}
tickers_signal = {ticker: "" for ticker in tickers}
tickers_ret = {ticker: [] for ticker in tickers}

entry = {ticker: 0 for ticker in tickers}
stop_loss = {ticker: 0 for ticker in tickers} #2%
take_profit = {ticker: 0 for ticker in tickers} #4%

for ticker in tickers:
    ohlc_dict[ticker][["MB","UB","LB","BB_Width"]] = Boll_Band(ohlc_dict[ticker])
    
for ticker in tickers:
    ohlc_dict[ticker][["MACD","SIGNAL"]] = MACD(ohlc_dict[ticker], a=12 ,b=26, c=9)  
    
for ticker in tickers:
    ohlc_dict[ticker]["RSI"] = RSI(ohlc_dict[ticker])

#entries
for ticker in tickers:
    # Use .loc for creating the 'position' column
    ohlc_dict[ticker].loc[:, 'position'] = ""
    tickers_ret[ticker] = [0]
    for i in range(1, len(ohlc_dict[ticker])):
        # Use .iloc for accessing values by integer position
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if (ohlc_dict[ticker]["close"].iloc[i-1] > ohlc_dict[ticker]["close"].iloc[i] and
                ohlc_dict[ticker]["close"].iloc[i-1] > ohlc_dict[ticker]["UB"].iloc[i] and
                ohlc_dict[ticker]["MACD"].iloc[i] < 0 and
                ohlc_dict[ticker]["volume"].iloc[i] > ohlc_dict[ticker]["volume"].iloc[i-1]):
                tickers_signal[ticker] = "Sell"
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "Sell"
                entry[ticker] = ohlc_dict[ticker]["close"].iloc[i]
                stop_loss[ticker] = ohlc_dict[ticker]["close"].iloc[i] * 1.02
                take_profit[ticker] = ohlc_dict[ticker]["close"].iloc[i] * 0.96
                
            elif (ohlc_dict[ticker]["close"].iloc[i-1] < ohlc_dict[ticker]["close"].iloc[i] and
                  ohlc_dict[ticker]["close"].iloc[i-1] < ohlc_dict[ticker]["LB"].iloc[i] and
                  ohlc_dict[ticker]["MACD"].iloc[i] > 0 and
                  ohlc_dict[ticker]["volume"].iloc[i] > ohlc_dict[ticker]["volume"].iloc[i-1]):
                tickers_signal[ticker] = "Buy"
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "Buy"
                entry[ticker] = ohlc_dict[ticker]["close"].iloc[i]
                stop_loss[ticker] = ohlc_dict[ticker]["close"].iloc[i] * 0.98
                take_profit[ticker] = ohlc_dict[ticker]["close"].iloc[i] * 1.04
     
        elif tickers_signal[ticker] == "Sell":
            if ohlc_dict[ticker]["high"].iloc[i] >= stop_loss[ticker]:
                tickers_signal[ticker] = ""
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "SL"
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["high"].iloc[i])
                
            elif ohlc_dict[ticker]["low"].iloc[i] <= take_profit[ticker]:
                tickers_signal[ticker] = ""
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "TP"
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["low"].iloc[i])
            else:
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["close"].iloc[i])
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = ""
                
        elif tickers_signal[ticker] == "Buy":
            if ohlc_dict[ticker]["low"].iloc[i] <= stop_loss[ticker]:  # Use low price for stop loss
                tickers_signal[ticker] = ""
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "SL"
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["low"].iloc[i])
                
            elif ohlc_dict[ticker]["high"].iloc[i] >= take_profit[ticker]:  # Use high price for take profit
                tickers_signal[ticker] = ""
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = "TP"
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["high"].iloc[i])
                
            else:
                tickers_ret[ticker].append(ohlc_dict[ticker]["close"].iloc[i-1] - ohlc_dict[ticker]["close"].iloc[i])
                ohlc_dict[ticker].loc[ohlc_dict[ticker].index[i], 'position'] = ""
                
    ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])
    
for ticker in tickers:
    ohlc_dict[ticker]['cum_ret'] = (ohlc_dict[ticker]['ret']).cumsum()

strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_dict[ticker]["ret"]
strategy_df['strategy_ret'] = strategy_df.sum(axis=1)  
strategy_df['cum_ret'] = strategy_df['strategy_ret'].cumsum()
total_return = strategy_df['cum_ret'].iloc[-1]
initial_value = 1000  
total_return_pct = (total_return / initial_value) * 100
CAGR(strategy_df)
vol = volatility(strategy_df)
sharpe(strategy_df,0.025)
max_dd(strategy_df)  

# vizualization of strategy return
(1+strategy_df["strategy_ret"]).cumsum().plot()


#calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
max_drawdown = {}
for ticker in tickers:
    print("calculating KPIs for ",ticker)      
    cagr[ticker] =  CAGR(ohlc_dict[ticker])
    
    # Create a temporary DataFrame for the single ticker
    temp_df = pd.DataFrame(ohlc_dict[ticker]["ret"])
    temp_df['strategy_ret'] = temp_df.sum(axis=1) #set the ret
    sharpe_ratios[ticker] =  sharpe(temp_df,0.025)
    
    # Create a temporary DataFrame for the single ticker
    temp_df2 = pd.DataFrame(ohlc_dict[ticker]["ret"])
    temp_df2['strategy_ret'] = temp_df2.sum(axis=1) #set the ret
    max_drawdown[ticker] =  max_dd(temp_df2)

KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
KPI_df.T

print(total_return_pct)