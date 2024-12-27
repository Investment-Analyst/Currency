import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 下載/載入 USD/CNY (示例: 離岸人民幣 CNH=X)
# --------------------------
symbol = "CNY=X"  # 在 Yahoo Finance 代表離岸人民幣
start_date = "2020-01-01"
end_date = "2024-12-25"

df = yf.download(symbol, start=start_date, end=end_date)
df.dropna(inplace=True)
df['Price'] = df['Close']  # 收盤價

# --------------------------
# 2. 計算技術指標 (MA, Bollinger, MACD)
# --------------------------
short_window = 20
long_window = 50

# 移動平均 (MA)
df['MA_short'] = df['Price'].rolling(window=short_window).mean()
df['MA_long'] = df['Price'].rolling(window=long_window).mean()

# MACD
exp1 = df['Price'].ewm(span=12, adjust=False).mean()
exp2 = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# 布林通道 (以短均線為中心) - 20日為例
df['std'] = df['Price'].rolling(window=short_window).std()
df['Boll_Upper'] = df['MA_short'] + 2 * df['std']
df['Boll_Lower'] = df['MA_short'] - 2 * df['std']

# --------------------------
# 3. 建立交易信號 (MA Cross)
#    短均線 > 長均線 => 信號=1 (看多), 否則=-1 (看空)
# --------------------------
df['Signal'] = 0
df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1

# 觀察信號變化 (差分) 以捕捉交叉點
df['Crossover'] = df['Signal'].diff()

# --------------------------
# 4. 回測邏輯：多空策略 + 印交易紀錄
# --------------------------
position = 0  # 0=無倉, +1=多單, -1=空單
entry_price = None
contract_size = 200000  # 單筆交易規模 (例如10萬美元)
initial_capital = 1000000  # 假設初始資金 100萬

trade_records = []  # 紀錄每筆交易 (開倉、平倉)
daily_values = [initial_capital]  # 紀錄每日資產價值

for i in range(1, len(df)):
    date_i = df.index[i]
    price_i = df['Price'].iloc[i]
    crossover = df['Crossover'].iloc[i]  # 與上一根相比的變化
    signal_i = df['Signal'].iloc[i]

    # 若 crossover != 0, 代表短均線與長均線發生交叉 (或由0 -> 1 or -1)
    if crossover != 0:
        # --- Buy Signal ---
        if signal_i == 1:
            # 若手上有空單, 先平空
            if position == -1:
                # 空單損益 = (做空時的 entry_price - 現在的平倉價) * 合約數量
                pnl = (entry_price - price_i) * contract_size
                trade_records.append({
                    'Date': date_i.strftime('%Y-%m-%d'),
                    'Action': 'Close Short',
                    'Price': price_i,
                    'PnL': pnl
                })
            # 再開多
            position = 1
            entry_price = price_i
            # 開多時，PnL=0 (開倉動作本身無盈虧)
            trade_records.append({
                'Date': date_i.strftime('%Y-%m-%d'),
                'Action': 'Buy',
                'Price': price_i,
                'PnL': 0
            })

        # --- Sell Signal ---
        elif signal_i == -1:
            # 若手上有多單, 先平多
            if position == 1:
                pnl = (price_i - entry_price) * contract_size
                trade_records.append({
                    'Date': date_i.strftime('%Y-%m-%d'),
                    'Action': 'Close Long',
                    'Price': price_i,
                    'PnL': pnl
                })
            # 再開空
            position = -1
            entry_price = price_i
            # 開空時，PnL=0
            trade_records.append({
                'Date': date_i.strftime('%Y-%m-%d'),
                'Action': 'Sell',
                'Price': price_i,
                'PnL': 0
            })

    # 記錄每日資產價值 (假設當前持倉的未實現盈虧)
    daily_value = initial_capital + sum([rec['PnL'] for rec in trade_records])
    if position == 1:
        daily_value += (price_i - entry_price) * contract_size
    elif position == -1:
        daily_value += (entry_price - price_i) * contract_size
    daily_values.append(daily_value)

# 若最後一天還有倉位，結算平倉
if position != 0:
    final_price = df['Price'].iloc[-1]
    final_date = df.index[-1]
    if position == 1:
        pnl = (final_price - entry_price) * contract_size
        trade_records.append({'Date': final_date, 'Action': 'Close Long (Final)', 'Price': final_price, 'PnL': pnl})
    elif position == -1:
        pnl = (entry_price - final_price) * contract_size
        trade_records.append({'Date': final_date, 'Action': 'Close Short (Final)', 'Price': final_price, 'PnL': pnl})

# --------------------------
# 5. 計算策略績效指標
# --------------------------
total_pnl = sum([rec['PnL'] for rec in trade_records])
final_strategy_value = initial_capital + total_pnl
strategy_return_percent = (final_strategy_value - initial_capital) / initial_capital * 100

# 最大回撤
daily_values = np.array(daily_values)
drawdowns = (daily_values / np.maximum.accumulate(daily_values)) - 1
max_drawdown = drawdowns.min()

# 勝率
winning_trades = [rec for rec in trade_records if rec['PnL'] > 0]
win_rate = len(winning_trades) / len(trade_records) * 100 if trade_records else 0

# 風險報酬比 (假設年化收益率 / 年化波動率)
annualized_return = (1 + strategy_return_percent / 100) ** (1 / (len(df) / 252)) - 1
annualized_volatility = np.std(drawdowns) * np.sqrt(252)
risk_reward_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

# --------------------------
# 6. 列印回測結果
# --------------------------
print("=== 交易紀錄 ===")
for r in trade_records:
    print(f"{r['Date']} | {r['Action']:<15} | Price={r['Price']:.4f} | 單次盈虧={r['PnL']:.2f}")

print("\n=== 回測結果 ===")
print(f"最終策略資產價值: {final_strategy_value:,.2f}")
print(f"策略報酬率: {strategy_return_percent:.2f} %")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"勝率: {win_rate:.2f} %")
print(f"風險報酬比: {risk_reward_ratio:.2f}")
