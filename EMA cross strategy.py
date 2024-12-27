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

# 若最後一天還有倉位，結算平倉
if position != 0:
    final_price = df['Price'].iloc[-1]
    final_date = df.index[-1].strftime('%Y-%m-%d')
    if position == 1:
        # 平多
        pnl = (final_price - entry_price) * contract_size
        trade_records.append({
            'Date': final_date,
            'Action': 'Close Long (Final)',
            'Price': final_price,
            'PnL': pnl
        })
    elif position == -1:
        # 平空
        pnl = (entry_price - final_price) * contract_size
        trade_records.append({
            'Date': final_date,
            'Action': 'Close Short (Final)',
            'Price': final_price,
            'PnL': pnl
        })

# --------------------------
# 5. 計算策略最終資產價值 vs. Buy & Hold
# --------------------------
# (A) 策略回測：根據每筆交易的PnL累加
total_pnl = sum([rec['PnL'] for rec in trade_records])
final_strategy_value = initial_capital + total_pnl
strategy_return_percent = (final_strategy_value - initial_capital) / initial_capital * 100

# (B) Buy & Hold：假設在2020-01-01用全部CNY買美元後持有到最後
first_price = df['Price'].iloc[0]
last_price = df['Price'].iloc[-1]
# 能買多少美元
buyhold_usd = initial_capital / first_price
# 結束時的資產價值 (回成CNY)
final_buyhold_value = buyhold_usd * last_price
buyhold_return_percent = (final_buyhold_value - initial_capital) / initial_capital * 100

# --------------------------
# 6. 繪圖 (A) 主圖 - 價格, MA, Bollinger; (B) MACD
# --------------------------
df.dropna(inplace=True)  # 確保繪圖不會因NA出錯

fig = plt.figure(figsize=(14, 8))

# (A) 主圖
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax1.set_title(f"USD/CNY Price (Daily) | {start_date} ~ {end_date}")
ax1.plot(df.index, df['Price'], label='Price', color='black')
ax1.plot(df.index, df['MA_short'], label=f'MA_{short_window}', color='blue', alpha=0.7)
ax1.plot(df.index, df['MA_long'], label=f'MA_{long_window}', color='orange', alpha=0.7)
ax1.plot(df.index, df['Boll_Upper'], label='Boll_Upper', color='green', linestyle='--', alpha=0.4)
ax1.plot(df.index, df['Boll_Lower'], label='Boll_Lower', color='green', linestyle='--', alpha=0.4)

# 在圖上標記交易點
for r in trade_records:
    trade_date = pd.to_datetime(r['Date'])
    trade_px = r['Price']
    action = r['Action']

    if 'Buy' in action and 'Close' not in action:
        # Buy 開多用紅色三角向上
        ax1.scatter(trade_date, trade_px, marker='^', color='red', s=100)
    elif 'Sell' in action and 'Close' not in action:
        # Sell 開空用綠色三角向下
        ax1.scatter(trade_date, trade_px, marker='v', color='green', s=100)
    elif 'Close' in action:
        # 平倉用紫色菱形
        ax1.scatter(trade_date, trade_px, marker='D', color='purple', s=80)

ax1.legend(loc='upper left')

# (B) MACD 圖
ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
ax2.set_title("MACD")
ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
ax2.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
ax2.bar(df.index, df['MACD_Hist'], label='MACD_Hist', color='gray', alpha=0.5)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

# --------------------------
# 7. 列印交易紀錄 & 回測結果
# --------------------------
print("=== 交易紀錄 ===")
for r in trade_records:
    print(f"{r['Date']} | {r['Action']:<15} | Price={r['Price']:.4f} | 單次盈虧={r['PnL']:.2f}")

print("\n=== 回測結果 ===")
print(f"最終策略資產價值: {final_strategy_value:,.2f}")
print(f"策略報酬率: {strategy_return_percent:.2f} %")
print(f"Buy & Hold 資產價值: {final_buyhold_value:,.2f}")
print(f"Buy & Hold 報酬率: {buyhold_return_percent:.2f} %")
