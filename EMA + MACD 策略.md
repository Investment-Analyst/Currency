美元/人民幣 匯率的技術分析 - EMA 交叉策略

交易信號(MA Cross)：

df['Signal'] = 0
df.loc[
    (df['MA_short'] > df['MA_long']) & (df['MACD'] > df['MACD_Signal']),
    'Signal'
] = 1  # Buy 信號

df.loc[
    (df['MA_short'] < df['MA_long']) & (df['MACD'] < df['MACD_Signal']),
    'Signal'
] = -1  # Sell 信號

假設初始資金100萬元，單筆交易規模30萬元




![image](https://github.com/user-attachments/assets/ee0c8376-2627-4661-9433-4afcb6c268b8)
