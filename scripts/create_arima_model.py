# Description: Create an ARIMA model
# Model: ARIMA

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

print("0")

df = pd.read_csv(
    '../data/processed/Bitcoin_Closing_Prices.csv',
    index_col=0)

print("1")

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
df.index = pd.DatetimeIndex(df.index).to_period(
    'S')  # Set frequency to seconds

# split data to 10% of data

train, test = train_test_split(df, test_size=0.1, shuffle=False)

print("2")

# Split data into train and test sets
model = ARIMA(test,
              order=(1, 0, 0),
              seasonal_order=(0, 1, 1, 7))

print("3")

model_fit = model.fit()

# print(model_fit.summary())

print("4")


forecast = model_fit.forecast(steps=len(test))

print(len(forecast))


# plt.figure(figsize=(12, 8))
# plt.plot(df['Close'], label='Actual')
# plt.plot(forecast, label='Forecast', color='red')
# plt.legend()
# plt.show()
