import pandas as pd
import matplotlib.pyplot as plt

# Read the data

df = pd.read_csv(
    '../data/raw/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

print(df.head())

# Create a time series plot using the timestamp as the index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Timestamp')
df = df.dropna()

# Plot the data
df['Close'].plot(figsize=(15, 7))
plt.title('Bitcoin Price')
plt.ylabel('Price (USD)')
plt.savefig('../images/bitcoin_price.png')
plt.show()
plt.close()
