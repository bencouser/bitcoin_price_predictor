import pandas as pd

# Read the data
df = pd.read_csv(
    '../data/raw/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Drop NA values
df = df.dropna()

# Create a new DataFrame with only 'Timestamp' and 'Close' price
bitcoin_prices = df[['Timestamp', 'Close']]

# Check the info of the new DataFrame
print(bitcoin_prices.info())


bitcoin_prices.to_csv(
    '../data/processed/Bitcoin_Closing_Prices.csv'
)

print('Done!')
