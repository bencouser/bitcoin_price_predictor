import pandas as pd

# Read the data
df = pd.read_csv(
    '../data/raw/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# Create a time series plot using the timestamp as the index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Timestamp')
df = df.dropna()

df.to_csv(
    '../data/processed/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'
)
