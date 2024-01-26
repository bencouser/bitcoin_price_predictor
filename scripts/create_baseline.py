# Description: Create baseline for the dataset
# Model: Niave Forecasting

from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


def naive_forecast(dataframe):
    """
    Naive Forecasting.
    Shift only the 'Close' column by one position downwards,
    duplicating the first value of the 'Close' column.
    All other column values remain unchanged.
    The last row is dropped after the shift.
    """
    predictions = dataframe.copy()

    # Save the first value of the 'Close' column
    first_close_value = dataframe['Close'].iloc[0]

    # Shift the 'Close' column by one position downwards
    predictions['Close'] = dataframe['Close'].shift(1)

    # Replace the first value of the shifted 'Close' column
    predictions['Close'].iloc[0] = first_close_value

    return pd.DataFrame(predictions)


df = pd.read_csv(
    '../data/processed/Bitcoin_Closing_Prices.csv',
    index_col=0)


# split into train and test sets (%90-%10)
train_size = int(len(df) * 0.9)
train, test = df[0:train_size], df[train_size:len(df)]


predictions = naive_forecast(test)

print("Real Values")
print(test.head())
print("Naive Predictions")
print(predictions.head())

print(test)
print(predictions)


mae = mean_absolute_error(test['Close'], predictions['Close'])
print('MAE: %f' % mae)

# # Plotting the actual and predicted values
# plt.figure(figsize=(10, 6))
# plt.plot(test.index, test['Close'], label='Actual Values')
# plt.plot(test_copy.index, test_copy['naive_predictions'],
#          label='Naive Predictions', color='red', linestyle='--')
# plt.title('Naive Forecast vs Actual Data')
# plt.xlabel('Time')
# plt.ylabel('Bitcoin Price')
# plt.legend()
# plt.savefig('../images/NiaveForecastPrediction(downsampled).png')
# plt.show()

# Downsampling for plotting: taking every 10th value
# downsampled_test = test_copy.iloc[::10]
#
# # Plotting the actual and predicted values with downsampling
# plt.figure(figsize=(10, 6))
# plt.plot(downsampled_test.index,
#          downsampled_test['Close'],
#          label='Actual Values',
#          marker='o')
# plt.plot(downsampled_test.index,
#          downsampled_test['naive_predictions'],
#          label='Naive Predictions',
#          color='red',
#          linestyle='--',
#          marker='x')
# plt.title('Naive Forecast vs Actual Data (Downsampled)')
# plt.xlabel('Time')
# plt.ylabel('Bitcoin Price')
# plt.legend()
# plt.savefig('../images/NiaveForecastPrediction_Downsampled.png')
# plt.show()
