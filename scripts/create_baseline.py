# Description: Create baseline for the dataset
# Model: Niave Forecasting

from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    '../data/processed/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
    index_col=0)


# split into train and test sets (%90-%10)
train_size = int(len(df) * 0.9)
train, test = df[0:train_size], df[train_size:len(df)]


# make a copy of test set
test_copy = test.copy()


# Naive forcast: use last value from training set to predict
last_value = train.iloc[-1]['Close']
naive_prediction = [last_value] * len(test_copy)


# Compare Predictions to Actual Values
test_copy.loc[:, 'naive_predictions'] = naive_prediction


mae = mean_absolute_error(test_copy['Close'], test_copy['naive_predictions'])
print('MAE: %f' % mae)


# Plotting the actual and predicted values
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
downsampled_test = test_copy.iloc[::10]

# Plotting the actual and predicted values with downsampling
plt.figure(figsize=(10, 6))
plt.plot(downsampled_test.index,
         downsampled_test['Close'], label='Actual Values', marker='o')
plt.plot(downsampled_test.index, downsampled_test['naive_predictions'],
         label='Naive Predictions', color='red', linestyle='--', marker='x')
plt.title('Naive Forecast vs Actual Data (Downsampled)')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.savefig('../images/NiaveForecastPrediction_Downsampled.png')
plt.show()
