# Description: Create model that uses grad of last two elements
# Model: One layer deep model

from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


def one_deep_model(dataframe):
    """
    One Deep Model.
    Using the last two values calculate the gradient
    and make prediction using that.
    """
    # make empty dataframe of length of the dataframe
    predictions = pd.DataFrame(index=dataframe.index,
                               columns=['Close'])

    predictions['Close'].iloc[0] = dataframe['Close'].iloc[0]

    for i in range(1, len(dataframe)):
        # Calculate the gradient
        gradient = dataframe['Close'].iloc[i] - \
            dataframe['Close'].iloc[i - 1]

        # Make the prediction
        predictions['Close'].iloc[i] = dataframe['Close'].iloc[i - 1] + gradient

    return pd.DataFrame(predictions)


df = pd.read_csv(
    '../data/processed/Bitcoin_Closing_Prices.csv',
    index_col=0)


# split into train and test sets (%90-%10)
train_size = int(len(df) * 0.9)
train, test = df[0:train_size], df[train_size:len(df)]


predictions = one_deep_model(test)

print(predictions.head())


mae = mean_absolute_error(test['Close'], predictions['Close'])
print('MAE: %f' % mae)


# Downsampling for plotting: taking every 10th value
downsampled_predictions = predictions.iloc[::10]
downsampled_data = test.iloc[::10]

# Plotting the actual and predicted values with downsampling
plt.figure(figsize=(10, 6))
plt.plot(downsampled_data.index,
         downsampled_data['Close'],
         label='Actual Values',
         marker='o',
         markersize=1)
plt.plot(downsampled_predictions.index,
         downsampled_predictions['Close'],
         label='Naive Predictions',
         color='red',
         linestyle='--',
         marker='x',
         markersize=1)
plt.title('Naive Forecast vs Actual Data (Downsampled)')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.savefig('../images/one_deep_predictions.png')
plt.show()
