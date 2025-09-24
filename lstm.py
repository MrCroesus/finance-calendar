import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

company = "Microsoft"
tick = "MSFT"
ticker = yf.Ticker(tick)
tickers = yf.Tickers([tick])

# get historical market data
history = ticker.history(period="5y")
history.index = history.index.date
print(history.keys())
print(history.index)

stock_data = history



# Assuming 'stock_data' has a 'Date' column or is using Date as the index
plt.figure(figsize=(10, 6))
plt.plot(stock_data['High'], label='High', linewidth=2)
plt.plot(stock_data['Low'], label='Low', linewidth=1)

# Adding labels and title
plt.title('Stock High and Low Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Rotating the x-axis labels to avoid overlap
plt.xticks(rotation=45)

# Optionally, limit the number of x-ticks for better readability
plt.xticks(stock_data.index[::len(stock_data)//10])  # Adjust the step size based on your data size

# Show the plot
plt.show()



# Assuming 'stock_data' has a 'Date' column or is using Date as the index
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Open'], label='Open', color='red', linewidth=2)
plt.plot(stock_data['Close'], label='Close', color='green', linewidth=1)

# Adding labels and title
plt.title('Stock Open and Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Rotating the x-axis labels to avoid overlap
plt.xticks(rotation=45)

# Optionally, limit the number of x-ticks for better readability
plt.xticks(stock_data.index[::len(stock_data)//10])  # Adjust the step size based on your data size

# Show the plot
plt.show()



#Data Preprocessing
target_y = stock_data['Close']
X_feat = stock_data.iloc[:,0:3]
stock_data = stock_data[['Open', 'High', 'Low', 'Close']]

#Feature Scaling
sc = StandardScaler()
stock_data_ft = sc.fit_transform(stock_data.values)
stock_data_ft = pd.DataFrame(columns=stock_data.columns,
                            data=stock_data_ft,
                            index=stock_data.index)


def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # Extract a window of size `n_steps` for features
        X.append(data[i:i + n_steps, :-1])  # All but last column
        
        # Extract the `n_steps`-th target value
        y.append(data[i + n_steps - 1, -1])  # Target value (last column)
    return np.array(X), np.array(y)



X1, y1 = lstm_split(stock_data_ft.values, n_steps=1)
train_split=0.8
split_idx= int(np.ceil(len(X1)*train_split))
date_index = stock_data_ft.index
X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]
print(X1. shape, X_train. shape, X_test.shape, y_test.shape)



# Define LSTM Model
lstm = Sequential()

# LSTM Layer (Set return_sequences=False if you want one output per sample)
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))

# Output Layer
lstm.add(Dense(1))

# Compile the Model
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Print Model Summary
lstm.summary()



history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=2, shuffle=False)



# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)

# Annotating the plot
plt.title('Training Loss vs. Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



y_pred = lstm.predict(X_test)

#If n = 1

# Ensure y_test is squeezed (flattened) if it has an extra dimension
y_test_flat = y_test.squeeze()
y_pred_flat = y_pred.squeeze()

# Plot the true values and predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_flat, label="True Values", color="green", linewidth=2)
plt.plot(y_pred_flat, label="LSTM Predictions", color="red", linewidth=2)

# Add labels, title, and legend
plt.title("LSTM Predictions vs. True Values", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()



mse = mean_squared_error(y_test, y_pred_flat)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred_flat)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("MAPE: ", mape)



lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]),
              activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()



history=lstm.fit(X_train, y_train,
                 epochs=100, batch_size=4,
                 verbose=2, shuffle=False)
y_pred = lstm.predict(X_test)



# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)

# Annotating the plot
plt.title('Training Loss vs. Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

# Ensure y_test is squeezed (flattened) if it has an extra dimension
y_test_flat = y_test.squeeze()

# Plot the true values and predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_flat, label="True Values", color="green", linewidth=2)
plt.plot(y_pred, label="LSTM Predictions", color="red", linewidth=2)

# Add labels, title, and legend
plt.title("LSTM Predictions vs. True Values", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()



mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("MAPE: ", mape)



n_steps=10
X1, y1 = lstm_split(stock_data_ft.values, n_steps=n_steps)
train_split=0.8
split_idx = int(np.ceil(len(X1)*train_split))
date_index = stock_data_ft.index
X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:-n_steps]
print(X1.shape, X_train.shape, X_test.shape, X_test_date.shape, y_test.shape)



lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]),
              activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()



history=lstm.fit(X_train, y_train,
                 epochs=100, batch_size=4,
                 verbose=2, shuffle=False)



# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)

# Annotating the plot
plt.title('Training Loss vs. Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



y_pred = lstm.predict(X_test)

# Ensure y_test is squeezed (flattened) if it has an extra dimension
y_test_flat = y_test.squeeze()

# Plot the true values and predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_flat, label="True Values", color="green", linewidth=2)
plt.plot(y_pred, label="LSTM Predictions", color="red", linewidth=2)

# Add labels, title, and legend
plt.title("LSTM Predictions vs. True Values", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()



mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("MAPE: ", mape)



train_split = 0.8
split_idx = int(np.ceil(len(stock_data)*train_split))
train = stock_data[['Close']].iloc[:split_idx]
test = stock_data[['Close']].iloc[split_idx:]
print(test)
test_pred = np.array([train.rolling(10).mean().iloc[-1]]*len(test)).reshape((-1,1))
print('Test MSE: %.3f' % mean_squared_error(test, test_pred))
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(test, test_pred)))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test, test_pred))
plt.figure(figsize=(10,5))
plt.plot(test.index, test)
plt.plot(test.index, test_pred)
plt.show()



from statsmodels.tsa.api import SimpleExpSmoothing
X = stock_data[['Close']].values
train_split = 0.8
split_idx = int(np.ceil(len(X)*train_split))
train = X[:split_idx]
test = X[split_idx:]
test_concat = np.array([]).reshape((0,1))

for i in range(len(test)):
  train_fit = np.concatenate((train, np.asarray(test_concat)))
  fit = SimpleExpSmoothing(np.asarray(train_fit)).fit(smoothing_level=0.1)
  test_pred = fit.forecast(1)
  test_concat = np.concatenate((np.asarray(test_concat), test_pred.reshape((-1,1))))

# The change is here: Using test_concat instead of test_pred for RMSE calculation
print('Test MSE: %.3f' % mean_squared_error(test, test_concat))
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(test, test_concat)))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test, test_concat))
plt.figure(figsize=(10,5))
plt.plot(test)
plt.plot(test_concat)
plt.show()
