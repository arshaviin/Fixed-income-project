# -*- coding: utf-8 -*-
"""FixedIncomeTrading.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QhfUV4E4p74ojjmYDzO5l2gXPV-OSINc
"""

!pip install copulas

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from copulas.multivariate import GaussianMultivariate

# Function to calculate bond prices including coupon payments
def calculate_bond_price(face_value, annual_yield, years_to_maturity, coupon_rate, pay_freq):
    """ Calculate the bond price using the formula for bonds with coupon payments. """
    if years_to_maturity <= 1:  # No coupon for <= 1 year maturities
        periods = years_to_maturity  # Ensure 'periods' is an integer
        discount_rate = annual_yield
        face_value_pv = face_value / ((1 + discount_rate) ** periods)
        coupon_pv=0
    else:
      annual_coupon = face_value * coupon_rate
      coupon_payment = annual_coupon / pay_freq
      discount_rate = annual_yield / pay_freq
      periods = int(years_to_maturity * pay_freq)  # Ensure 'periods' is an integer

    # Calculate the present value of the future coupon payments
      coupon_pv = sum(coupon_payment / ((1 + discount_rate) ** period) for period in range(1, periods + 1))

    # Calculate the present value of the face value
      face_value_pv = face_value / ((1 + discount_rate) ** periods)

    return coupon_pv + face_value_pv

# Load datasets
data_2023 = pd.read_csv('daily-treasury-rates.csv')
data_2024 = pd.read_csv('daily-treasury-rates-2.csv')

# Concatenate and sort the data
data = pd.concat([data_2023, data_2024])
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Define years to maturity based on column names
column_to_maturity = {
    '1 Mo': 1/12, '2 Mo': 2/12, '3 Mo': 3/12, '4 Mo': 4/12, '6 Mo': 6/12,
    '1 Yr': 1, '2 Yr': 2, '3 Yr': 3, '5 Yr': 5, '7 Yr': 7, '10 Yr': 10, '20 Yr': 20, '30 Yr': 30
}

maturity_years = [column_to_maturity[col] for col in data.columns]

def prepare_data(data, look_back=30):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_data)

    # Prepare sequences for LSTM
    X, y = [], []
    for i in range(len(principal_components) - look_back):
        X.append(principal_components[i:i + look_back])
        y.append(principal_components[i + look_back])
    X, y = np.array(X), np.array(y)

    return X, y, pca, scaler

def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(X.shape[2]))  # Predicting all principal components
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)
    return model

profits_losses = pd.DataFrame(index=data.index[31:], columns=data.columns)
look_back = 30
face_value = 100
coupon_rate = 0.03  # Assume a 3% annual coupon rate
payment_frequency = 2  # Semi-annual payments

for start in range(len(data) - look_back - 1):
    current_data = data.iloc[start:start + look_back]
    X, y, pca, scaler = prepare_data(current_data, look_back - 1)
    model = train_lstm(X, y)

    # Predict the next day's PCs and transform to yield space
    next_day_pcs = model.predict(X[-1].reshape(1, -1, 3))
    predicted_rates = scaler.inverse_transform(pca.inverse_transform(next_day_pcs)).flatten()

    # Actual rates from the last day in the current data window
    actual_rates = data.iloc[start + look_back - 1].values
    actual_rates_1 = data.iloc[start + look_back].values

    # Calculate bond prices for 30th and 31st day using maturity years
    prices_30th = [calculate_bond_price(face_value, rate, maturity_years[i], coupon_rate if maturity_years[i] > 1 else 0, payment_frequency)
                   for i, rate in enumerate(actual_rates)]
    prices_31st = [calculate_bond_price(face_value, rate, maturity_years[i], coupon_rate if maturity_years[i] > 1 else 0, payment_frequency)
                   for i, rate in enumerate(actual_rates_1)]

    # Calculate profit/loss
    profits = np.where(predicted_rates < actual_rates,
                       np.array(prices_30th) - np.array(prices_31st),
                       np.array(prices_31st) - np.array(prices_30th))
    profits_losses.iloc[start] = profits

# Summarize results
total_pnl = profits_losses.sum()
print("Total P&L by Maturity:")
print(total_pnl)

import numpy as np
import pandas as pd

# Example data, replace this with your actual returns data
data = {
    'Maturity': ['1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr'],
    'PnL': [-0.303551, -0.16941, 6.241824, 0.789583, -1.422649, -2.099851, -0.114557, 0.029741, -0.070687, 0.179026, 0.256188, 0.178165, 0.017201]
}

df = pd.DataFrame(data)
# Convert P&L to returns (assuming P&L are in percentages)
df['Returns'] = df['PnL'] / 100  # If your PnL are absolute values, adjust the calculation accordingly

# Risk-free rate - let's assume 0.5% annualized for example
risk_free_rate = 0.005 / 12  # Monthly risk-free rate if annualized risk-free rate is 0.5%

# Calculate average return
average_return = df['Returns'].mean()

# Calculate standard deviation of returns
std_dev = df['Returns'].std()

# Calculate Sharpe Ratio
sharpe_ratio = (average_return - risk_free_rate) / std_dev

print("Sharpe Ratio:", sharpe_ratio)