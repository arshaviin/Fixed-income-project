# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to calculate bond prices including coupon payments
def calculate_bond_price(face_value, annual_yield, years_to_maturity, coupon_rate, pay_freq):
    """ Calculate the bond price using the formula for bonds with coupon payments. """
    periods = int(years_to_maturity * pay_freq)
    annual_coupon = face_value * coupon_rate
    coupon_payment = annual_coupon / pay_freq
    discount_rate = annual_yield / pay_freq
    
    # Calculate the present value of the future coupon payments
    coupon_pv = sum(coupon_payment / ((1 + discount_rate) ** period) for period in range(1, periods + 1))
    # Calculate the present value of the face value
    face_value_pv = face_value / ((1 + discount_rate) ** periods)
    return coupon_pv + face_value_pv

# Load and concatenate datasets
data_2023 = pd.read_csv("C:/Users/shila/OneDrive/Documents/Courses/Fixed Income/daily-treasury-rates.csv")
data_2024 = pd.read_csv("C:/Users/shila/OneDrive/Documents/Courses/Fixed Income/daily-treasury-rates-2.csv")

data = pd.concat([data_2023, data_2024])
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Define maturity for columns if not already defined
column_to_maturity = {
    '1 Mo': 1/12, 
    '2 Mo': 2/12, 
    '3 Mo': 3/12, 
    '4 Mo': 4/12, 
    '6 Mo': 6/12,
    '1 Yr': 1, 
    '2 Yr': 2, 
    '3 Yr': 3, 
    '5 Yr': 5, 
    '7 Yr': 7, 
    '10 Yr': 10, 
    '20 Yr': 20, 
    '30 Yr': 30
}
maturity_years = [column_to_maturity[col] for col in data.columns]
# Prepare features and target
def prepare_features_targets(data, look_back=30):
    X, y = [], []
    scaler = StandardScaler()
    
    for start in range(len(data) - look_back - 1):
        end = start + look_back
        current_data = data.iloc[start:end]
        next_day_data = data.iloc[end]
        
        # Feature: scaled current data
        scaled_data = scaler.fit_transform(current_data)
        X.append(scaled_data.flatten())  # Flatten the features into a single vector
        
        # Target: price differences for each bond maturity
        prices_today = {col: calculate_bond_price(100, current_data[col].iloc[-1], column_to_maturity[col], 0.03, 2) for col in data.columns}
        prices_tomorrow = {col: calculate_bond_price(100, next_day_data[col], column_to_maturity[col], 0.03, 2) for col in data.columns}
        pnl = {col: prices_tomorrow[col] - prices_today[col] for col in data.columns}
        y.append([pnl[col] for col in data.columns])
    
    return np.array(X), np.array(y)

X, y = prepare_features_targets(data)

# Split data into training and testing
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train Gradient Boosting Regressor with MultiOutputWrapper
multioutput_regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1))
multioutput_regressor.fit(X_train, y_train)

# Predict and calculate mean squared error
y_pred = multioutput_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Model Mean Squared Error: {mse}")

# Calculate and display total P&L by maturity
total_pnl_by_maturity = np.sum(y_pred - y_test, axis=0)  # Sum differences in predictions and actuals by column
maturity_columns = list(column_to_maturity.keys())  # Column names
total_pnl_by_maturity_dict = dict(zip(maturity_columns, total_pnl_by_maturity))
print("Total P&L by Maturity:")
print(total_pnl_by_maturity_dict)

pnl_dataframe = pd.DataFrame(list(total_pnl_by_maturity_dict.items()), columns=['Maturity', 'Total P&L'])
print(pnl_dataframe)

def plot_pnl_by_maturity(pnl_dict):
    # Sorting the dictionary by maturity years to maintain order in the plot
    sorted_pnl = dict(sorted(pnl_dict.items(), key=lambda x: column_to_maturity[x[0]]))
    maturities = list(sorted_pnl.keys())
    pnl_values = list(sorted_pnl.values())

    plt.figure(figsize=(10, 6))
    plt.bar(maturities, pnl_values, color='skyblue')
    plt.xlabel('Maturity')
    plt.ylabel('Total P&L')
    plt.title('Total Profit and Loss by Maturity')
    plt.xticks(rotation=45)  # Rotate maturity labels for better readability
    plt.grid(True)
    plt.show()

# Call the plotting function
plot_pnl_by_maturity(total_pnl_by_maturity_dict)

# P&L data from the image (assumed to be percentages)
pnl_data = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.50348, 
                     1.859743, 0.999891, 0.313441, 0.154188, 
                     0.161772, 0.092596, 0.073056, 0.108659])

# Risk-free rate assumption (annualized)
risk_free_rate_annual = 0.005  # 0.5%
# Assuming monthly returns for simplicity of example
risk_free_rate = risk_free_rate_annual / 12  

# Calculate average return (mean of pnl_data)
average_return = np.mean(pnl_data)

# Calculate standard deviation of returns
std_dev = np.std(pnl_data)

# Calculate Sharpe Ratio
sharpe_ratio = (average_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan

print(f"Sharpe Ratio: {sharpe_ratio:.4f}")





