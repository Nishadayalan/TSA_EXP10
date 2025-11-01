# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 01-11-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("/content/plane_crash.csv")

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with missing or invalid dates
data = data.dropna(subset=['Date'])

# Sort by date
data = data.sort_values(by='Date')

# Reset index
data.reset_index(drop=True, inplace=True)

# Select target variable
target_variable = 'Fatalities'  # You can change this to 'Aboard' or 'Ground' if you prefer

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data[target_variable])
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title(f'{target_variable} Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of the target variable
print("\n--- Checking Stationarity ---")
check_stationarity(data[target_variable])

# Plot ACF and PACF
plt.figure(figsize=(8, 4))
plot_acf(data[target_variable].dropna(), lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(8, 4))
plot_pacf(data[target_variable].dropna(), lags=30)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Split dataset into train and test (80/20)
train_size = int(len(data) * 0.8)
train = data[target_variable][:train_size]
test = data[target_variable][train_size:]

# Build SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('\nRMSE:', rmse)

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel(target_variable)
plt.title(f'SARIMA Model Predictions for {target_variable}')
plt.legend()
plt.show()



```

### OUTPUT:

<img width="998" height="510" alt="image" src="https://github.com/user-attachments/assets/6c1e42a5-99b4-466b-beb8-6f2b30ec5e1c" />

<img width="663" height="635" alt="image" src="https://github.com/user-attachments/assets/4a90ccf5-4fe6-4705-9836-c8f1aa902b61" />


<img width="698" height="540" alt="image" src="https://github.com/user-attachments/assets/de3a0c28-5a56-4895-9339-c530fdb420d9" />


<img width="975" height="503" alt="image" src="https://github.com/user-attachments/assets/41217b15-4d06-43a8-9584-11a499ae560d" />



### RESULT:
Thus the program run successfully based on the SARIMA model.
