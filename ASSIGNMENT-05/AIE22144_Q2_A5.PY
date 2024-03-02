import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

file_path = r"C:\Users\aasri\Desktop\ASSIGNMENT_5\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
# Read the CSV file using pd.read_csv() instead of pd.read_excel()
df = pd.read_csv(file_path)

print("All Column Names:", df.columns)

# Select the column names for actual and predicted prices
actual_prices_column_name = 'Milk'
predicted_prices_column_name = 'Milk'

# Drop rows with NaN values in the selected columns
df_cleaned = df.dropna(subset=[actual_prices_column_name, predicted_prices_column_name])

# Extract actual and predicted prices as arrays
actual_prices = df_cleaned[actual_prices_column_name].values
predicted_prices = df_cleaned[predicted_prices_column_name].values

print("Actual Values:", actual_prices)
print("Predicted Values:", predicted_prices)

# Calculate metrics
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

if len(actual_prices) >= 2:
    r2 = r2_score(actual_prices, predicted_prices)
    print(f'R-squared (R2) score: {r2}')
else:
    print('Insufficient samples to calculate R-squared.')

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
