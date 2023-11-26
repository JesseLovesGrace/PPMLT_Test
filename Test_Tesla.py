import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Step 1: Read stock data
stock_data = pd.read_csv('C:/Users/jesse/Desktop/PPMLT/Tests/TESLA/TSLA.csv')

# Calculate price change based on 'Close' prices of current day to n days later
stock_data['Price_Change_Column'] = (stock_data['Close'].shift(20) - stock_data['Close']) / stock_data['Close'] * 100

# Step 2: Create an empty DataFrame to store merged data
merged_data = stock_data.copy()

# Step 3: Read and merge index data
'''
index_files = [
    '^VIX.csv', 'SPY.csv',
    '^IRX.csv', '^TNX.csv',
    '^VXN.csv'
    ]
'''
index_files = [
    '^VIX.csv', 'SPY.csv',
    '^IRX.csv', '^TNX.csv',
    '^VXN.csv'
]

for index_file in index_files:
    index_data = pd.read_csv(f'C:/Users/jesse/Desktop/PPMLT/Tests/TESLA/Indexes/{index_file}')
    # Use suffixes to handle duplicate columns
    merged_data = pd.merge(merged_data, index_data, on='Date', how='inner', suffixes=('', f'_{index_file[:-4]}'))

# Step 4: Preprocess the data

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use 'mean', 'median', 'most_frequent', or a constant value
X = merged_data.drop(['Date', 'Price_Change_Column'], axis=1)  # Adjust 'Price_Change_Column'
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Drop rows with NaN values
merged_data = pd.concat([merged_data['Date'], X_imputed, merged_data['Price_Change_Column']], axis=1)
merged_data.dropna(inplace=True)

# Step 5: Split the data
y = merged_data['Price_Change_Column']  # Adjust 'Price_Change_Column'

# Ensure that X_imputed and y have the same number of rows
X_imputed = X_imputed.iloc[:len(y)]

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 6: Build and evaluate the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
