import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
file_path = "C:\\Users\\jesse\\Desktop\\PPMLT\\Tests\\APPL\\AAPL_train.csv"
df = pd.read_csv(file_path)

# Feature engineering
'''
features = [
    'RSI', 'Short_EMA_12', 'Long_EMA_26',
    'MACD_Line', 'Signal_Line', 'MACD_Histogram',
    'EMA_5', 'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26',
    'EMA_50', 'EMA_100', 'EMA_200'
]
'''
features = [
    'RSI', 'Short_EMA_12', 'Long_EMA_26',
    'MACD_Line', 'Signal_Line', 'MACD_Histogram'
]

# Shift the Close column by one day to align with the features of the previous day
df['Close'] = df['Close'].shift(1)

# Drop the last row since it will have NaN in the Close column after shifting
df = df[:-1]

# Create a new column for the percentage change
df['Percentage Change'] = (df['Close'].shift(1) - df['Close']) / df['Close'] * 100

# Separate features and labels
X = df[features]
y = df['Percentage Change']

# Handle missing values in y
y = y.dropna()

# Handle missing values in X
X = X.iloc[y.index]  # Align X with the indices of non-NaN values in y

# Handle missing values in X using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

