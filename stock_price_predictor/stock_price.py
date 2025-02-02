import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'stock_price_predictor\yahoo_stock.csv')
print(df.head())
print(df.info())
print(df.columns)
print(df.duplicated().sum())

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create new features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop the original Date column
df = df.drop(columns=['Date'])

# Normalize/Standardize features
scaler = StandardScaler()
df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']] = scaler.fit_transform(df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']])

# Visualizations
# Scatter plot matrix
# sns.pairplot(df, diag_kind='kde')
# plt.show()

# # Box plot
# plt.figure(figsize=(12, 8))
# df.boxplot()
# plt.xticks(rotation=90)
# plt.show()

# # Correlation matrix heatmap
# plt.figure(figsize=(10, 8))
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.show()

df.head()