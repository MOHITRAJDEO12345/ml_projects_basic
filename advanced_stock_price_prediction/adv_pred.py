import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

# df = pd.read_csv(r'advanced_stock_price_prediction\train.csv')
# print(df.head())

# print(df.info())
# print(df.describe())
# print(df.columns)
# print(df.duplicated().sum())

# sns.pairplot(df, diag_kind='kde')
# plt.show()

# # Box plot
# plt.figure(figsize=(20, 10))
# df.boxplot()
# plt.xticks(rotation=90)
# plt.show()

# Correlation matrix heatmap
# plt.figure(figsize=(20, 15))
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.show()

# # Histograms
# df.hist(figsize=(20, 15), bins=30)
# plt.show()
# last 2 not working due to objects in the dataset

# learned something new about the dataset
# take both the train and test datset together for better prediction

train_df = pd.read_csv(r'advanced_stock_price_prediction\train.csv')
test_df = pd.read_csv(r'advanced_stock_price_prediction\test.csv')

print(train_df.head())
print(test_df.head())

# print(train_df.info())
# print(test_df.info())
# print(train_df.describe())
# print(test_df.describe())

print(train_df.columns)