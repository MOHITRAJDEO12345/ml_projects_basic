import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv('diabetes_prediction\diabetes.csv')
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.describe())

df = df.dropna()
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('int')
# print(df.info())
# print(df.describe())
# print(df.columns)
# print(df.duplicated().sum())

# sns.pairplot(df, diag_kind='kde')
# plt.show()

# # Box plot
# plt.figure(figsize=(12, 8))
# df.boxplot()
# plt.xticks(rotation=90)
# plt.show()

# # Correlation matrix
# plt.figure(figsize=(10, 8))
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.show()


# import pandas as pd
# import numpy as np

# df = pd.read_csv(r'diabetes_prediction\diabetes.csv')

# Handle missing values by replacing zeros with the median of the column
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zeros:
    df[column] = df[column].replace(0, np.nan)
    df[column] = df[column].fillna(df[column].median())

# Create new features
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70, 80, np.inf], labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])

# Normalize/Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = scaler.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.duplicated().sum())

# # Visualizations
# # Scatter plot matrix
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