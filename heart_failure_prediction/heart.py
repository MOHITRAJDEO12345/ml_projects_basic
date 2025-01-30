import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv('heart_failure_prediction\heart_failure_clinical_records_dataset.csv')
print(df.head())

print(df.info())
# print(df.describe())
print(df.columns)
print(df.duplicated().sum())

sns.pairplot(df, diag_kind='kde')
plt.show()

# Box plot
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=90)
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()