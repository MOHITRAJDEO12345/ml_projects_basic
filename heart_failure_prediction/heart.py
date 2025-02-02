import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'heart_failure_prediction\heart_failure_clinical_records_dataset.csv')
print(df.head())
print(df.info())
print(df.columns)
print(df.duplicated().sum())

# Create new features
df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80, np.inf], labels=['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])

# Normalize/Standardize features
scaler = StandardScaler()
df[['creatinine_phosphokinase', 'platelets', 'serum_creatinine', 'serum_sodium']] = scaler.fit_transform(df[['creatinine_phosphokinase', 'platelets', 'serum_creatinine', 'serum_sodium']])

# Handle outliers (example: capping outliers)
for column in ['creatinine_phosphokinase', 'platelets', 'serum_creatinine', 'serum_sodium']:
    upper_limit = df[column].quantile(0.99)
    lower_limit = df[column].quantile(0.01)
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])

# Visualizations
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

# Histograms
df.hist(figsize=(20, 15), bins=30)
plt.show()

df.head()