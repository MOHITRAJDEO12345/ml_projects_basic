import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('diabetes_prediction\diabetes.csv')
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.describe())

df = df.dropna()
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('int')
print(df.info())
print(df['Age'].head())

X = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
y = df['Outcome']
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

