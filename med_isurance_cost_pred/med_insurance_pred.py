# -*- coding: utf-8 -*-
"""credit_fraud_pred.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1egYJDCsgDOJj6zuiJ0ZP5KeltSWBDXxX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

insurance_dataset = pd.read_csv('med_isurance_cost_pred\insurance.csv')

insurance_dataset.head()

insurance_dataset.shape

insurance_dataset.info()

insurance_dataset.isnull().sum()

insurance_dataset.describe()

insurance_dataset['sex'].value_counts()

sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()

insurance_dataset['smoker'].value_counts()

insurance_dataset['region'].value_counts()

plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Encoding 'sex' column
le_sex = LabelEncoder()
insurance_dataset['sex'] = le_sex.fit_transform(insurance_dataset['sex'])
print("Sex Encoding:", dict(enumerate(le_sex.classes_)))

# Encoding 'smoker' column
le_smoker = LabelEncoder()
insurance_dataset['smoker'] = le_smoker.fit_transform(insurance_dataset['smoker'])
print("Smoker Encoding:", dict(enumerate(le_smoker.classes_)))

# Encoding 'region' column
le_region = LabelEncoder()
insurance_dataset['region'] = le_region.fit_transform(insurance_dataset['region'])
print("Region Encoding:", dict(enumerate(le_region.classes_)))

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

training_data_prediction =regressor.predict(X_train)

r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

test_data_prediction =regressor.predict(X_test)

r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

input_data = (31,1,25.74,0,1,0)

import numpy as np

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print('The insurance cost is USD', prediction[0])

