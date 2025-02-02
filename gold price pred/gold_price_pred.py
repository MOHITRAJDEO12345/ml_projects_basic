# -*- coding: utf-8 -*-
"""car_price_pred.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1egYJDCsgDOJj6zuiJ0ZP5KeltSWBDXxX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv('/content/gld_price_data.csv')

gold_data.head()

gold_data.tail(9)

gold_data.info()

gold_data.describe()

gold_data.isnull().sum()

gold_data.drop('Date',axis=1,inplace=True)

correlation = gold_data.corr()
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Reds')

sns.histplot(gold_data['GLD'],color='blue')

X = gold_data.drop(['GLD'],axis=1)
Y = gold_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
test_data_prediction = regressor.predict(X_test)

error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

Y_test = list(Y_test)

plt.plot(Y_test, color='red', label = 'Actual Value')
plt.plot(test_data_prediction, color='blue', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

