# -*- coding: utf-8 -*-
"""car_price_pred.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1egYJDCsgDOJj6zuiJ0ZP5KeltSWBDXxX
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

car_dataset = pd.read_csv('car_price_pred\car data.csv')

car_dataset.head()

car_dataset.describe()
car_dataset.info()
car_dataset.isnull().sum()

car_dataset.Transmission.value_counts()

car_dataset.Seller_Type.value_counts()

car_dataset.Fuel_Type.value_counts()

car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)
training_data_prediction = lin_reg_model.predict(X_train)

error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()





lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)
training_data_prediction = lass_reg_model.predict(X_train)

error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

"""training graph


"""

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

test_data_prediction = lass_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()





single_row = X_test.iloc[[10]]
predicted_price = lin_reg_model.predict(single_row)
print("Predicted Price:", predicted_price[0])

