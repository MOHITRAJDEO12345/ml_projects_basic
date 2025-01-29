import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('stock_price_predictor\yahoo_stock.csv')
print(df.head())
print(df.columns)
print(df.info())
# print(df.describe())