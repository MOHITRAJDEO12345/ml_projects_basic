import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('advanced_stock_price_prediction\train.csv')
print(df.head())