import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('heart_failure_prediction\heart_failure_clinical_records_dataset.csv')
print(df.head())