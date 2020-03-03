import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
df = pd.read_csv("E:\multiplel_r.csv")
print(df)
df = df.fillna(3)
print(df)
model = linear_model.LinearRegression()
model.fit(df[['area','bedroom','age']],df.price)
print(model.coef_)
print(model.intercept_)
predicted_price = model.predict([[3000,3,40]])
print(predicted_price)