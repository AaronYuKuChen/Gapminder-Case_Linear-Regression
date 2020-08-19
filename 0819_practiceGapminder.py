from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('Gapminder.csv')
y = df['life'].values
X = df.loc[:, ['fertility', 'population', 'GDP']].values
# print(y)
# print(type(X))
y_pd = pd.DataFrame(y)
# y_pd
X_pd = pd.DataFrame(X)
# X_pd
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

sns.heatmap(df.corr(), square=False, cmap='RdYlGn')
# plt.show()
print(df.corr())
X_fertility = np.array(df.fertility)
print(X_fertility)
# type(X_fertility)
y = np.array(df.life)
# y
# type(y)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train,y_train)
y_pred = reg_all.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))