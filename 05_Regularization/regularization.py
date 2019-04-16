import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
#from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#%%
# Load datasets
dataset = pd.read_csv('boston.csv')

X = dataset.drop('MEDV', axis=1)
y = dataset['MEDV']

#%%
# Zad. 1
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

#%%
# Zad. 2
# Create model, learn and predict on test data

# Create linear regressor
regr = LinearRegression()
 
# Fit the model to train data
regr.fit(X_train, y_train)

# Predict values on test data after learing
y_predicted = regr.predict(X_test)

#%%
# Score
score_train = regr.score(X_train, y_train)
print('Score - train data = {}'.format(score_train))

score_test = regr.score(X_test, y_test)
print('Score - test data = {}'.format(score_test))

#%%
# Overfit data

#%%
# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
# Aproximation with second order polynomial
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)

#%%
# Split data once again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

#%%
# Fit the model to train data
regr.fit(X_train, y_train)

# Predict values on test data after learing
y_predicted = regr.predict(X_test)

# Score
score_train = regr.score(X_train, y_train)
print('Score - train data = {}'.format(score_train))

score_test = regr.score(X_test, y_test)
print('Score - test data = {}'.format(score_test))

#%%
# Zad. 4
# Ridge regularization
ridge_reg = Ridge(alpha=10)
ridge_reg.fit(X_train, y_train)

# Score
score_train = ridge_reg.score(X_train, y_train)
print('Score - train data = {}'.format(score_train))

score_test = ridge_reg.score(X_test, y_test)
print('Score - test data = {}'.format(score_test))

#%%
# Draw plot with accuracy against 'alpha' parameter
acc_ridge = []
alphas = []
for i in range(-3, 4):
    alphas.append(10**i)
    ridge_reg = Ridge(alpha=10**i)
    ridge_reg.fit(X_train, y_train)
    acc_ridge.append(ridge_reg.score(X_test, y_test))

plt.figure()
plt.plot(alphas, acc_ridge)
plt.xlabel('Alpha')
plt.ylabel('Test accuracy')
plt.show()

#%%
# Lasso regression
acc_lasso = []
alphas_lasso = []
for i in range(1, 101):
    alphas_lasso.append(i*0.01)

for alpha in alphas_lasso:
    lasso_reg = Lasso(alpha, max_iter=10000)
    lasso_reg.fit(X_train, y_train)
    acc_lasso.append(lasso_reg.score(X_test, y_test))

best_lasso_alpha = alphas_lasso[np.argmax(acc_lasso)] 
lasso_reg = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lasso_reg.fit(X_train, y_train)

print('Best Lasso\'s "alpha" is {}'.format(best_lasso_alpha))
print('Train score {}'.format(lasso_reg.score(X_train, y_train)))
print('Test score {}'.format(lasso_reg.score(X_test, y_test)))

#%%
# Regularyzacja regresji logistycznej

import os
path = os.getcwd() + '/breast_cancer.txt'
dataset = pd.read_csv(path, header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

























