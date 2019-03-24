#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris


# In[4]:


# Load raw data and extract it to more readable dataframe
irisRaw = load_iris()

iris = pd.DataFrame(data=np.c_[irisRaw['data'], irisRaw['target']],
                    columns=irisRaw['feature_names'] + ['target'])


# In[6]:


# 0. Display raw data
for key, value in irisRaw.items():
    print(key, ":", value)


# In[46]:


# 1. Show all data
# print(iris)
print(iris.to_string())


# In[47]:


# 2. Number of rows and columns
print('Rows:', iris.shape[0])
print('Cols:', iris.shape[1])


# In[48]:


# 3. Display information about data
print('Basic information:\n', iris.describe().to_string())


# In[49]:


print('Info about every column separately')
for col in iris.columns:
    print('--------------------------------------------------------------------------------------------------------------------------------------------------')
    groupby = iris.groupby(by=col)
#     print(groupby.describe().to_string())
    print(groupby.describe())


# In[50]:


# 4. Show first 5 rows
print(iris.head().to_string())


# In[51]:


# 5. Check wheter there is some data missing
iris_wo_null = iris.dropna()
if iris.equals(iris_wo_null):
    print('There is no data missing')
else:
    print('There is some data missing')


# In[52]:


# 6. Sorting
iris_second_col_sorted = iris.sort_values(by='sepal width (cm)')
print('Sorted dataframe:\n', iris_second_col_sorted.to_string())


# In[53]:


# 7. Min and max value of petal length
max_length = iris['petal length (cm)'].max()
min_length = iris['petal length (cm)'].min()
max_length_index = iris['petal length (cm)'].idxmax()
min_length_index = iris['petal length (cm)'].idxmin()
print('Max value {} at index {}'.format(max_length, max_length_index))
print('Min value {} at index {}'.format(min_length, min_length_index))


# In[54]:


# 8. Standard deviation for every column
for col in iris.columns:
    std = iris[col].std()
    print('Standard deviation of "{}" is equal to {}'.format(col, std))


# In[55]:


# 9. Extract columns in which sepal length is greater than average sepal length
mean = iris['sepal length (cm)'].mean() 
items_idx = iris['sepal length (cm)'] > mean # get indexes of rows with sepal length greater than mean
iris_filtered = iris.loc[items_idx, :]
print(iris_filtered.to_string())


# In[56]:


# Ploting
for i, feature in enumerate(irisRaw['feature_names']):
    plt.figure(i)
    feature_data = iris[feature]
    
    # iterate through 3 iris types and extract value for every type
    iris_types_list = []
    for j, iris_type in enumerate(irisRaw['target_names']): 
        specific_iris_type = feature_data[iris['target'] == j]
        iris_types_list.append(specific_iris_type)
    
    plt.hist(x=iris_types_list, bins=30, color=['r', 'g', 'b'], label=irisRaw['target_names'])
   
    plt.title('Histogram of ' + feature)
    plt.xlabel(feature)
    plt.ylabel('Amount')
    plt.legend(loc='upper right')
plt.show()

