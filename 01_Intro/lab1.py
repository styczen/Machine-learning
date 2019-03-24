import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris

irisRaw = load_iris()
# for key, value in irisRaw.items():
#     print(key, ":", value)
iris = pd.DataFrame(data=np.c_[irisRaw['data'], irisRaw['target']],
                    columns=irisRaw['feature_names'] + ['target'])

# 1. Show all data
print('\n************ 1. *************')
print(iris.to_string())

# 2. Number of rows and columns
print('\n************ 2. *************')
print('Rows:', iris.shape[0])
print('Cols:', iris.shape[1])

# 3. Display information about data
print('\n************ 3. *************')
print('Basic information:\n', iris.describe())

print('Info about every column separately')
for col in iris.columns:
   groupby = iris.groupby(by=col)
   print(groupby.describe())

# 4. Show first 5 rows
print('\n************ 4. *************')
print(iris.head())

# 5. Check wheter there is some data missing
print('\n************ 5. *************')
iris_wo_null = iris.dropna()
if iris.equals(iris_wo_null):
   print('There is no data missing')
else:
   print('There is some data missing')
    
# 6. Sorting
print('\n************ 6. *************')
iris_second_col_sorted = iris.sort_values(by='sepal width (cm)')
print('Sorted dataframe:\n', iris_second_col_sorted)

# 7. Min and max value of petal length
print('\n************ 7. *************')
max_length = iris['petal length (cm)'].max()
min_length = iris['petal length (cm)'].min()
max_length_index = iris['petal length (cm)'].idxmax()
min_length_index = iris['petal length (cm)'].idxmin()
print('Index: {}; max: {}'.format(max_length_index, max_length))
print('Index: {}; min: {}'.format(min_length_index, min_length))

# 8. Standard deviation for every column
print('\n************ 8. *************')
for col in iris.columns:
    std = iris[col].std()
    print('Standard deviation of "{}" is equal to {}'.format(col, std))

# 9. Extract columns in which sepal lenght is greater than average length
print('\n************ 9. *************')
mean = iris['sepal length (cm)'].mean() 
items_idx = iris['sepal length (cm)'] > mean # get indexes of rows with sepal length greater than mean
iris_filtered = iris.loc[items_idx, :]
print(iris_filtered)

# Ploting
# 1. 
print('\n*****************************')

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
