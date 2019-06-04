import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# %%

data = load_diabetes()

#df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
#                  columns=iris['feature_names'] + ['target'])
df = pd.DataFrame(data=data['data'],
                  columns=data['feature_names'])

# %%

# Preprocess data
# Replace 'inf' to 'nan'
df.replace(to_replace=[np.inf, -np.inf],
           value=np.nan,
           inplace=True)

# Remove 'nan'
df = df.dropna()

# %%

# Scale data
X = pd.DataFrame(data=scale(df), 
                 index=df.index, 
                 columns=df.columns)

# %%

# Split data
df_train, df_test = train_test_split(X,
                                     test_size=0.25)

# %%

# Covariance matrix
cov_mat = df.cov()

# %%

# Eigenvectors and eigenvalues
w, v = np.linalg.eig(cov_mat)
v = v.T

# %%

# Sort eigenvalues
eig = pd.DataFrame(data=v, index=w)
eig.sort_index(ascending=False, inplace=True)

# %%

plt.figure(figsize=(7, 5))
plt.plot(np.arange(1, 11), eig.index/sum(eig.index), '-o')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.ylim(0, 1.05)
plt.xticks(np.arange(1, 11))
plt.show()
