import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# %% Principal Components Analysis
df = pd.read_csv('usarrests.csv', index_col=0)

# Display simple info
df.info()

# %%

# Show first 5 entries
df.head()

# %%

# Calculate mean for every feature
df.mean()

# %%

# Calculate standard deviation for every feature
df.std()

# %%

# Calculate variance for every feature
df.var()

# %%

# Scale data
from sklearn.preprocessing import scale

X = pd.DataFrame(data=scale(df), 
                 index=df.index, 
                 columns=df.columns)

# %%

# Execute PCA decomposition
from sklearn.decomposition import PCA

pca_loadings = pd.DataFrame(data=PCA().fit(X).components_.T, 
                            index=df.columns,
                            columns=['V1', 'V2', 'V3', 'V4'])
pca_loadings

# %%

# Fit the PCA model and transform X to get the principal components
pca = PCA()
df_plot = pd.DataFrame(data=pca.fit_transform(X),
                       columns=['PC1', 'PC2', 'PC3', 'PC4'],
                       index=X.index)

# %%

# Show first few samples
df_plot.head()

# %%

# Plot results
fig, ax1 = plt.subplots(figsize=(9, 7))

ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)

# Plot Principal Components 1 and 2
for i in df_plot.index:
    ax1.annotate(s=i, 
                 xy=(-df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), 
                 ha='center')
# Plot reference lines
ax1.hlines(y=0, 
           xmin=-3.5, 
           xmax=3.5, 
           linestyles='dotted', 
           colors='grey')
ax1.vlines(x=0, 
           ymin=-3.5, 
           ymax=3.5, 
           linestyles='dotted', 
           colors='grey')
    
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

# Plot Principal Component loading vectors, using a second y-axis.
ax2 = ax1.twinx().twiny()

ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_xlabel('Principal Component loading vectors', color='red')
    
# Plot labels for vectors. Variable 'a' is a small offset parameter  
# to separate arrow tip and text.
a = 1.07
for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(s=i, 
                 xy=(-pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a),
                 color='red')
# Plot vectors
for i in range(4):
    ax2.arrow(x=0,
              y=0,
              dx=-pca_loadings.V1[i],
              dy=-pca_loadings.V2[i])
    
# %%
    
pca.explained_variance_ratio_

# %%

plt.figure(figsize=(7, 5))
plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xlim(0.75, 4.25)
plt.ylim(0, 1.05)
plt.xticks([1,2,3,4])
plt.show()

# %%

plt.figure(figsize=(7, 5))
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xlim(0.75, 4.25)
plt.ylim(0, 1.05)
plt.xticks([1,2,3,4])
plt.show()
    
# %% NCI60 Data Example    

# Loading data    
df2 = pd.read_csv('nci60.csv').drop('Unnamed: 0', axis=1)
df2.columns = np.arange(df2.columns.size)

# %%

# Display info
df2.info()

# %%

# Read in the labels to check our work later
y = pd.read_csv('nci60_y.csv', usecols=[1], skiprows=1, names=['type'])

# %%

# Scale the data
X = pd.DataFrame(scale(df2))
X.shape

# %%

# Fit the PCA model and transform X to get the principal components
pca2 = PCA()
df2_plot = pd.DataFrame(pca2.fit_transform(X))

# %%

# Plot first few components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

color_idx = pd.factorize(y.type)[0]
cmap = mpl.cm.hsv

# Left plot
ax1.scatter(df2_plot.iloc[:, 0],
            df2_plot.iloc[:, 1],
            c=color_idx,
            cmap=cmap,
            alpha=0.5,
            s=50)

ax1.set_ylabel('Principal Component 2')

# Right plot
ax2.scatter(df2_plot.iloc[:, 0],
            df2_plot.iloc[:, 2],
            c=color_idx,
            cmap=cmap,
            alpha=0.5,
            s=50)
ax2.set_ylabel('Principal Component 3')

# Custom legend for the classes (y) since we do not create scatter plots 
# per class (which could have their own labels).
handles = []
labels = pd.factorize(y.type.unique())
norm = mpl.colors.Normalize(vmin=0.0, vmax=14.0)

for i, v in zip(labels[0], labels[1]):
    handles.append(mpl.patches.Patch(color=cmap(norm(i)),
                                     label=v,
                                     alpha=0.5))
ax2.legend(handles=handles,
           bbox_to_anchor=(1.05, 1),
           loc=2,
           borderaxespad=0.)
# xlabel for both plots
for ax in fig.axes:
    ax.set_xlabel('Principal Component 1')

# %%

# Generate summary of the PVE (proportion of variance explained) of the first
# few principal components
pd.DataFrame([df2_plot.iloc[:, :5].std(axis=0, ddof=0).as_matrix(),
              pca2.explained_variance_ratio_[:5],
              np.cumsum(pca2.explained_variance_ratio_[:5])],
             index=['Standard Deviation', 'Proportion of Variance', 
                    'Cumulative Proportion'],
             columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# %%

df2_plot.iloc[:, :10].var(axis=0, ddof=0).plot(kind='bar', rot=0)
plt.ylabel('Variances')

# %%

# Plot PVE of each principal components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Left plot
ax1.plot(pca2.explained_variance_ratio_, '-o')
ax1.set_ylabel('Proportion of Variance Explained')
ax1.set_ylim(ymin=-0.01)

# Right plot
ax2.plot(np.cumsum(pca2.explained_variance_ratio_), '-ro')
ax2.set_ylabel('Cumulative Proportion of Variance Explained')
ax2.set_ylim(ymax=1.05)

for ax in fig.axes:
    ax.set_xlabel('Principal Component')
    ax.set_xlim(-1, 65)
