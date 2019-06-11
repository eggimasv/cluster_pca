# Principl component analysis example
# https://plot.ly/ipython-notebooks/principal-component-analysis/
import plotly.plotly as py
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()
print(df.tail())

# split data table into data X and class labels y
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

print("---")
print(X)
print("---")

# -----------plotting histograms
colors = {'Iris-setosa': '#0D76BF', 
          'Iris-versicolor': '#00cc96', 
          'Iris-virginica': '#EF553B'}

nr_rows = 4
nr_columns = 1
fig, axes = plt.subplots(nr_rows, nr_columns)

for colum_entry in range(nr_rows):
    plt_row = colum_entry

    col_name = df.columns[colum_entry]
    for key in colors:

        plt_data = list(X[y==key, colum_entry])
        color = colors[key]

        # the histogram of the data
        n, bins, patches = axes[plt_row].hist(
            x=plt_data, 
            bins=10,
            facecolor=color,
            alpha=0.75)

        axes[plt_row].set(
            xlabel=key,
            ylabel='Probability',
            title="Histogram with distribution of {}".format(col_name))
plt.show()

# Standardsie (because different measures)
X_std = StandardScaler().fit_transform(X)
print(X_std)



# ---------------------- Short version from package

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
print(".................")
print(Y_sklearn)

# plot Princial components
plt.scatter(x=Y_sklearn[:, 0], y=Y_sklearn[:, 1])
plt.show()

data = []

for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):
    entry = {
        'x': Y_sklearn[y==name, 0],
        'y': Y_sklearn[y==name, 1],
        'name': name,
        'color': col}


    data.append(entry)

fig, ax = plt.subplots()

for i in data:
    print("color: " + str(i['color']))
    ax.scatter(
        x=i['x'],
        y=i['y'],
        c=i['color'],
        alpha=0.5)

plt.show()


def cluster_pca():
    
    # Cluster principal components

    return


raise Exception("Finished short")


# ---------------------- Long full version (see online script)
'''
# Calculate covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#Short
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)
print('Covariance matrix \n%s' %cov_mat)


# we perform an eigendecomposition on the covariance matrix:
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Singular Vector Decomposition
u,s,v = np.linalg.svd(X_std.T)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

#In order to decide which eigenvector(s) can be dropped without
#losing too much information for the construction of lower-dimensional
#subspace, we need to inspect the corresponding eigenvalues: The eigenvectors
#with the lowest eigenvalues bear the least information about the distribution
#of the data; those are the ones can be dropped.
#In order to do so, the common approach is to rank the eigenvalues from highest
#to lowest in order choose the top k eigenvectors.

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


#After sorting the eigenpairs, the next question is "how many principal components are we
#going to choose for our new feature subspace?" A useful measure is the so-called "explained
#variance," which can be calculated from the eigenvalues. The explained variance tells us how
#much information (variance) can be attributed to each of the principal components.


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

fig, ax = plt.subplots()

x_values = range(1,5)

plt.bar(x_values, var_exp)

plt.scatter(
    x=x_values,
    y=cum_var_exp)
ax.plot(x_values,cum_var_exp)

plt.title('Explained variance by different principal components')
plt.show()


#Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace,
#  by choosing the "top 2" eigenvectors with the highest eigenvalues to construct our d×k-dimensional eigenvector matrix W.

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

#Projection Onto the New Feature Space
Y = X_std.dot(matrix_w)

data = []

for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):
    entry = {
        'x': Y[y==name, 0],
        'y': Y[y==name, 1],
        'name': name,
        'color': col}

    data.append(entry)

fig, ax = plt.subplots()

for i in data:
    print("color: " + str(i['color']))
    ax.scatter(
        x=i['x'],
        y=i['y'],
        c=i['color'],
        alpha=0.5)

plt.show()
'''

print("FINISHEd")
