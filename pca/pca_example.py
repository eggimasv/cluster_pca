# Principl component analysis example
# https://plot.ly/ipython-notebooks/principal-component-analysis/
import plotly.plotly as py
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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

        data = list(X[y==key, colum_entry])
        color = colors[key]

        # the histogram of the data
        n, bins, patches = axes[plt_row].hist(
            x=data, 
            bins=10,
            facecolor=color,
            alpha=0.75)

        axes[plt_row].set(
            xlabel=key,
            ylabel='Probability',
            title="Histogram with distribution of {}".format(col_name))
plt.show()

print("FINISHEd")