import pandas as pd
import numpy as np
from util import preprocessing, get_X_and_label, accuracy_std 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib as mpb

'''Preprocessing data
standardize testing data using training data's mean and standard deviation
'''
data_train = pd.read_csv('D_train.csv')
data_test = pd.read_csv('D_test.csv')

data_train_new, new_label = preprocessing(data_train)
data_test_new, new_label = preprocessing(data_test)
scaler = StandardScaler()
data_train_new[new_label] = scaler.fit_transform(data_train_new[new_label].to_numpy())
data_test_new[new_label] = scaler.transform(data_test_new[new_label].to_numpy())


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.
    Thanks to this plot function from https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

##------------------------------------------------------------
'''Count how many datapoints are in Different Classes
1. Training dataset
2. Testing dataset
'''
df_class = data_train[['Class']].apply(pd.value_counts)
df_class = df_class.T
ax1 = df_class.plot.bar(rot=0)
ax1.title.set_text('Class type in training dataset')
add_value_labels(ax1)
##------------------------------------------------------------
df_class_test = data_test[['Class']].apply(pd.value_counts)
df_class_test = df_class_test.T
ax2 = df_class_test.plot.bar(rot=0)
ax2.title.set_text('Class type in testing dataset')
add_value_labels(ax2)
##------------------------------------------------------------

'''Applying PCA to show how data are distributing in 2D space
'''
pca = PCA(n_components=2)
train_X, train_label = get_X_and_label(data_train_new)
train_X = pca.fit_transform(train_X)
train_X
colors = ['red','green','blue','purple','yellow']
plt.scatter(x=train_X[:,0], y=train_X[:,1], c=train_label, cmap=mpb.colors.ListedColormap(colors))
plt.title(label = "Class distribution with PCA, reducing features into 2")

cb = plt.colorbar()
loc = np.arange(0,max(train_label),max(train_label)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels([1,4,5,3,2])
plt.show()