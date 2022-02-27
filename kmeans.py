import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns

def kmeans_algorithm(data, clusters=2, max_iter=10):
    """
    Kmeans algorithm implementation for unsupervised learning.
    
    parameters:
    -----------
    data: pandas dataframe or numpy ndarray
        data points with n dimension without labels
    clusters: int
        number of clusters
    max_iter: int
        maximum number of iterations
    
    returns: labels and centroids
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    no_data_points = data.shape[0]
    # Initialize centers randomly
    centroid = data[np.random.choice(no_data_points, clusters, replace=False)]
    labels = np.zeros(no_data_points)
    
    for i in range(max_iter):
        # previous_labels = labels
        # Compute closest center for each point
        dist = cdist(data, centroid)
        labels = np.argmin(dist, axis=1)
        # Update centers to be the mean of the points closest to it
        for i in range(clusters):
            centroid[i, :] = data[labels == i].mean(axis=0)
        # Convergence criterion: break if all(previous_labels == labels) and use while loop True instead of for loop
        
    return labels, centroid

def visualization(df, centroid, labels):
    """
    Visualize centers and data points labels from Kmeans.
    
    Parameters:
    -----------
    df: pandas dataframe
        data points with n dimension with labels
    centroid: numpy ndarray
        centers of clusters
    labels: numpy ndarray
        labels of data points
    """
    df['labels'] = labels
    # print(df)
    sns.scatterplot(data=df, x=df.iloc[:,0].values, y=df.iloc[:,1].values, hue=df.labels)
    sns.scatterplot(x=centroid[:,0], y=centroid[:,1], color='black', marker='x')
    
df = pd.DataFrame(np.random.random((20,2)))
labels, centroid = kmeans_algorithm(df, clusters=2, max_iter=10)
visualization(df, centroid, labels)