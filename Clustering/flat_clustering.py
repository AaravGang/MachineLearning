# K Means clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from Clustering.load_data import *



def cluster_titanic():
    df,prediction_col = load_titanic(drop=["name","body","ticket","boat"])
    
    X = np.array(df.drop(columns=[prediction_col]))
    X = preprocessing.scale(X)
    y = np.array(df[prediction_col])
    
    
    # clustering is an unsupervised learning algorithm
    clf = KMeans(n_clusters=2) # this is a flat clustering algorithm, so we need to specify how many cluster are there
    clf.fit(X)
    predictions = clf.labels_ # unsupervised algo, so there is no need to test
    
    correct = 0
    for i in range(len(y)):
        if y[i]==predictions[i]:
            correct+=1
    
    # sometimes the accuracy may be flipped, i.e, 0 in our stokes_data set might represent deaths,
    # but the classifier doesn't really care and arbitrarily picks a number
    # so an acc of 0.2 may actually be 0.8
    print("Accuracy: ",correct/len(y))


def cluster_points():
    centroids = np.array([[1, 1,5], [10, 10,-10], [-3, -7,9]])
    X, _ = make_blobs(n_samples=100, centers=centroids,cluster_std = 1.5)
    
    clf = KMeans(n_clusters=3)
    clf.fit(X)
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_ # centroids
    print(centroids,"####",cluster_centers)
    
    colors = ["g", "b", "y", "k", "m", "r"]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, features in enumerate(X):
        ax.scatter(*features, c=colors[labels[i]], s=100,marker="o")
    
    for center in cluster_centers:
        ax.scatter(*center, c="k", marker="X", s=150)
    
    plt.show()
    
    
cluster_titanic()
cluster_points()