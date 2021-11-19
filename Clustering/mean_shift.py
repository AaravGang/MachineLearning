import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from sklearn import preprocessing
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from Clustering.load_data import load_titanic


# visualisation example
def cluster_points():
    # centroids = np.array([[1, 1, 5], [10, 10, -10], [-3, -7, 9]]) # 3d points
    centroids = np.array([[1,1],[-10,9],[12,2]])
    X,_ = make_blobs(n_samples=100,centers=centroids,cluster_std=1.8)
    
    clf = MeanShift()
    clf.fit(X)
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_
    
    n_clusters = len(cluster_centers)
    print("Clusters identified: ",n_clusters)
    
    print("Correct centroids: ",centroids, "vs.", "predicted: ",cluster_centers)
    
    colors = ["g", "b", "y", "k", "m", "r"]
    
    fig = plt.figure()
    # ax = fig.add_subplot(projection="3d") # 3d
    ax = fig.add_subplot()
    for i,features in enumerate(X):
        ax.scatter(*features,c=colors[labels[i]],marker="o")
        
    for center in cluster_centers:
        ax.scatter(*center, c="k", marker="X", s=150,zorder=10)
        
    plt.show()
    

# cluster the titanic stokes_data set
def cluster_titanic():
    original_df,_ = load_titanic(convert_to_numeric=False)
    df, prediction_col = load_titanic(drop=["name", "body", "ticket", "boat"])
    
    X = np.array(df.drop(columns=[prediction_col]))
    X = preprocessing.scale(X)
    y = np.array(df[prediction_col])
    
    # clustering is an unsupervised learning algorithm
    # this is a hierarchical clustering algorithm,
    # it will find the number of clusters also on its own
    clf = MeanShift()
    clf.fit(X)
    
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_
    n_clusters = len(cluster_centers)
    
    original_df["cluster_group"] = [ label for label in labels]
    # print(original_df.head())
    
    survival_rates = {}
    for i in range(n_clusters):
        temp_df = original_df[(original_df["cluster_group"]==i)]
        print(temp_df.describe())
        print("#"*100)
        survived = temp_df["survived"].values.tolist().count(1)
        survival_rates[i]=survived/len(temp_df)
        
    print(survival_rates)
    


cluster_titanic()
cluster_points()
