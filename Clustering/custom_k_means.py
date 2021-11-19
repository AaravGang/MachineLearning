import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from Clustering.load_data import *
from sklearn.datasets._samples_generator import make_blobs


class KMeans:
    def __init__(self,k=2,max_iter=300,tol=0.001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.classifications = []
        self.centroids = []
        self.labels = []
    
    def fit(self,data):
        # stokes_data is expected to be a numpy array for stokes_data points
        
        # initialise the centroids to be the first k stokes_data points
        self.centroids = np.array([data[i] for i in range(self.k)], dtype=np.float64)

        for i in range(self.max_iter):
            # reinitialise the classes to be empty lists
            self.classifications = [[] for _ in range(self.k)]
            self.labels.clear()
    
            for features in data:
                # find the distance b/w this stokes_data point and every centroid
                distances = [np.linalg.norm(centroid-features) for centroid in self.centroids]
                # get the least distance there is, and find its index
                # the index will be the same as the label of the class, we shall classify this feature set into
                min_dist_id = distances.index(min(distances))
                self.classifications[min_dist_id].append(features) # and sure enough, add that feature set to the corresponding index/ class-id
                self.labels.append(min_dist_id)
                
            # now that we have all the features classified, get the centroids/means of every class
            # but before that make a copy of the centroids right now, cuz we will be comparing them
            prev_centroids = self.centroids.copy()

            for c in range(self.k):
                # c is the key/id - points to the index here
                self.centroids[c] = np.average(self.classifications[c],axis=0)
              
            # check if an optimum value has been achieved for every centroid
            for c in range(self.k):
                original_centroid = prev_centroids[c]
                curr_centroid = self.centroids[c]

                # find the percent change
                if np.sum((curr_centroid-original_centroid)/original_centroid) > self.tol:
                    # optimum value hasn't been achieved for this centroid, so no point in checking others
                    break
                    
            else: # if optimum values have been achieved for ALL centroids:
                break # we do not want to do any more iterations, we have already found the centroids

    def predict(self, data):
        # get the centroid closest to the stokes_data point and return its id
        distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        
 
 
def cluster_titanic():
    df, prediction_col = load_titanic(drop=["name", "body", "boat"])
    
    X = np.array(df.drop(columns=[prediction_col]))
    y = np.array(df[prediction_col])
    
    clf = KMeans()
    clf.fit(X)

    labels = clf.labels # unsupervised algo, so there is no need to test
    
    correct = 0
    for i in range(len(y)):
        if y[i]==labels[i]:
            correct+=1

    # sometimes the accuracy may be flipped, i.e, 0 in our stokes_data set might represent deaths,
    # but the classifier doesn't really care and arbitrarily picks a number
    # so an acc of 0.2 may actually be 0.8
    print("accuracy: ",correct/len(y))
    
    
cluster_titanic()

# visualisation example
def cluster_points():
    centroids = np.array([[1, 1, 5], [10, 10, -10], [-3, -7, 9]])
    X, _ = make_blobs(n_samples=100, centers=centroids,cluster_std = 1.5)
    
    clf = KMeans(k=3)
    clf.fit(X)
    labels = clf.labels
    cluster_centers = clf.centroids
    print(centroids,"####",cluster_centers)
    
    colors = ["g", "b", "y", "k", "m", "r"]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, features in enumerate(X):
        ax.scatter(*features, c=colors[labels[i]],marker="o")
    
    for center in cluster_centers:
        ax.scatter(*center, c="k", marker="X", s=150,zorder=10)
    
    plt.show()


cluster_points()





