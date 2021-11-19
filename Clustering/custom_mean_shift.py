import numpy as np
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style



class MeanShift:
    def __init__(self, radius=None, num_steps=100, tol=None):
        self.radius = radius
        self.num_steps = num_steps
        self.tol = tol
    
    def fit(self, data):
        # stokes_data is expected to be a numpy array
        if self.radius is None:  # figure out an appropriate radius
            all_data_center = np.average(data, axis=0)  # find the average point of all the stokes_data points
            dist_from_origin = np.linalg.norm(all_data_center)
            self.radius = dist_from_origin / self.num_steps  # a good way to approx. the radius
            self.tol = self.tol or self.radius # if a tolerance has been set, then use that otherwise use the radius
        
        # create a list full of weights, the closer a point is to a centroid the higher it is weighted
        weights = [i for i in range(self.num_steps+1, 1, -1)] # a nice reverse list [101,100,..1]
        centroids = data.copy()
        
        while True:
            new_centroids = []
            for centroid in centroids:
                in_bandwidth = [] # what all stokes_data points are within a the radius of this centroid
                for features in data:
                    distance = np.linalg.norm(centroid - features) # calculate the distance
                    weight_ind = int(distance / self.radius) # calculate the index of the weights, to which this will belong to
                    if weight_ind < self.num_steps: # if this index is bigger than the weights list, then skip it
    
                        # we want to weight the closer stokes_data points more heavily
                        # so if a stokes_data point is [2,3], we add that stokes_data point the number of times,
                        # as calculated by getting the weight index
                        # so if weight index is 5, and the value at that index is 96
                        # we add [2,3] to the points in bandwidth 96 times!
                        in_bandwidth.extend(weights[weight_ind]*[features])
                        
                # now we have all the points that are in bandwidth,
                # weighted accordingly, We can calculate the new centroid
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid)) # we add a tuple cuz, set doesn't like numpy arrays
                
            # at this point all new centroids have been calculated
            # so let's get rid of the ones that are over lapping,
            new_centroids = list(set(new_centroids))
            # and keep only one for all those that are very close to each other
            # this is my way of getting rid of those close points, without raising errors
            for c in range(len(new_centroids)-1,-1,-1):
                for j in range(len(new_centroids)-1,c,-1):
                    if np.linalg.norm(np.array(new_centroids[j])-np.array(new_centroids[c]))<=self.tol:
                        new_centroids.pop(c)
                        
                        
            prev_centroids = centroids.copy() # make a copy of the centroids to compare later
            centroids = np.array(new_centroids) # convert the centroids to an np array, cuz we had converted them to tuple before
            
            if np.array_equal(centroids, prev_centroids): # if both the previous and the current centroids are the same, then we are done
                self.centroids = centroids
                break
           
        # calculate the labels for all the stokes_data points
        self.labels = []
        for features in data:
            # get the centroid closest to the stokes_data point and return its id
            distances = [np.linalg.norm(features - centroid) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.labels.append(classification)
            
    def predict(self,feature_set):
        # get the centroid closest to the stokes_data point and return its id
        distances = [np.linalg.norm(feature_set - centroid) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        
        
# visualisation example
def cluster_points():
    centroids = np.array([[1, 1], [-10, 9], [12, 2]])
    X, _ = make_blobs(n_samples=100, centers=centroids)
    
    clf = MeanShift(num_steps=100,radius=None,tol=None)
    clf.fit(X)
    
    print("radius: ",clf.radius)
    
    labels = clf.labels
    cluster_centers = clf.centroids
    print(centroids, "####", cluster_centers)
    
    colors = ["g", "b", "y", "k", "m", "r"]*10
    
    for i, features in enumerate(X):
        plt.scatter(*features, c=colors[labels[i]], marker="o")
    
    for center in cluster_centers:
        plt.scatter(*center, c="k", marker="X", s=150)
    
    plt.show()
    
    
cluster_points()
                