import numpy as np
from collections import Counter
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib import style


def k_nearest_neighbors(data,to_predict,k=3):
    if k<len(data): # unrecommended value of k
        warnings.warn("k is less than the number of categories, idiot!")

    if k%2 == len(data.keys())%2: # unrecommended value of k
        warnings.warn("Both k and the number of categories are either both odd, or both even...")
     
    predict_single = False # by default just assume the user wants to predict multiple points at once

    try:
        iter(to_predict[0]) # check to see if the user has given multiple points
    except:
        to_predict = [to_predict]
        predict_single = True # if not, then set the predict single flag to true
        
    # this is expected to be a nested list, i.e, a list containing all the points to predict
    to_predict = np.array(to_predict) # convert to array, to make it easier to work with

    ret = [] # this what will be returned

    for data_point in to_predict:

        distances = [] # store all distances in a list and sort them to find the best k
        for group in data: # stokes_data is expected to be a dict or a similar type
            for features in data[group]: # get every stokes_data point for each individual group, and find euclidean distance
                # features are the "co ordinates" of the stokes_data point
                # so just find the euclidean distance now
                euclidean_distance = np.linalg.norm(features-data_point)
                distances.append((euclidean_distance,group))
                
        # all distances have been found, get the lowest k
        # we don't really care what the distance was once we get the lowest k, so just grab their groups
        groups = [d[1] for d in sorted(distances)[:k]] # python by default sorts tuples inside lists, by their first index - so by their distances in this case
        
        votes = Counter(groups).most_common() # this returns most common elements in the list in order
        best = votes[0] # will be in the format - (<category>, <count>)
        conf = best[1]/k # calculate the confidence
        ret.append((best[0],conf)) # (<group_classified_into>,<confidence>)
        
    return ret[0] if predict_single else ret # return the group and confidence


if __name__ == '__main__':
    dataset = {"g": [(1, 2), (2, 3), (3, 4)], "r": [(30, 40), (10, 20), (15, 10)]}
    # dataset = pd.DataFrame(dataset)

    new_features = [[100, 0],[0,100],[23,45],[10,10],(0,1)]
     
    classifications= k_nearest_neighbors(dataset,new_features)
    print(classifications)
    
    style.use("fivethirtyeight")
    [[plt.scatter(*p,c=group,s=100) for p in dataset[group]] for group in dataset]
    [plt.scatter(*new_features[i],c=grp,s=200) for i,(grp,conf) in enumerate(classifications)]
    plt.show()


