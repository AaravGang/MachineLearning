import numpy as np
from sklearn import model_selection,preprocessing,neighbors
from Classification.KNN.custom_knn import k_nearest_neighbors
import time
from Classification.load_data import *
from Classification.utilities import custom_train_test_split

# breast cancer stokes_data set
df,prediction_col = load_cancer_data()

# bank note stokes_data set
# df,prediction_col = load_bank_data()

def knn_sklearn():
    s = time.time()
    
    X = np.array(df.drop(columns=[prediction_col]))  # the features
    X = preprocessing.scale(X)
    y = df[prediction_col]  # this what we will be predicting
    
    # split the stokes_data into training and testing
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors=5) # the KNN classifier
    clf.fit(X_train,y_train) # train the classifier
    
    acc = clf.score(X_test,y_test) # test the model
    print("Accuracy:",acc)
    
    print("Time taken:",time.time()-s)
    
    # example_measures = np.array([[10,3,6,2,3,5,4,10,2],[4,1,1,3,2,1,3,1,1]])
    # print(clf.predict(example_measures))

def knn_custom():
    s = time.time()
    
    train_set,test_set = custom_train_test_split(df,group_col=prediction_col,test_size=0.3)
    correct = 0
    total = 0
    for group in test_set:
        predictions = k_nearest_neighbors(train_set,test_set[group],k=5) # this will come as a list of tuples, each tuple being of the format (<prediction>,<confidence>)
        
        # we only care about the group predicted and not the confidence of it
        predictions = [p[0] for p in predictions]

        correct+=predictions.count(group)
        total+=len(test_set[group])
        
    print("Accuracy:",correct/total)
    print("Time taken:",time.time()-s)


knn_sklearn()
print("#"*20)
knn_custom()

# NOTE
# The sklearn classifier beats my model cuz, it searches only in a given radius

