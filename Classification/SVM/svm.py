from sklearn import model_selection,preprocessing,svm
import time
import pandas as pd
import numpy as np
from Classification.load_data import *
from Classification.utilities import custom_train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

# load the stokes_data
# df,prediction_col = load_bank_data()


data = {
   'x1':[-1,-5,0,3,4,9,3,5,10],
    'x2':[2,3,1,3,4,3,10,9,8],
    'class':[1,1,1,2,2,2,3,3,3]
    
}
df = pd.DataFrame(data)



def svm_sklearn(df,prediction_col):
    s = time.time()
    
    X = np.array(df.drop(columns=[prediction_col]))  # the features
    X = preprocessing.scale(X)
    y = df[prediction_col]  # this what we will be predicting
    
    # split the stokes_data into training and testing
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    print(len(X_train),len(X_test),len(y_train),len(y_test))
    
    clf =svm.SVC(C=1,kernel="poly",decision_function_shape="ovr")  # the SVM classifier
    clf.fit(X_train, y_train)  # train the classifier
    
    acc = clf.score(X_test, y_test)  # test the model
    print("Accuracy:", acc)
    
    print("Time taken:", time.time() - s)
    #
    # example_measures = np.array([[10,3,6,2,3,5,4,10,2],[4,1,1,3,2,1,3,1,1]])
    # print(clf.predict(example_measures))


svm_sklearn(df,'class')

plt.scatter(data['x1'],data['x2'])
plt.show()