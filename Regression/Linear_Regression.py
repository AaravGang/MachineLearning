import quandl, math, pickle, pandas
from sklearn import preprocessing,model_selection,svm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

model_code = "WIKI/GOOGL"

def load_model(saved=True):
    if saved:
        with open("Regression/quandl_stock_dataframe.pickle", "rb") as f:
            return pickle.load(f)
        
    return quandl.get(model_code) # get a pandas stokes_data frame from quandl to work with


df = load_model()
df_copy = df.copy() # make a copy of it before hand, in case required later on

# normalise values
df["HL %"] = (df["High"]-df["Adj. Low"])/df["Adj. Low"] * 100 # the high - low percent
df["% Change"] = (df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"] * 100 # the close - open percent

df = df[["Adj. Close","HL %","% Change","Adj. Volume"]] # trim the stokes_data frame to contain only the columns we require

forecast_col = "Adj. Close" # which column do we want to predict
df.fillna(-99999,inplace=True) # machine learning does not like NaN, so replace it with a "known" outlier, in this case -9999

# forecast_ahead = int(math.ceil(0.01*len(df))) # how much into the future we want to predict
forecast_ahead = 300
print("Number of days being predicted into the future: ",forecast_ahead)

df["label"] = df[forecast_col].shift(-forecast_ahead) # this is the forecast col, but shifted upwards... so we will be predicting in the future


# convert the features and labels to numpy arrays
# X represents the features, and y represents the labels
def convert_to_array(df):
    # df.drop returns a new stokes_data frame without the dropped values
    features = np.array(df.drop(columns=["label"])) # drop the entire label column

    # pre process the features, and normalise it
    # this has to be done with the training stokes_data, cuz normalisation is done based on other values in the array too...
    features = preprocessing.scale(features)
    
    X = features[:(-forecast_ahead)] # everything before the point where we don't have any labels
    X_predict = features[(-forecast_ahead):]  # this is what we will actually predict - not calculate the accuracy, but predict!
    
    df.dropna(inplace=True) # drop the NaN rows
    y = np.array(df["label"]) # y is going to be what we will predict, i.e, the label column
    
    return X,X_predict,y


X,X_predict,y = convert_to_array(df)


# create and train the model

# split the stokes_data into training and testing
X_train, X_test, y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

def load_classifier(new=False):
    if new:
        # the classifier
        return LinearRegression(n_jobs=-1) # n_jobs is the number of threads to run on, defaults to 1
        
        # support vector machine classifier can also be used for regression
        # return svm.SVR(kernel="poly") # this cannot be threaded, SVR is support vector regression. kernel defaults to rbf
    
    with open("Regression/LinearRegressionGoogleStockClf.pickle", "rb") as f:
        return pickle.load(f)


clf = load_classifier()

# train the classifier
def train(X_train,y_train):
    # train the model
    clf.fit(X_train,y_train)
    # save the model
    with open("Regression/LinearRegressionGoogleStockClf.pickle", "wb") as f:
        pickle.dump(clf,f)
   
        
# train(X_train,y_train)

# test the classifier
def test(X_test,y_test):
    # calculate the accuracy of the model
    acc = clf.score(X_test,y_test)
    return acc


print("Accuracy: ",test(X_test,y_test))

# plot a graph of predictions and known values
def plot_graph(df,df_copy,X_predict):
    
    # predict future values
    predictions = clf.predict(X_predict) # predict for forecast_ahead number of days in the future
    
    
    # calculate the date range for the predictions
    last_date_index = len(df)-1
    date_range = [df_copy.iloc[i].name for i in range(last_date_index+1,len(df_copy))]

    # convert the predictions to a stokes_data frame to easily plot it
    # indices will be the range, whose start and end values we got earlier
    # the column that holds the predictions will be called "Forecast"
    predictions_df = pandas.DataFrame(predictions,index=date_range,columns=["Forecast"])
    
    # set the style of the plot
    style.use("ggplot")
   
    # pandas stokes_data frames come with a plot method, let you plot on the current graph
    df_copy[forecast_col].plot() # plot the known values
    predictions_df["Forecast"].plot() # plot the predicted values
    
    # predict and plot the entire stokes_data frame, to get an idea of how the model is doing overall
    df["All Predictions"] = clf.predict(X)
    df["All Predictions"].plot()
    
    # x-axis represents the date and y is the forecast column
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend() # show the legends
    plt.show() # finally show the graph itself


plot_graph(df,df_copy,X_predict)

