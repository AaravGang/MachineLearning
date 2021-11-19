import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import random




class LinearRegression:
    def __init__(self):
        self.xs,self.ys,self.b,self.m,self.best_fit_line,self.acc = None,None,None,None,None,None
        

    # train the model
    def train(self,xs,ys):
        self.xs = np.array(xs, dtype=np.float64)
        self.ys = np.array(ys, dtype=np.float64)
    
        self.m = self.calc_slope()
        self.b = self.calc_y_intercept()
    
        self.best_fit_line = self.predict(self.xs)
    
        self.acc = self.coeff_of_determination(self.ys, self.best_fit_line)
    
        print("Model accuracy: ", self.acc)
        
        return self.acc
        
    
    # function to predict the y values, from provided x values
    def predict(self,xs):
        xs = np.array(xs).flatten()
        ys = []
        for x in xs:
            ys.append(self.m * x + self.b)
        
        return np.array(ys)

    # squared error
    # find the difference between the y points in the stokes_data set and y points of a given line, square it and sum it!
    @staticmethod
    def squared_error(y_original,y_line):
        return sum((y_original-y_line)**2)
        
    @staticmethod
    def coeff_of_determination(ys,y_hat): # ys are the actual ys in the stokes_data set, and y_hat are the ys of the best fit line
        # calculating the accuracy / r^2
        SE_y_hat = LinearRegression.squared_error(ys, y_hat)  # get the squared error of the best fit line
        SE_y_mean = LinearRegression.squared_error(ys, [mean(ys)] * len(ys))  # get the squared error of the y mean line
    
        # this is the formula for r^2
        # we want SE_y_hat to be as low as possible, but everything is relative -
        # so we want SE of the best fit line / SE of the mean line, to be as low as possible, i.e;
        # we want that "COMPARISION" to be as low as possible
        # and then we subtract that from 1, cuz... 1-low = high
        r_squared = (1 - (SE_y_hat / SE_y_mean))
        return r_squared
    
    
    # test with some provided stokes_data points
    # more the number of stokes_data points - higher the accuracy this will return, so biased in some way
    def test(self,xs,ys):
        xs = np.array(xs)
        predictions = self.predict(xs)
        ys = np.array(ys)
        acc = self.coeff_of_determination(ys,predictions)
        print("Accuracy: ",acc)
        
        self.plot(xs,ys,predictions)
    
        return acc
      
    
    # calculate the slope or the m
    def calc_slope(self):

        numerator = (mean(self.xs)*mean(self.ys)) - mean(self.xs*self.ys)
        denominator = (mean(self.xs)**2) - mean(self.xs**2)
    
        return numerator/denominator
    
    # calculate the y-intercept or the b
    def calc_y_intercept(self):
        return mean(self.ys) - (self.m*mean(self.xs))
    
    
    # plot a graph of the stokes_data points and best fit line
    def plot(self,xs=None,ys=None,line=None):
        if xs is not None and ys is not None and line is not None:
            plt.scatter(xs, ys,c="g",label="test points")
            plt.plot(xs, line,c="r",label="line for test points")
        
        else:
            plt.scatter(self.xs,self.ys,label="train points",c="b")
            plt.plot(self.xs,self.best_fit_line,label="best fit line",c="y")
  
  
def create_data_set(n=400,max_deviation=50,step=3):
    ys = []
    val = 0
    for _ in range(n):
        ys.append(val+random.uniform(-max_deviation, max_deviation))
        val+=step
        
    xs = [i for i in range(n)]
    
    return xs,ys
  
  
def train_test_split(X,y,test_size=0.2):
    test_size = int(len(X)*test_size)
    return X[:-test_size],X[-test_size:],y[:-test_size],y[-test_size:]
  
 
 
def sample_test():
    X,y = create_data_set()
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    
    clf = LinearRegression()
    clf.train(X_train,y_train)
    clf.test(X_test,y_test)
    
    style.use('fivethirtyeight')
    clf.plot()
    plt.legend()
    plt.show()
  
  
if __name__ == '__main__':
    sample_test()