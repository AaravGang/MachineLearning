import pandas as pd
import os
from Clustering.utilities import handle_non_numeric_data


# load the stokes_data
# 2 stokes_data sets

# 1)
# breast cancer stokes_data from UCI
# predict if the cancer type is benign or malignant
# 2 - benign, 4 - malignant


def load_cancer_data():
    df = pd.read_csv(os.path.join("data_sets","breast-cancer.csv"))
    df.drop(columns=["id"],inplace=True) # we do not need id, as id does not affect the class.
    df.replace("?",0,inplace=True) # the missing values are represented with ?, replace them with a definite outlier
    prediction_col = "class"
    return df,prediction_col

# bank note stokes_data set
# predict whether a client has subscribed for a term deposit
def load_bank_data():
    dataset = pd.read_csv(os.path.join("data_sets","bank-additional.csv"), sep=';')
    dataset.drop(columns=['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','day_of_week','contact'],inplace=True)
    dataset = handle_non_numeric_data(dataset)
    prediction_col = "y"
    return dataset,prediction_col

# titanic stokes_data set
def load_titanic(drop=None, convert_to_numeric=True):
    df = pd.read_excel("data_sets/titanic.xls")
    if drop is not None:
        df.drop(columns=drop, inplace=True)
    df.fillna(0,inplace=True)
    
    if convert_to_numeric:
        df = handle_non_numeric_data(df)
        
    prediction_col = "survived"
    return df,prediction_col


