from pandas import DataFrame
import random
import numpy as np

def custom_train_test_split(df: DataFrame, group_col="class", test_size=0.2):
    full_df = df.astype(float).values  # get the values of the df, in array format
    random.shuffle(full_df)  # shuffle them in place
    
    train_data = full_df[
                 :-int(test_size * len(full_df))]  # make the training stokes_data the first 1-test_size% of the whole df
    test_data = full_df[-int(test_size * len(full_df)):]  # make the training stokes_data the last test_size %
    
    groups = set(df[group_col])  # get all the groups to classify into
    group_col_ind = list(df.columns).index(
        group_col)  # what was the index of the column, this will be useful while predicting
    
    # convert all the training stokes_data to a dict, with the keys as the group name
    # and values as an array of what all stokes_data points fall in that group
    train_set = {grp: [] for grp in groups}
    test_set = {grp: [] for grp in groups}
    
    for features in train_data:
        train_set[features[group_col_ind]].append(features[:group_col_ind])
    
    for features in test_data:
        test_set[features[group_col_ind]].append(features[:group_col_ind])
    
    return train_set, test_set  # that's it!


def handle_non_numeric_data(df):
    df = df.copy()
    columns = df.columns
    for col in columns:
        if df[col].dtype !=np.int and df[col].dtype !=np.float:
            # then we know that we have non numeric stokes_data
            unique_elements = set(df[col]) # get all the unique elements in the column
            vals = {el:id for id,el in enumerate(unique_elements)} # assign each unique element an id
            df[col] = list(map(lambda el:vals[el],df[col])) # change all elements in that column to be their corresponding id
            
    return df

