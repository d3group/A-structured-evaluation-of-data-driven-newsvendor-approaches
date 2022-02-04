import pandas
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series


def add_lag_features(X, y, column_id, column_sort, feature_dict, time_windows):
    """
    Create lag features for y and add them to X

    Parameters:
    -----------
    X: pandas.DataFrame 
    feature matrix to which TS features are added.

    y: pandas.DataFrame, 
    time series to compute the features for.

    column_id: list, 
    list of column names to group by, e.g. ["shop","product"]. If set to None, 
    either there should be nothing to groupby or each group should be 
    represented by a separate target column in y. 

    column_sort: str,
    column name used to sort the DataFrame. If None, will be filled by an 
    increasing number, meaning that the order of the passed dataframes are used 
    as “time” for the time series.

    feature_dict: dict,
    dictionary containing feature calculator names with the corresponding 
    parameters

    time_windows : list of tuples, 
    each tuple (min_timeshift, max_timeshift), represents the time shifts for 
    ech time windows to comupute e.g. [(7,7),(1,14)] for two time windos 
    a) time window with a fix size of 7 and b) time window that starts with size
    1 and increases up to 14. Then shifts by 1 for each step. 
    """

    if column_id == None:
        X['id'] = 1

    else:
        X['id'] = X[column_id].astype(str).agg('_'.join, axis=1)

    if column_sort == None:
        X['time'] = range(X.shape[0])  

    else:
        X["time"] = X[column_sort]

    y["time"] = X["time"]
    y["id"] = X["id"]

    X = X.set_index(['id', 'time'])

    for window in time_windows:

        # create time series for given time window 
        df_rolled = roll_time_series(y, column_id="id", column_sort="time", 
                                     min_timeshift= window[0]-1, 
                                     max_timeshift= window[1]-1)

        # create lag features for given time window 
        df_features = extract_features(df_rolled, column_id="id", 
                                       column_sort="time",
                                       default_fc_parameters=feature_dict)

        # Add time window to feature name for clarification 
        feature_names = df_features.columns.to_list()
        feature_names = [name+"_"+str(window[1]) for name in feature_names]
        df_features.columns = feature_names

        # add features for given time window to feature matrix temp
        X = pandas.concat([X,df_features],axis=1)

    y = y.set_index(['id', 'time'])
    y_column_names = y.columns.to_list()

    df = pandas.concat([X,y],axis=1)
    df = df.dropna()
    df = df.reset_index(drop=True)

    y = df[y_column_names]
    X = df.drop(y_column_names, axis=1)

    return X, y

def month_to_string(x):
    if x==1:
        return 'JAN'
    elif x==2:
        return 'FEB'
    elif x==3:
        return 'MAR'
    elif x==4:
        return 'APR'
    elif x==5:
        return 'MAY'
    elif x==6:
        return 'JUN'
    elif x==7:
        return 'JUL'
    elif x==8:
        return 'AUF'
    elif x==9:
        return 'SEP'
    elif x==10:
        return 'OCT'
    elif x==11:
        return 'NOC'
    else:
        return 'DEC'
    
def day_to_string(x):
    if x==1:
        return 'MON'
    elif x==2:
        return 'TUE'
    elif x==3:
        return 'WED'
    elif x==4:
        return 'THU'
    elif x==5:
        return 'FRI'
    elif x==6:
        return 'SAT'
    else:
        return 'SUN'