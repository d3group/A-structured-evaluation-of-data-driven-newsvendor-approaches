import pandas as pd
import time
import numpy as np
from joblib import Parallel, delayed
from ddop.metrics import average_costs, prescriptiveness_score
from ddop.newsvendor import SampleAverageApproximationNewsvendor
from ddop.newsvendor import DecisionTreeWeightedNewsvendor
from ddop.newsvendor import RandomForestWeightedNewsvendor 
from ddop.newsvendor import KNeighborsWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import GaussianWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import DeepLearningNewsvendor
from sklearn.model_selection import ParameterGrid, GridSearchCV, KFold, cross_val_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

#####################################################################################
###                                    SETTINGS                                    ##
#####################################################################################

# define jobs for multiprocessing
n_jobs = 1

# define cv strategy
n_splits = 10
cv = KFold(n_splits=n_splits)

# define estimator tuples test
estimator_tuple_list = []
estimator_tuple_list.append(('SAA', SampleAverageApproximationNewsvendor()))
estimator_tuple_list.append(('LR', LinearRegressionNewsvendor()))
estimator_tuple_list.append(('DTW', DecisionTreeWeightedNewsvendor(random_state=1)))
estimator_tuple_list.append(('RFW', RandomForestWeightedNewsvendor(random_state=1)))
estimator_tuple_list.append(('KNNW',KNeighborsWeightedNewsvendor()))
estimator_tuple_list.append(('KW', GaussianWeightedNewsvendor()))
estimator_tuple_list.append(('DL', DeepLearningNewsvendor(random_state=1)))

# define feature categories
feature_cat_dict = {
    "calendar": ['weekday', 'month', 'year'],
    "lag": ['demand__sum_values_7', 'demand__median_7',
       'demand__mean_7', 'demand__standard_deviation_7', 'demand__variance_7',
       'demand__root_mean_square_7', 'demand__maximum_7',
       'demand__absolute_maximum_7', 'demand__minimum_7',
       'demand__sum_values_14', 'demand__median_14', 'demand__mean_14',
       'demand__standard_deviation_14', 'demand__variance_14',
       'demand__root_mean_square_14', 'demand__maximum_14',
       'demand__absolute_maximum_14', 'demand__minimum_14',
       'demand__sum_values_28', 'demand__median_28', 'demand__mean_28',
       'demand__standard_deviation_28', 'demand__variance_28',
       'demand__root_mean_square_28', 'demand__maximum_28',
       'demand__absolute_maximum_28', 'demand__minimum_28'],
    "special_yaz": ['is_holiday', 'is_closed', 'wind', 'clouds', 'rain', 'sunshine', 'temperature'],
    "special_m5": ['is_sporting_event', 'is_cultural_event', 'is_national_event',
                   'is_religious_event', 'is_snap_day'],
    "special_bakery": ['is_schoolholiday', 'is_holiday',
                       'is_holiday_next2days', 'rain', 'temperature', 'promotion_currentweek',
                       'promotion_lastweek']}

# define all datasets to run with the corresponding feature categories
dataset_dict = {
    "m5": [["calendar"], ["calendar", "lag"],["calendar", "lag", "special_m5"]],
    "SID": [["calendar"], ["calendar", "lag"]],
    "yaz": [["calendar"], ["calendar", "lag"], ["calendar", "lag", "special_yaz"]],
    "bakery": [["calendar"], ["calendar", "lag"], ["calendar", "lag", "special_bakery"]]
}

# define under- and overage costs
cu = [9, 7.5, 5, 2.5, 1]
co = [1, 2.5, 5, 7.5, 9]

#------------------------------------------------------------------------------------

# define grids
def get_grid(estimator_name, n_features):
    if estimator_name == "DTW":
        grid = {
            "max_depth":[None,2,4,6,8,10],
            "min_samples_split": [2,4,6,8,16,32,64]
        }
        
    elif estimator_name == "RFW":
        grid = {
            "max_depth":[None,2,4,6,8,10],
            'min_samples_split':[2,4,6,8,16,32,64],
            'n_estimators':[10,20,50,100]}
        
    elif estimator_name == "KNNW":
        grid = {'n_neighbors':[1,2,4,8,16,32,64,128]}
        
    elif estimator_name == "GKW":
        grid = {'kernel_bandwidth':[*np.arange(0.5, int(np.sqrt(n_features/2))+0.25, 0.25)]}
        
    elif estimator_name == "DL":
        grid = {"optimizer": ["adam"],
                "neurons": [
                    (round(0.5*n_features),round(0.5*0.5*n_features)),
                    (round(0.5*n_features),round(0.5*1*n_features)),
                    (1*n_features,round(1*0.5*n_features)),
                    (1*n_features,1*1*n_features),
                    (2*n_features,round(2*0.5*n_features)),
                    (2*n_features,2*1*n_features),
                    (3*n_features,round(3*0.5*n_features)),
                    (3*n_features,3*1*n_features)],
                "epochs": [10,100,200]}
    else:
        grid = None
        
    return grid


def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def get_wsaa_sl_scores(X, y, params, cu, co, estimator, cv):
    estimator.set_params(**params)
    scores = []
    sl_scores = []
    for train_index, test_index in cv.split(X):
        estimator.fit(X[train_index],y[train_index])
        for cu_i, co_i in zip(cu,co):
            estimator.cu = [cu_i]
            estimator.co = [co_i]
            estimator.cu_ = [cu_i]
            estimator.co_ = [co_i]
            score = estimator.score(X[test_index],y[test_index])
            scores.append(score)
        sl_scores.append(scores)
        scores = []

    return sl_scores


def get_wsaa_results(group, X_train, X_test, y_train, y_test, param_grid, cu, co, estimator, estimator_name, cv, 
                     dataset, feature_combi, scaler_target, n_jobs):
    
    cv_results = pd.DataFrame()
    best_results = pd.DataFrame()

    candidate_params = list(ParameterGrid(param_grid))
        
    parallel = Parallel(n_jobs=n_jobs)
    scores = parallel(delayed(get_wsaa_sl_scores)(X_train, y_train, params, cu, co, 
                                                  estimator, cv) for params in candidate_params)

    scores = np.array(scores)
    mean_scores = scores.mean(axis=1)
    rank_scores = mean_scores.argmax(axis=0)
    best_scores = mean_scores.max(axis=0)
    best_params = [candidate_params[rank] for rank in rank_scores]

    for i in range(len(cu)):
        cv_results_temp = pd.DataFrame(scores.T[i].T, columns = ['split'+str(split)+'_test_score' for split 
                                                                 in range(n_splits)])
        cv_results_temp["mean_test_score"] = mean_scores.T[i]
        cv_results_temp["dataset"] = dataset
        cv_results_temp["feature combi"] = str(feature_combi)
        cv_results_temp["group"] = str(group)
        cv_results_temp["model"] = estimator_name
        cv_results_temp["cu"] = cu[i]
        cv_results_temp["co"] = co[i]
        cv_results_temp["sl"] = cu[i]/(cu[i]+co[i])
        cv_results_temp["params"] = candidate_params
        cv_results = pd.concat([cv_results, cv_results_temp], ignore_index=True)

    for i in range(len(cu)):
        best_estimator = clone(estimator).set_params(**best_params[i])
        best_estimator.set_params(cu=cu[i],co=co[i])
        best_estimator.fit(X_train, y_train)
        pred = best_estimator.predict(X_test)
        pred = scaler_target.inverse_transform(pred)
        avg_costs = average_costs(y_test,pred,cu[i],co[i])
        avg_costs = round(avg_costs,4)
        saa_pred = SampleAverageApproximationNewsvendor(cu[i],co[i]).fit(y_train).predict(X_test.shape[0])
        saa_pred = scaler_target.inverse_transform(saa_pred)
        SoP = prescriptiveness_score(y_test, pred, saa_pred, cu[i], co[i])
        SoP = round(SoP,4)

        d = {'dataset': dataset, 'feature combi': feature_combi, 'group': str(group), 'model': estimator_name, 'cu': cu[i],
             'co': co[i], 'sl': cu[i]/(cu[i]+co[i]), 'average costs': avg_costs, 'coefficient of prescriptiveness': SoP, 
             'best params': best_params[i]}

        best_results = best_results.append(d, ignore_index=True)
    
    return cv_results, best_results


def get_model_results(group, X_train, X_test, y_train, y_test, param_grid, cu, co, estimator, estimator_name, 
                      cv, dataset, feature_combi, scaler_target):
    
    cv_results = pd.DataFrame()
    best_results = pd.DataFrame()
    
    for cu_i, co_i in zip(cu,co):
        
        base_estimator = clone(estimator)
        base_estimator.set_params(cu=cu_i,co=co_i)
        
        if estimator_name in ["LR", "SAA"]:
            
            #cv_scores = cross_val_score(estimator, y_train, cv=cv)
            cv_scores = cross_val_score(base_estimator, X=X_train, y=y_train, cv=cv)
            cv_results_temp = pd.DataFrame([cv_scores], columns = ['split'+str(split)+'_test_score' for split 
                                                                   in range(10)])
            cv_results_temp["mean_test_score"] = cv_scores.mean()
            best_estimator = base_estimator
            best_estimator.fit(X=X_train, y=y_train)
            params = np.nan
            best_params = np.nan
        
        else:
            gs = GridSearchCV(base_estimator, param_grid, cv=cv)
            gs.fit(X_train,y_train)
            cv_results_temp = pd.DataFrame({k: v for k, v in gs.cv_results_.items() if k.startswith('split') or 
                                            k == 'mean_test_score'})
            best_estimator = gs.best_estimator_
            params = gs.cv_results_["params"]
            best_params = gs.best_params_
            
            
        cv_results_temp["dataset"] = dataset
        cv_results_temp["feature combi"] = str(feature_combi)
        cv_results_temp["group"] = str(group)
        cv_results_temp["model"] = estimator_name
        cv_results_temp["cu"] = cu_i
        cv_results_temp["co"] = co_i
        cv_results_temp["sl"] = cu_i/(cu_i+co_i)
        cv_results_temp["params"] = params
        cv_results = pd.concat([cv_results, cv_results_temp], ignore_index=True)

        if estimator_name == "SAA":
            pred = best_estimator.predict(X_test.shape[0])
            
        else:
            pred = best_estimator.predict(X_test)
            
        pred = scaler_target.inverse_transform(pred)
        avg_costs = average_costs(y_test,pred,cu_i,co_i)
        avg_costs = round(avg_costs,4)        

        saa_pred = SampleAverageApproximationNewsvendor(cu_i,co_i).fit(y_train).predict(X_test.shape[0])
        saa_pred = scaler_target.inverse_transform(saa_pred)
        SoP = prescriptiveness_score(y_test, pred, saa_pred, cu_i, co_i)
        SoP = round(SoP,4)

        d = {'dataset': dataset, 'feature combi': feature_combi, 'group': str(group), 'model': estimator_name, 'cu': cu_i,
             'co': co_i, 'sl': cu_i/(cu_i+co_i), 'average costs': avg_costs, 'coefficient of prescriptiveness': SoP, 
             'best params': best_params}
        best_results = best_results.append(d, ignore_index=True)
        
    return cv_results, best_results


def get_results(group, X, y, cu, co, estimator_tuple_list, cv, dataset, feature_combi):
    
    cv_results = pd.DataFrame()
    best_results = pd.DataFrame()
    
    X = X.get_group(group)
    y = y.iloc[X.index.values.tolist()]

    X = X.drop(["store", "item"], axis=1)
    n_features = len(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=False)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #scale target variable
    scaler_target = StandardScaler()
    scaler_target.fit(y_train)
    y_train = scaler_target.transform(y_train).ravel()
    
    for estimator_tuple in estimator_tuple_list:
        
        estimator_name = estimator_tuple[0]
        estimator = estimator_tuple[1]
        param_grid = get_grid(estimator_name, n_features)
        
        if estimator_name in ["KNNW", "RFW", "DTW", "GKW"]:
            cv_results_temp, best_results_temp = get_wsaa_results(group, X_train, X_test, y_train, y_test, param_grid, 
                                                                  cu, co, estimator, estimator_name, cv, dataset,
                                                                  feature_combi, scaler_target, n_jobs=1)
        
        else:
            cv_results_temp, best_results_temp = get_model_results(group, X_train, X_test, y_train, y_test, param_grid, 
                                                                   cu, co, estimator, estimator_name, cv, dataset,
                                                                   feature_combi, scaler_target)
        
        cv_results = pd.concat([cv_results, cv_results_temp], ignore_index=True)
        best_results = pd.concat([best_results, best_results_temp], ignore_index=True)
        
        logger.info('Group {}'.format(group))
        
    return cv_results, best_results


if __name__ == "__main__":
    
    # set a logger file
    logger = log(path="", file="results.logs")

    for dataset in dataset_dict:

        X = pd.read_csv("Data/final/"+dataset+"_data.csv.zip")
        y = pd.read_csv("Data/final/"+dataset+"_target.csv.zip")

        cv_results = pd.DataFrame()
        best_results = pd.DataFrame()

        for feature_combi in dataset_dict[dataset]:
            
            logger.info("------------------------------------------------------------------")
            logger.info('Dataset {}; Feature Combi {}'.format(dataset, feature_combi))
            
            cols = []
            for feature_cat in feature_combi:
                cols = cols + feature_cat_dict[feature_cat]

                X_cols = X[cols+["store", "item"]]

            X_cols = pd.get_dummies(X_cols, columns=["weekday", "month"])

            X_grouped = X_cols.groupby(["store", "item"])
            groups = list(X_grouped.groups.keys())

            parallel = Parallel(n_jobs=n_jobs)
            results = parallel(delayed(get_results)(group, X_grouped, y, cu, co, estimator_tuple_list, cv, dataset,
                                                    feature_combi) for group in groups)

            for result in results:
                cv_results = pd.concat([cv_results, result[0]], ignore_index=True) 
                best_results = pd.concat([best_results, result[1]], ignore_index=True)

            cv_results.to_csv("Results/cv_results_"+dataset+".csv", index=False)
            best_results.to_csv("Results/best_results_"+dataset+".csv", index=False)