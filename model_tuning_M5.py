from sklearn.utils.validation import check_array
from sklearn.model_selection import ParameterGrid, GridSearchCV, RepeatedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from ddop.datasets import load_yaz, load_bakery, load_SID
from ddop.metrics import average_costs, prescriptiveness_score
from ddop.newsvendor import SampleAverageApproximationNewsvendor
from ddop.newsvendor import DecisionTreeWeightedNewsvendor
from ddop.newsvendor import RandomForestWeightedNewsvendor 
from ddop.newsvendor import KNeighborsWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import GaussianWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import DeepLearningNewsvendor
import numpy as np
import pandas as pd
import time
import os
import logging
import statistics
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.base import clone

# a function  to create and save logs in the log files
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


def get_sl_scores(X, y, params, cu, co, estimator, cv):
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


def main():
    
    # define file paths    
    log_path = ""
    result_path = ""
    
    # set a logger file
    logger = log(path=log_path, file="cross_val.logs")
    
    # load data (data available in team drive space)
    data = pd.read_csv("M5_final.csv")
    X = data.drop(["demand"], axis=1)
    y = pd.DataFrame(data["demand"])
    
    # group data 
    X_grouped = X.groupby(["store_id", "dept_id"])
    groups = list(X_grouped.groups.keys())
    
    n_features = len(X.columns)-2
    
    # define grids
    dl = {"optimizer": ["adam"],
          "neurons": [
                      (round(0.5*n_features),round(0.5*0.5*n_features)),
                      (round(0.5*n_features),round(0.5*1*n_features)),
                      (1*n_features,round(1*0.5*n_features)),
                      (1*n_features,1*1*n_features),
                      (2*n_features,round(2*0.5*n_features)),
                      (2*n_features,2*1*n_features),
                      (3*n_features,round(3*0.5*n_features)),
                      (3*n_features,3*1*n_features)],
          "epochs": [100]}

    dtw = {"max_depth":[None,2,4,6,8,10],
           "min_samples_split": [2,4,6,8,16,32,64]
          }

    rfw = {"max_depth":[None,2,4,6,8,10], 'min_samples_split':[2,4,6,8,16,32,64], 'n_estimators':[10,20,50,100]}

    knnw = {'n_neighbors':[1,2,4,8,16,32,64,128]}

    gkw = {'kernel_bandwidth':[1,1.5,2,2.5,3, *np.arange(3.5, n_features+0.5, 0.5)]}
    
    # Define model tuples: 'model_name', model, grid
    estimator_tuple_list = []
    estimator_tuple_list.append(('SAA', SampleAverageApproximationNewsvendor(),None))
    estimator_tuple_list.append(('LR', LinearRegressionNewsvendor(),None))
    estimator_tuple_list.append(('DTW', DecisionTreeWeightedNewsvendor(random_state=1),dtw))
    #estimator_tuple_list.append(('RFW', RandomForestWeightedNewsvendor(random_state=1),rfw))
    #estimator_tuple_list.append(('KNNW',KNeighborsWeightedNewsvendor(),knnw))
    #estimator_tuple_list.append(('GKW', GaussianWeightedNewsvendor(),gkw))
    #estimator_tuple_list.append(('DL', DeepLearningNewsvendor(),dl))
    
    # define under- and overage costs
    cu = [9, 7.5, 5, 2.5, 1]
    co = [1, 2.5, 5, 7.5, 9]
    
    # create dataframe to write best results in 
    results = pd.DataFrame()
     
    # dataframe for cv results
    cv_results = pd.DataFrame()
    
    #define cv strategy
    n_splits = 10
    cv = KFold(n_splits=n_splits)
    
    
    for  estimator_tuple in estimator_tuple_list:
        for group in groups:

            X_temp = X_grouped.get_group(group)
            y_temp = y.iloc[X_temp.index.values.tolist()]
            
            X_temp = X_temp.drop(["store_id", "dept_id"], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.75, shuffle=False)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #scale target variable
            scaler_target = StandardScaler()
            scaler_target.fit(np.array(y_train).reshape(-1, 1))
            y_train = scaler_target.transform(np.array(y_train).reshape(-1, 1))

            estimator_name = estimator_tuple[0]
            estimator = clone(estimator_tuple[1])
            param_grid = estimator_tuple[2]

            # TODO: Over SL
            
            if estimator_name == "SAA":
                for cu_i, co_i in zip(cu,co):
                    estimator.set_params(cu=cu_i,co=co_i)
                    estimator.fit(y_train)
                    pred = estimator.predict(X_test.shape[0])
                    pred = scaler_target.inverse_transform(pred)
                    avg_costs = average_costs(y_test,pred,cu_i,co_i)
                    avg_costs = round(avg_costs,4)
                    
                    d = {'Model': estimator_name, 'cu': cu_i, 'co': co_i, 'SL': cu_i/(cu_i+co_i), 'Group': str(group), 'Average costs': avg_costs, 'Coefficient of Prescriptiveness': 0, 'Best Params': np.nan}
                    results = results.append(d, ignore_index=True)
                    
                    
            elif estimator_name in ["KNNW", "RFW", "DTW", "GKW"]:
                logger.info('Parameter tuning {} for group {}'.format(estimator_name, group))
                grid = ParameterGrid(param_grid)
                candidate_params = list(grid)

                parallel = Parallel(n_jobs=-1)

                scores = parallel(delayed(get_sl_scores)(X=X_train, y=y_train, params=params, cu=cu, co=co, estimator=estimator, cv=cv) for params in candidate_params)
                scores = np.array(scores)
                mean_scores = scores.mean(axis=0)
                rank_scores = mean_scores.argmax(axis=0)
                best_scores = mean_scores.max(axis=0)
                best_params = [candidate_params[rank] for rank in rank_scores]
                
                for i in range(len(cu)):
                    cv_results_temp = pd.DataFrame(scores.T[0].T, columns = ['split'+str(split)+'_test_score' for split in range(n_splits)])
                    cv_results_temp["Model"] = estimator_name
                    cv_results_temp["cu"] = cu[i]
                    cv_results_temp["co"] = co[i]
                    cv_results_temp["SL"] = cu[i]/(cu[i]+co[i])
                    cv_results_temp["param"] = candidate_params
                    cv_results = pd.concat([cv_results, cv_results_temp], ignore_index=True)

                for i in range(len(cu)):
                    logger.info('Train {} with best params for group {} and service level {}'.format(estimator_name, group, cu[i]/(co[i]+cu[i])))
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
                    
                    d = {'Model': estimator_name, 'cu': cu[i], 'co': co[i], 'SL': cu[i]/(cu[i]+co[i]), 'Group': str(group), 'Average costs': avg_costs, 'Coefficient of Prescriptiveness': SoP, 'Best Params': best_params[i]}
                    results = results.append(d, ignore_index=True)
                    
                    logger.info("Average cost of best model: {}".format(avg_costs))
                    logger.info("Coeff of Prescriptiveness of best model: {}".format(SoP))
                    logger.info("------------------------------------------------------------------")
                    
            elif estimator_name == "LR":
                for cu_i, co_i in zip(cu,co):
                    logger.info('Train {} for group {} and service level {}'.format(estimator_name, group, cu_i/(co_i+cu_i)))
                    estimator.set_params(cu=cu_i,co=co_i)
                    estimator.fit(X_train, y_train)
                    pred = estimator.predict(X_test)
                    pred = scaler_target.inverse_transform(pred)
                    avg_costs = average_costs(y_test,pred,cu_i,co_i)
                    avg_costs = round(avg_costs,4)
                                        
                    saa_pred = SampleAverageApproximationNewsvendor(cu_i,co_i).fit(y_train).predict(X_test.shape[0])
                    saa_pred = scaler_target.inverse_transform(saa_pred)
                    SoP = prescriptiveness_score(y_test, pred, saa_pred, cu_i, co_i)
                    SoP = round(SoP,4)
                    
                    d = {'Model': estimator_name, 'cu': cu_i, 'co': co_i, 'SL': cu_i/(cu_i+co_i), 'Group': str(group), 'Average costs': avg_costs, 'Coefficient of Prescriptiveness': SoP, 'Best Params': np.nan}
                    results = results.append(d, ignore_index=True)
                    
                    logger.info("Average cost of best model: {}".format(avg_costs))
                    logger.info("Coeff of Prescriptiveness of best model: {}".format(SoP))
                    logger.info("------------------------------------------------------------------")
                    
            else:
                for cu_i, co_i in zip(cu,co):
                    base_estimator = clone(estimator)
                    base_estimator.set_params(cu=cu_i,co=co_i)
                    logger.info('Parameter tuning {} for group {} and SL {}'.format(estimator_name, group, cu_i/(co_i+cu_i)))
                    gs = GridSearchCV(base_estimator, param_grid, cv=cv, n_jobs=-1)
                    gs.fit(X_train,y_train)
                    
                    cv_results_temp = pd.DataFrame({k: v for k, v in gs.cv_results_.items() if k.startswith('split')})
                    cv_results_temp["Group"] = str(group)
                    cv_results_temp["Model"] = estimator_name
                    cv_results_temp["cu"] = cu_i
                    cv_results_temp["co"] = co_i
                    cv_results_temp["SL"] = cu_i/(cu_i+co_i)
                    cv_results_temp["param"] = gs.cv_results_["params"]
                    cv_results = pd.concat([cv_results, cv_results_temp], ignore_index=True)
                    
                    best_estimator = gs.best_estimator_
                    pred = best_estimator.predict(X_test)
                    pred = scaler_target.inverse_transform(pred)
                    avg_costs = average_costs(y_test,pred,cu_i,co_i)
                    avg_costs = round(avg_costs,4)

                    saa_pred = SampleAverageApproximationNewsvendor(cu_i,co_i).fit(y_train).predict(X_test.shape[0])
                    saa_pred = scaler_target.inverse_transform(saa_pred)
                    SoP = prescriptiveness_score(y_test, pred, saa_pred, cu_i, co_i)
                    SoP = round(SoP,4)
                    
                    d = {'Model': estimator_name, 'cu': cu_i, 'co': co_i, 'SL': cu_i/(cu_i+co_i), 'Group': str(group), 'Average costs': avg_costs, 'Coefficient of Prescriptiveness': SoP, 'Best Params': gs.best_params_}
                    results = results.append(d, ignore_index=True)
                    
                    logger.info("Average cost of best model: {}".format(avg_costs))
                    logger.info("Coeff of Prescriptiveness of best model: {}".format(SoP))
                    logger.info("------------------------------------------------------------------")
                    
            results.to_csv(result_path+'results.csv', index=False)
            cv_results.to_csv(result_path+'cv_results.csv', index=False)
            
            # break for testing --> consinder only one group. Remove for final testing !!!!!!
            print("Remove break before testing!!! See comment in code")
            break
            

if __name__ == '__main__':
    
    main()
