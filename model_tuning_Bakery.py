from sklearn.utils.validation import check_array
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from ddop.datasets import load_yaz, load_bakery
from ddop.metrics import average_costs, prescriptiveness_score
from ddop.newsvendor import SampleAverageApproximationNewsvendor
from ddop.newsvendor import DecisionTreeWeightedNewsvendor
from ddop.newsvendor import RandomForestWeightedNewsvendor 
from ddop.newsvendor import KNeighborsWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import GaussianWeightedNewsvendor
from ddop.newsvendor import LinearRegressionNewsvendor
from ddop.newsvendor import DeepLearningNewsvendor
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statistics

import os
import logging

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


def main():
    
    # define file paths
    
    log_path = "logs_bakery/"
    result_path = "results_bakery/"
    
    # set a logger file
    logger = log(path=log_path, file="cross_val.logs")
    
    # load data
    bakery = load_bakery(one_hot_encoding=True)
    X = bakery.data
    y = bakery.target
    
    X_grouped = X.groupby(['product', 'store'])
    groups = list(X_grouped.groups.keys())
    
    n_features = len(X.columns)
    
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
          "epochs": [10,100,200]}

    dtw = {"max_depth":[None,2,4,6,8,10] ,"min_samples_split": [2,4,6,8,16,32,64]}

    rfw = {"max_depth":[None,2,4,6,8,10],
              'min_samples_split':[2,4,6,8,16,32,64],
              'n_estimators':[10,20,50,100]}

    knnw = {'n_neighbors':[1,2,4,8,16,32,64,128]}

    gkw = {'kernel_bandwidth':[1,1.5,2,2.5,3, *np.arange(3.5, n_features+0.5, 0.5)]}
    
    # Define model tuples: 'model_name', model, grid
    estimator_tuple_list = []
    estimator_tuple_list.append(('SAA', SampleAverageApproximationNewsvendor(),None))
    estimator_tuple_list.append(('DTW', DecisionTreeWeightedNewsvendor(random_state=1),dtw))
    estimator_tuple_list.append(('RFW', RandomForestWeightedNewsvendor(n_jobs=4, random_state=1),rfw))
    estimator_tuple_list.append(('KNNW',KNeighborsWeightedNewsvendor(),knnw))
    estimator_tuple_list.append(('GKW', GaussianWeightedNewsvendor(),gkw))
    estimator_tuple_list.append(('DL', DeepLearningNewsvendor(),[dl]))
    estimator_tuple_list.append(('LR', LinearRegressionNewsvendor(),None))
    
    estimators = []
    best_model = pd.DataFrame()
    results = pd.DataFrame()
    for cu, co in zip([5,7.5,9],[5,2.5,1]):
    #for cu, co in zip([7.5],[2.5]):
      for  estimator_tuple in estimator_tuple_list:
        costs = []
        score = []
        for group in groups:
    
          X_temp = X_grouped.get_group(group)
          y_temp = y.iloc[X_temp.index.values.tolist()]
    
          X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.75, shuffle=False)
          scaler = StandardScaler()
          scaler.fit(X_train)
          X_train = scaler.transform(X_train)
          X_test = scaler.transform(X_test)
    
          #scale target variable
          scaler_target = StandardScaler()
          scaler_target.fit(np.array(y_train).reshape(-1, 1))
          y_train = scaler_target.transform(np.array(y_train).reshape(-1, 1))
    
          saa_pred = SampleAverageApproximationNewsvendor(cu,co).fit(y_train).predict(y_test.shape[0])
          saa_pred = scaler_target.inverse_transform(saa_pred)
          estimator_name = estimator_tuple[0]
          param_grid = estimator_tuple[2]
          estimator = estimator_tuple[1]
          estimator.set_params(cu=cu,co=co)
          
          if param_grid == None:
            if estimator_name=="SAA":
              logger.info('Train model {} for group {} and service level {}'.format(estimator_name, group, (cu/(co+cu))))
              pred = estimator.fit(y_train).predict(X_test.shape[0])
              pred = scaler_target.inverse_transform(pred)
            else:
              logger.info('Train model {} for group {} and service level {}'.format(estimator_name, group, (cu/(co+cu))))
              pred = estimator.fit(X_train,y_train).predict(X_test)
              pred = scaler_target.inverse_transform(pred)
          
          else:
            logger.info('Train model {} for group {} and service level {}'.format(estimator_name, group, (cu/(co+cu))))
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            gs = GridSearchCV(estimator, param_grid, cv=cv, n_jobs=-1)
            gs.fit(X_train,y_train)
            best_estimator = gs.best_estimator_
            pred = best_estimator.predict(X_test)
            pred = scaler_target.inverse_transform(pred)
            estimators.append(best_estimator)
            
            # save best estimator
            g = {'SL': [cu/(cu+co)], 'group' : str(group), 'model_type' : estimator_name, 'model': best_estimator}
            best_model = pd.concat([best_model, pd.DataFrame(data=g)])
            best_model.to_pickle(result_path+'best_models.csv')              
                
          logger.info('Finished training')
          avg_cost = average_costs(y_test,pred,cu,co,multioutput="uniform_average")
          p_score = prescriptiveness_score(y_test, pred, saa_pred, cu, co, multioutput="uniform_average")
          costs.append(avg_cost)
          score.append(p_score)
          logger.info("Average cost of best model: {}".format(avg_cost))
          logger.info("Coeff of Prescriptiveness of best model: {}".format(p_score))
          logger.info("------------------------------------------------------------------")

        d = {'SL': [cu/(cu+co)], 'Model': [estimator_name]}  
        for i in range(len(costs)):
          d[str(groups[i])+" AC"] = costs[i]
        for i in range(len(costs)):
          d[str(groups[i])+" SoP"] = score[i]
    
        average_cost = statistics.mean(costs)
        d["Average Cost"] = average_cost
        presc_score = statistics.mean(score)
        d["Score of Prescriptiveness"] = presc_score
        df = pd.DataFrame(data=d)
        results = pd.concat([results,df])
        results.to_csv(result_path+'results.csv')
        
    
if __name__ == '__main__':
    
    main()