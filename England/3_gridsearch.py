# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:17:06 2017

@author: AUGUSTE
"""
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
import time

from data_ingestor import load_X_y

TRAINING_DATA_FILEPATH = 'data/ML/training_E0_ML.csv'
TESTING_DATA_FILEPATH = 'data/ML/testing_E0_ML.csv'

ALGOS = {
         'rdmf': RandomForestClassifier(),
         'log_reg': LogisticRegression(solver='lbfgs', max_iter=500),
         'knn': KNeighborsClassifier(),
         'svm': SVC(),
         'xgb': XGBClassifier(n_estimators=100)
}

PARAM_GRID = {
              'rdmf': {'min_samples_leaf': [5, 10, 20, 40, 80],
                       'max_features': [0.2, 0.4, 0.8],
                       'n_estimators': [20, 50, 100]},
              'log_reg': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
              'knn': {'n_neighbors': range(10, 100, 10),
                      'p': [1, 2, 3]},
              'svm': {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1e-4, 0.001, 0.01],
                      'kernel': ['rbf', 'sigmoid', 'poly'],
                      'tol': [1e-3, 1e-2, 1e-4],
                      'degree': [2, 3, 4]},
              'xgb': {'max_depth': [2, 3, 4, 5],
                      'learning_rate': [0.1, 0.2, 0.3, 0.4]}
}
if __name__=='__main__':
    start_time = time.time()
    X, y, features = load_X_y(TRAINING_DATA_FILEPATH)
                 
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)
    
    for algo in tqdm(ALGOS):
        classifier = ALGOS[algo]
        
        score = make_scorer(precision_score)
        grid_search = GridSearchCV(classifier, param_grid=PARAM_GRID[algo], 
                                   scoring=score, n_jobs=-1, cv=4)
        grid_search.fit(X, y)
        
        print('Algo: %s' % algo)
        print('Best score: %f' % grid_search.best_score_)
        print(grid_search.best_params_)

    elapsed_time = time.time() - start_time
    print('---------------- Elapsed time : {} ----------------\n'.format(elapsed_time))

