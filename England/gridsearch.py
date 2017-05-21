# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:17:06 2017

@author: AUGUSTE
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

FILEPATH = 'data/ML/E0_ML.csv'

ML_ALGO = 'gauss_proc'
ALGOS = {'rdmf':RandomForestClassifier,
         'log_reg':LogisticRegression,
         'knn':KNeighborsClassifier,
         'svm':SVC,
         'gnb':GaussianNB,
         'ada':AdaBoostClassifier,
         'gauss_proc':GaussianProcessClassifier} 
PARAM_GRID = {'rdmf':{'min_samples_leaf':[5, 10, 20,40,80],
                      'max_features':[0.2, 0.4, 0.8]},
              'log_reg':{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]},
              'knn':{'n_neighbors':range(10,100,10),
                     'p':[1,2,3]},
              'svm':{'C':[0.1, 1, 10,100,1000],
                     'gamma':[0.001,0.01]},
              'gnb':{},
              'ada':{'learning_rate':[0.01,0.1,1,10,100],
                     'n_estimators':[100, 50]},
              'gauss_proc':{}}
if __name__=='__main__':
    data = pd.read_csv(FILEPATH)
    
    y = data['home_win'].values
    data = data.drop('home_win',1)
    features = data.columns
    X = data.values
                 
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)
    
    for algo in ALGOS:
        classifier = ALGOS[algo]()
        
        score = make_scorer(roc_auc_score)
        grid_search = GridSearchCV(classifier, param_grid=PARAM_GRID[algo], 
                                   scoring=score, n_jobs=-1)
        grid_search.fit(X,y)
        
        print('Algo: %s' % algo)
        print('Best score: %f' % grid_search.best_score_)
        print(grid_search.best_params_)
    
    
