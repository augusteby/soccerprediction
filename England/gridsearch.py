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
from xgboost import XGBClassifier
import xgboost as xgb


def get_earning_coeff(y, predictions, id_strings, home_win_odds):
    sum_profit_multipliers = 0
    home_win_odds.index = home_win_odds['id']
    for i in range(len(y)):
        pred = predictions[i]
        real = y[i]
        id_str = id_strings[i]
        home_win_odd = home_win_odds.loc[id_str]['BbAvH']

        # If algo thinks home team will win
        if pred == 1:
            # If this is indeed true, we make money
            if pred == real:
                sum_profit_multipliers += home_win_odd - 1

            # If the decision was wrong, we loose what we bet
            else:
                sum_profit_multipliers -= 1

    return sum_profit_multipliers

FILEPATH = 'data/ML/E0_ML_n3.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
ML_ALGO = 'log_reg'
ALGOS = {
# 'rdmf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
         'log_reg': LogisticRegression(n_jobs=-1),
         # 'knn': KNeighborsClassifier(n_jobs=-1),
         'svm': SVC()
         # 'ada': AdaBoostClassifier(),
         # 'xgboost': XGBClassifier(nthread=-1)
         }
PARAM_GRID = {'rdmf': {'min_samples_leaf': np.arange(0.1, 0.5, 0.01),
                       'max_features': np.arange(0.1, 1, 0.01),
                       'min_samples_split': np.arange(0.1, 1, 0.05)},
              'log_reg': {'C': [10**i for i in range(-4, 2)]},
              'knn': {'n_neighbors': range(1, 20, 3),
                      'p': [1, 2, 3]},
              'svm': {'C': [10**i for i in range(-4, 2)],
                      'gamma': [10**i for i in range(-3, 2)]},
              'ada': {'learning_rate': [10**i for i in range(-4, 2)],
                      'n_estimators': [100, 50]},
              'xgboost': {'n_estimators': range(5, 250, 5),
                          'learning_rate': [10**i for i in range(-4, 2)],
                          'max_depth': [2, 3, 4, 5]}}

if __name__ == '__main__':
    # Load data
    data_odds = pd.read_csv(ODDS_FILEPATH)
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    # select relevant data
    y = data['home_win'].values
    data = data.drop('home_win', 1)
    features = data.columns
    X = data.values

    # Standardisation
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    # Personalised metrics
    earning_coeff_score = make_scorer(get_earning_coeff, id_strings=id_str,
                                      home_win_odds=data_odds,
                                      greater_is_better=True)
    aucroc_score = make_scorer(roc_auc_score)
    log_loss_score = 'neg_log_loss'

    for algo in ALGOS:
        print(algo)
        classifier = ALGOS[algo]

        grid_search = GridSearchCV(classifier, param_grid=PARAM_GRID[algo],
                                   scoring=aucroc_score, cv=5,
                                   verbose=0, n_jobs=-1)
        grid_search.fit(X, y)

        print('Algo: %s' % algo)
        print('Best score: %f' % grid_search.best_score_)
        print(grid_search.best_params_)
        print('')
