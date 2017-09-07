# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:17:04 2017

@author: AUGUSTE
"""
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer


FILEPATH = 'data/ML/E0_ML.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
FEATURES_LOG = ['h_nb_victories', 'h_season_points',
                'a_nb_victories_draws', 'a_season_points']
FEATURES_TO_KEEP = ['h_nb_games_home', 'h_season_points',
                    'h_nb_goals_scored_home', 'h_nb_goals_conceded_home',
                    'a_season_points', 'a_nb_goals_scored_away',
                    'a_nb_goals_conceded_away']
RDMF = False
PROBA_THRESH = 0.6
if __name__ == '__main__':
    data_odds = pd.read_csv(ODDS_FILEPATH)
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    # if ('Date' in set(data.columns.values)):
    #     data['Date'] = pd.to_datetime(data['Date'])

    #data = data[data['h_nb_games_total']>18]

    data = data.drop('Month', 1)
    data = data.drop('Week', 1)
    # encode categorical data
    if 'Month' in data.columns.values:
        data = pd.get_dummies(data, columns=['Month'])
    if 'Week' in data.columns.values:
        data = pd.get_dummies(data, columns=['Week'])

    y = data['home_win'].values
    data = data.drop('home_win', 1)
    # data = data.drop('h_season_wages', 1)
    # data = data.drop('a_season_wages', 1)
    #data = data[FEATURES_TO_KEEP]

    # for feat in FEATURES_LOG:
    #data[feat] = data[feat].apply(lambda x: np.log10(1+x))
    features = data.columns
    X = data.values

    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    if RDMF:
        classifier = RandomForestClassifier(n_jobs=-1)
    else:
        classifier = LogisticRegression(n_jobs=-1)

    aucroc_score = make_scorer(roc_auc_score)
    rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2),
                  scoring=aucroc_score)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.ranking_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()