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
from xgboost import XGBClassifier
import xgboost as xgb


def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X, y, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy_score(y, dtrain_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(y, dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()
                         ).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def get_profit_multipliers(y, predictions, id_strings, home_win_odds):
    """
    We define a profit multiplier as follow: 
    It is equal to 'decimal_odd -1'. If it is positive you
    make (decimal_odd-1) time the money you bet, 
    otherwise it is equal to -1 and you've lost the money you bet

    Parameters
    ----------
    y: numpy.ndarray
        Real outcome: 1 means 'home team won' and 0 'home team lost or draw'
    predictions: numpy.ndarray
        Predicted outcome
    id_strings: numpy.ndarray
        List of ids in the same order of association as for the elements in 
        'y' and 'predictions'
    home_win_odds: pandas.DataFrame
        'Home win' odds associated with each game

    Returns
    -------
    profits_multipliers: list
        List that contains 
    """
    profits_multipliers = []
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
                profit_m = home_win_odd - 1
                profits_multipliers.append(profit_m)

            # If the decision was wrong, we loose what we bet
            else:
                profit_m = -1
                profits_multipliers.append(profit_m)

    return profits_multipliers


FILEPATH = 'data/ML/E0_ML.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
FEATURES_LOG = ['h_nb_victories', 'h_season_points',
                'a_nb_victories_draws', 'a_season_points']

CLASSIFIERS = {'rdmf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
               'logreg': LogisticRegression(n_jobs=-1),
               'xgboost': XGBClassifier(n_estimators=24, learning_rate=0.05, max_depth=3,
                                        min_child_weight=1, gamma=0,
                                        scale_pos_weight=1, nthread=-1, seed=27)}

FEATURES_TO_KEEP = {'rdmf': ['h_season_points', 'h_mean_nb_goals_scored_home',
                             'h_mean_nb_goals_conceded_home', 'h_season_wages',
                             'a_mean_nb_goals_scored_away',
                             'a_mean_nb_goals_conceded_away',
                             'a_season_wages', 'distance_km'],
                    'logreg': ['h_nb_games_home', 'h_nb_victories', 'h_season_points',
                               'h_nb_games_total', 'h_nb_goals_scored_home',
                               'h_season_wages', 'a_nb_games_away', 'a_season_points',
                               'a_nb_games_total', 'a_season_wages'],
                    'xgboost': ['h_nb_victories', 'h_season_points',
                                'h_nb_games_total', 'h_nb_goals_scored_home',
                                'h_mean_nb_goals_scored_home', 'h_nb_goals_conceded_home',
                                'h_mean_nb_goals_conceded_home', 'h_season_wages',
                                'a_season_points', 'a_nb_goals_scored_away',
                                'a_mean_nb_goals_scored_away', 'a_mean_nb_goals_conceded_away',
                                'a_season_wages', 'capacity_home_stadium']}


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

    # encode categorical data
    if 'Month' in data.columns.values:
        data = pd.get_dummies(data, columns=['Month'])
    if 'Week' in data.columns.values:
        data = pd.get_dummies(data, columns=['Week'])

    y = data['home_win'].values
    data = data.drop('home_win', 1)

    

    probas = {}
    for classif_name in CLASSIFIERS:
        # for feat in FEATURES_LOG:
        #data[feat] = data[feat].apply(lambda x: np.log10(1+x))
        features = data.columns
        X = data[FEATURES_TO_KEEP[classif_name]].values

        standardizer = StandardScaler()
        X = standardizer.fit_transform(X)

        classifier = CLASSIFIERS[classif_name]
        # modelfit(classifier, X, y)

        proba = cross_val_predict(classifier, X, y,
                                  method='predict_proba',
                                  cv=10, n_jobs=-1)
        proba_home_win = [p[1] for p in proba]
        probas[classif_name] = proba_home_win
        auc = roc_auc_score(y, proba_home_win)
        print(classif_name)
        print(auc)
        print('')

    mean_probas = []
    nb_classifiers = len(CLASSIFIERS)

    for i in range(len(y)):
        mean_prob = 0
        for classif_name in CLASSIFIERS:
            mean_prob += probas[classif_name][i]

        mean_prob /= nb_classifiers

        mean_probas.append(mean_prob)

    auc_mean = roc_auc_score(y, mean_probas)
    print('Mean')
    print(auc_mean)

