# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:17:04 2017

@author: AUGUSTE
"""
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.svm import SVC
import pickle


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

        else:
            profit_m = 0
            profits_multipliers.append(profit_m)

    return profits_multipliers

VALIDATION_FILEPATH = 'data/ML/E0_ML_n3_date_valid.csv'
MODEL_FILES = {1: 'model_part_1.p',
               2: 'model_part_2.p',
               3: 'model_part_3.p',
               4: 'model_part_4.p'}

STANDARDISER_FILES = {1: 'standardiser_part_1.p',
                      2: 'standardiser_part_2.p',
                      3: 'standardiser_part_3.p',
                      4: 'standardiser_part_4.p'}

VALID_DATES = {1: {'min_month': 8, 'max_month': 10},
               2: {'min_month': 11, 'max_month': 12},
               3: {'min_month': 1, 'max_month': 2},
               4: {'min_month': 3, 'max_month': 5}}

FEATURES_TO_KEEP = {1: ['h_season_wages', 'diff_nb_defeats', 'diff_season_wages'],
                    2: ['h_nb_goals_scored', 'h_nb_goals_diff', 'h_nb_games', 'h_nb_games_home', 'h_nb_goals_conceded_home', 'h_diff_goals_home', 'h_last_n_games_points_home', 'h_mean_nb_goals_scored_home', 'h_mean_nb_goals_conceded_home', 'h_season_wages', 'a_nb_goals_conceded', 'a_nb_goals_diff', 'a_nb_goals_conceded_away', 'a_mean_nb_goals_conceded_away', 'a_season_wages', 'Month', 'Week', 'distance_km'],
                    3: ['h_nb_goals_diff', 'a_season_wages'],
                    4: ['h_nb_goals_diff', 'a_nb_goals_scored', 'a_nb_goals_diff']}

ODDS_FILEPATH = 'data/ML/E0_home_win_odds_valid.csv'
PROBA_THRESH = 0.5
if __name__ == '__main__':
    data_odds = pd.read_csv(ODDS_FILEPATH)
    data_all = pd.read_csv(VALIDATION_FILEPATH)
    data_all['Date'] = pd.to_datetime(data_all['Date'])
    for part_i in MODEL_FILES:

        data = data_all[(data_all['Date'].dt.month >= VALID_DATES[part_i]['min_month'])
                        & (data_all['Date'].dt.month <= VALID_DATES[part_i]['max_month'])]

        if len(data) > 0:
            dates = data['Date'].values
            data = data.drop('Date', 1)
            # store id of games
            id_str = data['id'].values
            data = data.drop('id', 1)

            y = data['home_win'].values
            data = data.drop('home_win', 1)

            data = data[FEATURES_TO_KEEP[part_i]]

            # for feat in FEATURES_LOG:
            # data[feat] = data[feat].apply(lambda x: np.log10(1+x))
            features = data.columns
            X = data.values

            stdardiser_file = open(STANDARDISER_FILES[part_i], 'rb')
            standardizer = pickle.load(stdardiser_file)
            X = standardizer.fit_transform(X)

            model_file = open(MODEL_FILES[part_i], 'rb')
            classifier = pickle.load(model_file)
            # modelfit(classifier, X, y)

            proba = classifier.predict_proba(X)
            proba_home_win = [p[1] for p in proba]
            predictions = [1 if p[1] > PROBA_THRESH else 0 for p in proba]
            auc = roc_auc_score(y, proba_home_win)
            fpr, tpr, thresholds = roc_curve(y, proba_home_win, pos_label=1)

            acc = accuracy_score(y, predictions)
            conf_mat = confusion_matrix(y, predictions)
            f1 = f1_score(y, predictions)
            prec = precision_score(y, predictions)
            recall = recall_score(y, predictions)

            profits_multipliers = get_profit_multipliers(y, predictions,
                                                         id_str, data_odds)

            print('Part %d' % part_i)
            print(classifier)
            print('Sum of multipliers: %f' % np.sum(profits_multipliers))
            print('Area under the curve: %f' % auc)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            print('')
            print('')

            sum = 0
            cumulated_profits = []
            for profit in profits_multipliers:
                sum += profit
                cumulated_profits.append(sum)

            profit_over_time = pd.DataFrame({'cumulated_profits': cumulated_profits},
                                            index=dates)

            profit_over_time = profit_over_time.resample('D').mean()
            profit_over_time = profit_over_time.fillna(method='ffill')
            print(profit_over_time)
            profit_over_time.plot()

        plt.show()
