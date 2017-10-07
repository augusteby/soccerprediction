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
    odds = []
    for i in range(len(y)):
        pred = predictions[i]
        real = y[i]
        id_str = id_strings[i]
        home_win_odd = home_win_odds.loc[id_str]['BbAvH']
        odds.append(home_win_odd)

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

    return profits_multipliers,odds

VALIDATION_FILEPATH = 'data/ML/E0_ML_n3_date_valid.csv'
MODEL_FILES = {1: 'model_part_1.p',
               2: 'model_part_2.p',
               3: 'model_part_3.p',
               4: 'model_part_4.p',
               5: 'model_part_5.p'}

STANDARDISER_FILES = {1: 'standardiser_part_1.p',
                      2: 'standardiser_part_2.p',
                      3: 'standardiser_part_3.p',
                      4: 'standardiser_part_4.p',
                      5: 'standardiser_part_5.p'}

FEATURES_FILES = {1: 'features_part_1.p',
                  2: 'features_part_2.p',
                  3: 'features_part_3.p',
                  4: 'features_part_4.p',
                  5: 'features_part_5.p',}

VALID_DATES = {1: {'min_month': 8, 'max_month': 8},
                2: {'min_month': 9, 'max_month': 10},
                3: {'min_month': 11, 'max_month': 12},
                4: {'min_month': 1, 'max_month': 2},
                5: {'min_month': 3, 'max_month': 5}}



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

            features_file = open(FEATURES_FILES[part_i], 'rb')
            features = pickle.load(features_file)
            data = data[features]

            # for feat in FEATURES_LOG:
            # data[feat] = data[feat].apply(lambda x: np.log10(1+x))
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

            profits_multipliers,odds = get_profit_multipliers(y, predictions,
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
            
            profit_over_time.plot()
            profit_and_proba_over_time = pd.DataFrame({'cumulated_profits': cumulated_profits,
                                             'odds':odds, 
                                             'proba': proba_home_win,
                                             'result': y})
            print(profit_and_proba_over_time)
            profit_and_proba_over_time.plot.scatter(x='odds',y='proba',c='result',colormap='coolwarm')

        plt.show()
