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
    proba = cross_val_predict(classifier, X, y,
                              method='predict_proba',
                              cv=10, n_jobs=-1)
    proba_home_win = [p[1] for p in proba]
    predictions = [1 if p[1] > PROBA_THRESH else 0 for p in proba]
    auc = roc_auc_score(y, proba_home_win)
    fpr, tpr, thresholds = roc_curve(y, proba_home_win, pos_label=1)

    coeff_tnr = 0.6
    coeff_tpr = 0.4
    tnr_plus_tpr = [coeff_tnr * (1 - fpr[i]) + coeff_tpr * tpr[i]
                    for i in range(len(fpr))]
    index_max, max_tnr_plus_tpr = max(
        enumerate(tnr_plus_tpr), key=operator.itemgetter(1))
    thresholds_of_max = thresholds[index_max]

    acc = accuracy_score(y, predictions)
    conf_mat = confusion_matrix(y, predictions)
    f1 = f1_score(y, predictions)
    prec = precision_score(y, predictions)
    recall = recall_score(y, predictions)

    profits_multipliers = get_profit_multipliers(y, predictions,
                                                 id_str, data_odds)
    print('Sum of multipliers: %f' % np.sum(profits_multipliers))
    print('')

    print('Proba Threshold: %f' % PROBA_THRESH)
    print('Accuracy: %f' % acc)
    print('F1: %f' % f1)
    print('Precision: %f' % prec)
    print('Recall: %f' % recall)

    print('')
    print('Max %.2f*tnr + %.2f*tpr: %f' %
          (coeff_tnr, coeff_tpr, max_tnr_plus_tpr))
    print('tnr @ max: %f' % (1 - fpr[index_max]))
    print('tpr @ max: %f' % tpr[index_max])
    print('Threshold of max fpr + tnr: %f' % thresholds_of_max)
    print('Area under the curve: %f' % auc)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

    if RDMF:
        classifier.fit(X, y)
        print(features)
        print(classifier.feature_importances_)
    else:
        classifier.fit(X, y)
        for i in range(len(features)):
            print(features[i])
            print(classifier.coef_[0][i])
            print('')
