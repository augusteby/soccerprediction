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


FILEPATH = 'data/ML/E0_ML_n3_type2.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
FEATURES_LOG = ['h_nb_victories', 'h_season_points',
                'a_nb_victories_draws', 'a_season_points']

SELECTED_CLASSIFIER = 'logreg'
CLASSIFIERS = {'rdmf': RandomForestClassifier(n_estimators=100, min_samples_leaf=0.12,
                                              min_samples_split=0.45, max_features=0.18,
                                              n_jobs=-1),
               'logreg': LogisticRegression(C=0.0001, n_jobs=-1),
               'xgboost': XGBClassifier(n_estimators=115, learning_rate=0.01, max_depth=3,
                                        nthread=-1, seed=27),
               'svm': SVC(gamma=0.001, C=10, probability=True),
               'ada': AdaBoostClassifier(learning_rate=1, n_estimators=100)}

FEATURES_TO_KEEP = {'rdmf': ['h_nb_victories', 'h_nb_points', 'h_nb_goals_scored',
                             'h_nb_goals_diff', 'h_nb_victories_home',
                             'h_nb_points_home', 'h_nb_goals_scored_home',
                             'h_diff_goals_home', 'h_mean_nb_goals_scored_home',
                             'h_season_wages', 'a_nb_goals_diff',
                             'a_nb_victories_away', 'a_nb_defeats_away',
                             'a_nb_points_away', 'a_diff_goals_away',
                             'a_last_n_games_victories_away',
                             'a_last_n_games_defeats_away',
                             'a_mean_nb_goals_scored_away', 'a_season_wages',
                             'distance_km', 'capacity_home_stadium'],
                    'logreg': ['h_season_wages'],
                    'xgboost': ['h_nb_points', 'h_nb_goals_scored',
                                'h_nb_goals_diff', 'h_nb_draws_home',
                                'h_nb_goals_conceded_home', 'h_last_n_games_draws_home',
                                'h_mean_nb_goals_scored_home', 'h_mean_nb_goals_conceded_home',
                                'h_season_wages', 'a_nb_victories', 'a_nb_draws',
                                'a_nb_goals_conceded', 'a_nb_goals_diff', 'a_nb_draws_away',
                                'a_nb_defeats_away', 'a_nb_goals_scored_away',
                                'a_nb_goals_conceded_away', 'a_diff_goals_away',
                                'a_last_n_games_draws_away', 'a_mean_nb_goals_conceded_away',
                                'a_season_wages', 'Week'],
                    'ada': ['h_nb_draws', 'h_nb_goals_scored',
                            'h_nb_goals_diff', 'h_nb_victories_home',
                            'h_diff_goals_home', 'h_mean_nb_goals_scored_home',
                            'h_mean_nb_goals_conceded_home', 'h_season_wages',
                            'a_nb_points', 'a_nb_goals_conceded', 'a_nb_goals_diff',
                            'a_diff_goals_away', 'a_mean_nb_goals_scored_away',
                            'a_mean_nb_goals_conceded_away', 'a_season_wages']}


PROBA_THRESH = 0.5
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
    # if 'Month' in data.columns.values:
    #     data = pd.get_dummies(data, columns=['Month'])
    # if 'Week' in data.columns.values:
    #     data = pd.get_dummies(data, columns=['Week'])

    y = data['home_win_odd_above'].values
    data = data.drop('home_win_odd_above', 1)

    data = data[FEATURES_TO_KEEP[SELECTED_CLASSIFIER]]

    # for feat in FEATURES_LOG:
    #data[feat] = data[feat].apply(lambda x: np.log10(1+x))
    features = data.columns
    X = data.values

    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    classifier = CLASSIFIERS[SELECTED_CLASSIFIER]
    # modelfit(classifier, X, y)

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

    print('Classifier: %s' % SELECTED_CLASSIFIER)
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
