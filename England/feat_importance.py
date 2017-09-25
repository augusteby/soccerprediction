# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

FILEPATH = 'data/ML/E0_ML_n3.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
if __name__ == '__main__':


    data_odds = pd.read_csv(ODDS_FILEPATH)
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    y = data['home_win'].values
    data = data.drop('home_win', 1)

    # data = data[FEATURES_TO_KEEP[SELECTED_CLASSIFIER]]

    # for feat in FEATURES_LOG:
    # data[feat] = data[feat].apply(lambda x: np.log10(1+x))
    features = data.columns
    X = data.values

    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    print('Computation of feature importance score...')
    rdmF = RandomForestRegressor(n_estimators=500)
    rdmF.fit(X, y)

    importances = rdmF.feature_importances_
    print('Importance scores')
    print(importances)
    feat_importance_df = pd.DataFrame(
        {'features': features, 'importance': importances})

    feat_importance_df.plot(
        x='features', y='importance', kind='bar', rot=90)
    title = 'Importance score of each feature (Random Forest with %d trees)' % (
            500)

    plt.title(title)
    plt.show()
