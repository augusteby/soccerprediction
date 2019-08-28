# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:58:09 2017

@author: AUGUSTE
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib

# Use TkAgg backend with matplotlib because the backend by default might cause the following issue:
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
# More info here: https://matplotlib.org/faq/usage_faq.html
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, \
    roc_curve, precision_recall_curve, confusion_matrix

from constants import LABEL
from data_ingestor import load_X_y

TRAINING_DATA_FILEPATH = 'data/ML/training_E0_ML.csv'
TESTING_DATA_FILEPATH = 'data/ML/testing_E0_ML.csv'

if __name__=='__main__':
    X_train, y_train, features = load_X_y(TRAINING_DATA_FILEPATH)

    # Normalise based on mean and variance of variables in training data
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)

    classifier = SVC(C=0.1, gamma=0.001)
    classifier.fit(X_train, y_train)

    data_test = pd.read_csv(TESTING_DATA_FILEPATH)
    y = data_test[LABEL].values

    max_nb_games_played = int(np.max(data_test['h_nb_games_total'].values))
    
    prec1 = []
    prec2 = []
    nb_games_played_list = range(max_nb_games_played)
    for nb_games_played in tqdm(nb_games_played_list):
        
        sub_data1 = data_test[data_test['h_nb_games_total']>=nb_games_played]
        sub_data2 = data_test[data_test['h_nb_games_total']==nb_games_played]
    
        y1 = sub_data1[LABEL].values
        sub_data1 = sub_data1.drop(LABEL, 1)
        y2 = sub_data2[LABEL].values
        sub_data2 = sub_data2.drop(LABEL, 1)
        
        X1 = sub_data1.values
        X2 = sub_data2.values

        X1 = standardizer.transform(X1)
        X2 = standardizer.transform(X2)

        predictions1 = classifier.predict(X1)
        predictions2 = classifier.predict(X2)

        acc1 = accuracy_score(y1, predictions1)
        acc2 = accuracy_score(y2, predictions2)

        prec1.append(precision_score(y1, predictions1))
        prec2.append(precision_score(y2, predictions2))

        # recall_score(y1, predictions1)
        # recall_score(y2, predictions2)

    plt.figure()  
    plt.plot(nb_games_played_list, prec1)
    plt.xlabel('Minimum number of games played in season by home team')
    plt.ylabel('Precision')
    plt.title('At which point of the season should I start betting?')
    
    plt.figure()
    plt.plot(nb_games_played_list, prec2)
    plt.xlabel('Exact number of games played in season by home team')
    plt.ylabel('Precision')
    plt.title('At which days of the season should I bet?')
    
    plt.show()
