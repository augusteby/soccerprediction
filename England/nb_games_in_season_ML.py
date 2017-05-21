# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:58:09 2017

@author: AUGUSTE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier

FILEPATH = 'data/ML/E0_ML.csv'

if __name__=='__main__':
    data = pd.read_csv(FILEPATH)
    
    max_nb_games_played = int(np.max(data['h_nb_games_total'].values))
    
    AUCs1 = []
    AUCs2 = []
    nb_games_played_list = range(max_nb_games_played)
    for nb_games_played in nb_games_played_list:
        
        sub_data1 = data[data['h_nb_games_total']>=nb_games_played]
        sub_data2 = data[data['h_nb_games_total']==nb_games_played]
    
        y1 = sub_data1['home_win'].values
        sub_data1 = sub_data1.drop('home_win',1)
        y2 = sub_data2['home_win'].values
        sub_data2 = sub_data2.drop('home_win',1)
        
        X1 = sub_data1.values
        X2 = sub_data2.values
                     
        standardizer = StandardScaler()
        X1 = standardizer.fit_transform(X1)
        X2 = standardizer.fit_transform(X2)
        
        classifier = LogisticRegression()
        proba1 = cross_val_predict(classifier, X1, y1, method='predict_proba', cv=10)
        proba_home_win1 = [e[1] for e in proba1]
        predictions1 = [1 if p[1]>0.5 else 0 for p in proba1]
        proba2 = cross_val_predict(classifier, X2, y2, method='predict_proba', cv=10)
        proba_home_win2 = [e[1] for e in proba2]
        predictions2 = [1 if p[1]>0.5 else 0 for p in proba2]
        
        acc1 = accuracy_score(y1, predictions1)
        acc2 = accuracy_score(y2, predictions2)
        
        auc1 = roc_auc_score(y1, proba_home_win1)
        AUCs1.append(auc1)
        auc2 = roc_auc_score(y2, proba_home_win2)
        AUCs2.append(auc2)
        
    plt.figure()  
    plt.plot(nb_games_played_list, AUCs1)
    plt.xlabel('Minimum number of games played in season')
    plt.ylabel('AUC')
    plt.title('At which point of the season should I start betting?')
    
    plt.figure()
    plt.plot(nb_games_played_list, AUCs2)
    plt.xlabel('Number of games played in season')
    plt.ylabel('AUC')
    plt.title('At which days of the season should I bet?')
    
    plt.show()
        
       
        
        