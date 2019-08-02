# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:17:04 2017

@author: AUGUSTE
"""
import operator
import pandas as pd
import matplotlib

# Use TkAgg backend with matplotlib because the backend by default might cause the following issue:
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
# More info here: https://matplotlib.org/faq/usage_faq.html
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
TRAINING_FILEPATH = 'data/ML/training_E0_ML.csv'

RDMF = False
if __name__=='__main__':
    data = pd.read_csv(TRAINING_FILEPATH)

    y = data['home_win'].values
    data = data.drop('home_win', 1)

    features = data.columns
    X = data.values
                 
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)
    
    if RDMF:
        classifier = RandomForestClassifier(n_estimators=50, max_features=0.2, min_samples_leaf=20)
    else:
        classifier = LogisticRegression(C=0.01)
    proba = cross_val_predict(classifier, X, y, method='predict_proba', cv=5)
    proba_home_win = [e[1] for e in proba]
    predictions = [1 if p[1] > 0.478694 else 0 for p in proba]
    
    acc = accuracy_score(y, predictions)
    
    auc = roc_auc_score(y, proba_home_win)
    fpr, tpr, thresholds = roc_curve(y, proba_home_win, pos_label=1)
    
    tnr_plus_tpr = [0.7*(1-fpr[i])+0.3*tpr[i] for i in range(len(fpr))]
    index_max, max_tnr_plus_tpr = max(enumerate(tnr_plus_tpr), key=operator.itemgetter(1))
    thresholds_of_max = thresholds[index_max]
    
    print('Accuracy: %f' % acc)
    print('Max tnrr + tpr: %f' % max_tnr_plus_tpr)
    print('tnr @ max: %f' % (1-fpr[index_max]))
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
