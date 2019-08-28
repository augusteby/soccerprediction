# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:17:04 2017

@author: AUGUSTE
"""
import numpy as np
import matplotlib

# Use TkAgg backend with matplotlib because the backend by default might cause the following issue:
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
# More info here: https://matplotlib.org/faq/usage_faq.html
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

from data_ingestor import load_X_y

TRAINING_DATA_FILEPATH = 'data/ML/training_E0_ML.csv'
TESTING_DATA_FILEPATH = 'data/ML/testing_E0_ML.csv'

ALGO = 'svm'
if __name__=='__main__':
    X_train, y_train, features = load_X_y(TRAINING_DATA_FILEPATH)
    X_test, y_test, _ = load_X_y(TESTING_DATA_FILEPATH)

    # Normalise based on mean and variance of variables in training data
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)

    if ALGO == 'rdmf':
        classifier = RandomForestClassifier(n_estimators=50, max_features=0.2, min_samples_leaf=20)
    elif ALGO == 'svm':
        classifier = SVC(C=0.1, gamma=0.001)
        # classifier = SVC(C=0.1, gamma=0.01, kernel='poly', degree=3, tol=0.001)
    else:
        classifier = LogisticRegression(C=0.01)

    if ALGO == 'svm':
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        print('What are the metrics with probability==FALSE')
        print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
        print('Precision: {}'.format(precision_score(y_test, predictions)))
        print('Recall: {}'.format(recall_score(y_test, predictions)))
        print(confusion_matrix(y_test, predictions))
        print('')
    else:
        classifier.fit(X_train, y_train)
        probas = classifier.predict_proba(X_test)
        proba_home_win = [p[1] for p in probas]

        auc = roc_auc_score(y_test, proba_home_win)
        fpr, tpr, thresholds = roc_curve(y_test, proba_home_win, pos_label=1)

        print('What is the Area under the ROC curve?')
        print('AUROC: {}'.format(auc))
        print('')

        print('What are the metrics for a threshold of 0.5')
        predictions = [1 if prob[1] >= 0.5 else 0 for prob in probas]
        print('Accuracy @ thresh 0.5: {}'.format(accuracy_score(y_test, predictions)))
        print('Precision @ thresh 0.5: {}'.format(precision_score(y_test, predictions)))
        print('Recall @ thresh 0.5: {}'.format(recall_score(y_test, predictions)))
        print(confusion_matrix(y_test, predictions))
        print('')

        print('What is the threshold @ maximum precision?')
        precisions, recalls, thresh_prec_rec = precision_recall_curve(y_test, proba_home_win)
        precisions = precisions[:-1]
        recalls = recalls[:-1]
        idx_max_precision = np.argmax(precisions)
        thresh_max_precision = thresh_prec_rec[idx_max_precision]
        print('Thresh @ Highest Precision: {}'.format(thresh_max_precision))
        print('')

        print('What are the metrics for the threshold @ maximum precision?')
        predictions_max_prec = [1 if prob[1] >= thresh_max_precision else 0 for prob in probas]
        print(
            'Accuracy @ Highest Precision: {}'.format(accuracy_score(y_test, predictions_max_prec)))
        print('Highest Precision: {}'.format(precision_score(y_test, predictions_max_prec)))
        print('Recall @ Highest Precision: {}'.format(recall_score(y_test, predictions_max_prec)))
        print(confusion_matrix(y_test, predictions_max_prec))
        print('')

        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.show()

    if ALGO == 'rdmf':
        print(features)
        print(classifier.feature_importances_)
    elif ALGO == 'svm':
        pass
    else:
        for i in range(len(features)):
            print(features[i])
            print(classifier.coef_[0][i])
            print('')
