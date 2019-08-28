from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

from data_ingestor import load_X_y
from constants import RANDOM_SEED

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


TRAINING_DATA_FILEPATH = 'data/ML/training_E0_ML.csv'
TESTING_DATA_FILEPATH = 'data/ML/testing_E0_ML.csv'
BATCH_SIZE = 256


def store_reliability_diagram(predictions_df, dpi=100):
    """Method that generates the reliability diagram associated with the predictions made by
    the machine learning algorithm

    Args:
        predictions_df (pandas.DataFrame): Predictions
        dpi (int): The resolution in dot per inch

    Returns:
        None
    """
    (expected_calibration_error, avg_confidences,
     avg_accuracies) = get_expected_calibration_error(predictions_df)

    plt.plot(avg_confidences, avg_confidences, label='perfect calibration')
    plt.plot(avg_confidences, avg_accuracies, label='current model calibration')
    plt.xlabel('Average confidence')
    plt.ylabel('Fraction of correct prediction')
    plt.title('Reliability diagram (ECE = {})'.format(expected_calibration_error))
    plt.legend()
    plt.savefig('nn_reliability_diagram.png', dpi=dpi)
    plt.clf()


def get_expected_calibration_error(proba_values, predictions_df, nb_bins=20):
    """Method that computes the ECE (Expected Calibration Error) associated with the
    predictions of a machine learning model.
    ECE is computed as shown in https://arxiv.org/pdf/1706.04599.pdf

    Args:
        proba_values (pandas.DataFrame): All predictions
        nb_bins (int): Number of bins

    Returns:
        float: ECE (Expected Calibration Error)
        list: For each bin, the average confidence level
        list: For each bin, the average accuracy

    """
    # Select ML predictions only
    selec_pred_df = predictions_df[['output_proba', 'ml_correct']]

    n = len(selec_pred_df)  # total number of samples

    avg_confidences = []
    avg_accuracies = []
    expected_calibration_error = 0

    bins = np.linspace(0, 1, nb_bins)
    is_ml_correct_values = predictions_df['ml_correct'].values

    digitized = np.digitize(proba_values, bins)
    for i in range(1, len(bins)):
        bin_confidences = proba_values[digitized == i]
        bin_is_correct_values = is_ml_correct_values[digitized == i]

        nb_correct_values = list(bin_is_correct_values).count(True)
        bin_size = len(bin_is_correct_values)

        avg_confidence = 0
        avg_accuracy = 0
        if bin_size > 0:
            avg_confidence = round(np.mean(bin_confidences), 2)
            avg_confidences.append(avg_confidence)

            avg_accuracy = nb_correct_values/bin_size
            avg_accuracies.append(avg_accuracy)

        expected_calibration_error += bin_size / n * abs(avg_accuracy - avg_confidence)

    return expected_calibration_error, avg_confidences, avg_accuracies


if __name__=='__main__':

    X_train, y_train, features = load_X_y(TRAINING_DATA_FILEPATH)
    X_test, y_test, _ = load_X_y(TESTING_DATA_FILEPATH)

    # Training Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=RANDOM_SEED)

    # Normalise based on mean and variance of variables in training data
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_val = standardizer.transform(X_val)
    X_test = standardizer.transform(X_test)

    model = Sequential()

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,
                                   restore_best_weights=True)
    callbacks = [early_stopping]

    model.fit(X_train, y_train, epochs=100, batch_size=BATCH_SIZE, callbacks=callbacks,
              validation_data=(X_val, y_val))

    probas = model.predict(X_test)

    print('What is the Area under the ROC curve?')
    print('AUROC: {}'.format(roc_auc_score(y_test, probas)))
    print('')

    print('What are the metrics for a threshold of 0.5')
    predictions = [1 if prob[0]>=0.5 else 0 for prob in probas]
    print('Accuracy @ thresh 0.5: {}'.format(accuracy_score(y_test, predictions)))
    print('Precision @ thresh 0.5: {}'.format(precision_score(y_test, predictions)))
    print('Recall @ thresh 0.5: {}'.format(recall_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))
    print('')

    print('What is the threshold @ maximum precision?')
    precisions, recalls, thresh_prec_rec = precision_recall_curve(y_test, probas)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    idx_max_precision = np.argmax(precisions)
    thresh_max_precision = thresh_prec_rec[idx_max_precision]
    print('Thresh @ Highest Precision: {}'.format(thresh_max_precision))
    print('')

    print('What are the metrics for the threshold @ maximum precision?')
    predictions_max_prec = [1 if prob[0] >= thresh_max_precision else 0 for prob in probas]
    print('Accuracy @ Highest Precision: {}'.format(accuracy_score(y_test, predictions_max_prec)))
    print('Highest Precision: {}'.format(precision_score(y_test, predictions_max_prec)))
    print('Recall @ Highest Precision: {}'.format(recall_score(y_test, predictions_max_prec)))
    print(confusion_matrix(y_test, predictions_max_prec))
    print('')

    print(len(y_test))
    print([i for i in range(len(predictions_max_prec)) if predictions_max_prec[i]==1])

    fpr, tpr, thresholds = roc_curve(y_test, probas)
    plt.plot(fpr, tpr)
    plt.show()







