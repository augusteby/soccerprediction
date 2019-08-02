from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from data_ingestor import load_X_y
from constants import RANDOM_SEED

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


TRAINING_DATA_FILEPATH = 'data/ML/training_E0_ML.csv'
TESTING_DATA_FILEPATH = 'data/ML/testing_E0_ML.csv'
BATCH_SIZE = 256

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

    predictions = [1 if prob[0]>=0.5 else 0 for prob in probas]

    print(accuracy_score(y_test, predictions))
    print(roc_auc_score(y_test, probas))
    fpr, tpr, thresholds = roc_curve(y_test, probas)

    plt.plot(fpr, tpr)
    plt.show()







