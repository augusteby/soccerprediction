import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))

FILEPATH = 'data/ML/E0_ML_n3.csv'
if __name__ == '__main__':
        # load data
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    # extract labels and features
    y = data['home_win'].values
    feature_names = data.drop('home_win', 1).columns
    X = data.drop('home_win', 1).values

    # Standardisation
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    n_all = len(y)
    n_train = int(3 * n_all / 4)
    n_test = n_all - n_train
    y_train = y[0:n_train]
    y_test = y[n_train:n_all]
    X_train = X[0:n_train]
    X_test = X[n_train:n_all]

    # source:
    # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # alpha = 10
    # rule_of_thumb_nb_hidden_layers = 1.0 * len(
    #     X_train) / (alpha * (len(feature_names) + 1))
    # print('You should use less than %f hidden layers' %
    #       rule_of_thumb_nb_hidden_layers)

    # Creation of NN model
    model = Sequential()
    model.add(Dense(5, input_dim=len(feature_names), init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(10, input_dim=len(feature_names), init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    hist = model.fit(X_train, y_train,
                     epochs=1000,
                     batch_size=32, validation_split=0.33)
    print(hist.history.keys())
    score = model.evaluate(X_test, y_test, batch_size=32)
    print('')
    print('')
    print(model.metrics_names)
    print(score)

    hist_train_loss = hist.history['loss']
    hist_val_loss = hist.history['val_loss']
    hist_train_acc = hist.history['acc']
    hist_val_acc = hist.history['val_acc']
    epochs = range(1, len(hist_val_loss) + 1)

    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, hist_train_loss)
    ax1.plot(epochs, hist_val_loss)
    ax1.legend(['train_loss', 'validation_loss'], loc='upper left')

    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, hist_train_acc)
    ax2.plot(epochs, hist_val_acc)
    ax2.legend(['train_acc', 'validation_acc'], loc='upper left')

    plt.show()
