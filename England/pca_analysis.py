import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILEPATH = 'data/ML/E0_ML_n3.csv'

if __name__ == '__main__':
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    y = data['home_win'].values
    data = data.drop('home_win', 1)

    features = data.columns
    X = data.values

    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    pca = PCA(n_components=15)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
