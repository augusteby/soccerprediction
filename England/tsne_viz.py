import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
from sklearn.manifold import TSNE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def _get_tsne_best_model(X, dim_red_space=2,
                         learning_rate_val=250, perplexity_val=50,
                         nb_iterations=10):
    """
    Performs the t-SNE algorithm multiple times on the dataset to return
    the model (data reduction) with the lowest KL score

    Non-convexity of the cost function leads to high variability in KL
    scores over several computation of t-SNE. Thus, we extract the best
    model out of OUTER_ITERATIONS iterations of the algorithm.

    t-SNE is performed using the TSNE function from sklearn.manifold
    package TSNE parameters are kept constant through iterations
    Barnes-Hut t-SNE is the default algorithm implemented in TSNE

    Parameters
    ----------
    X: numpy.ndarray
        Matrix of features X
    dim_red_space: int (optional, default 2)
        Dimension of the reduced space
        WARNING: only 2D reduction available
    learning_rate_val: float (optional, default 250)
        Learning rate of the t-SNE algorithm, parameter of TSNE function
        Advised range: [100, 1000]
    perplexity_val: float (optional, default 50)
        Perplexity of the t-SNE algorithm, parameter of TSNE function
        Advised range: [5, 50]
    nb_iterations: int (optional, default 10)
        Number of iterations of the t-SNE algorithm

    Returns
    -------
    best_model: TSNE
        Model (TSNE class instance) that had the lowest KL score over
        iterations
        Attributes are kl_divergence_ (: float) that contains the KL score of
        the model, and embedding_ (: numpy.ndarray) the representation of data
        in the reduced space
    """
    best_score_kl_diver = np.Inf
    best_model = None

    for ind_iter in range(nb_iterations):
        # Default model parameters of TSNE are n_components=2, perplexity=30.0,
        # early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
        # n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean',
        # init='random', verbose=0, random_state=None, method='barnes_hut',
        # angle=0.5
        current_model = TSNE(n_components=dim_red_space,
                             perplexity=perplexity_val,
                             early_exaggeration=4.0,
                             learning_rate=learning_rate_val, n_iter=1000,
                             n_iter_without_progress=30, min_grad_norm=1e-07,
                             metric='euclidean', init='random', verbose=0,
                             random_state=None, method='barnes_hut', angle=0.5)

        # Apply the t-SNE algorithm
        X_trans = current_model.fit_transform(X)
        score_kl_diver = current_model.kl_divergence_

        if score_kl_diver <= best_score_kl_diver:
            best_score_kl_diver = score_kl_diver
            best_model = current_model

    return best_model


def run_tsne_grid_search(X, y_labels, reference_dict=None,
                         dim_red_space=3,
                         forced_perplexity=50, forced_learning_rate=250,
                         optimi_parameter='', op_parameter_grid=[],
                         op_plot=True, op_print=True,
                         nb_outer_iterations=10):
    """
    Performs a grid search to find the t-SNE data representation that
    displays the lowest KL score over the parameter mesh/grid

    WARNING: currently only 1-dimensional grid search is available
    Only two TSNE parameters can be optimised: perplexity and learning rate

    Parameters
    ----------
    X: numpy.ndarray
        Array of features
    y_labels: numpy.array
        Vector of labels strictly associated to X
    reference_dict: str dict
        String dictionary that relates Test identifiers to their full name
        (str)
    dim_red_space: int  (optional, default 2)
        Dimension of the reduced space
        WARNING: only 2D reduction available
    forced_perplexity: float (optional, default 5)
        Perplexity of the t-SNE algorithm, parameter of TSNE function
        Advised range: [5, 50]
    forced_learning_rate: float (optional, default 250)
        Learning rate of the t-SNE algorithm, parameter of TSNE function
        Advised range: [100, 1000]
    optimi_parameter: str (optional, default '')
        Parameter that is tuned during optimisation process/grid search
        Use either None, 'perplexity' or 'learning_rate'
    op_parameter_grid: numpy.array (optional, default [])
        Array of the values that can be taken by the optimi_parameter
        It defines the grid/mesh for the grid search.
        ex: np.linspace(5, 50, 20, endpoint=True)
    op_plot: bool (optional, default True)
        Option for scatter plotting the 2D best representation
    op_print: bool (optional, default True)
        Option for printing the resulting best KL score
    nb_outer_iterations: int (optional, default 10)
        Number of iterations of the t-SNE algorithm in the get_best_tsne
        function

    Returns
    -------
    result_list: list of dict
        List of dict entries
        Format of dict: {'kl_score': score_kl_diver, 'perplexity': ele}
        Each dict entry contains the parameters of the t-SNE, and the
        resulting
        KL score computed with the function get_best_tsne
    """
    if ((optimi_parameter != '')
            & (op_parameter_grid == [])):
        result_list = []
        print('Input Error: op_parameter_grid must be assigned in '
              + 'run_tsne_grid_search.')
        return result_list

    if ((optimi_parameter == '')
            & (op_parameter_grid == [])):
        # No grid search required
        best_model = _get_tsne_best_model(X,
                                          dim_red_space=dim_red_space,
                                          perplexity_val=forced_perplexity,
                                          learning_rate_val=forced_learning_rate,
                                          nb_iterations=nb_outer_iterations)
        score_kl_diver = best_model.kl_divergence_
        X_trans = best_model.embedding_

        best_parameter_set = {'Grid search': False,
                              'KL score': score_kl_diver,
                              'Perplexity': forced_perplexity,
                              'Learning rate': forced_learning_rate}
        result_list = [best_parameter_set]

    else:
        # Multi-dimensional grid search
        # 1-D search only currently supported
        nb_dim_grid = 1

        # Creation of the result list
        result_list = []
        best_kl_score = np.inf
        best_model = None
        best_parameter_set = None

        # Assuming only 1 optimisation parameter
        # Either 'perplexity' or 'learning_rate'
        if nb_dim_grid == 1:

            grid = op_parameter_grid

            for ele in grid:

                if optimi_parameter == 'perplexity':
                    model = _get_tsne_best_model(X,
                                                 dim_red_space=dim_red_space,
                                                 perplexity_val=ele,
                                                 nb_iterations=nb_outer_iterations)

                else:
                    model = _get_tsne_best_model(X,
                                                 dim_red_space=dim_red_space,
                                                 learning_rate_val=ele,
                                                 nb_iterations=nb_outer_iterations)

                score_kl_diver = model.kl_divergence_
                X_trans = model.embedding_

                # Result list [{cost : _ , param_1 : _ ,...},...]
                # as dict list
                if optimi_parameter == 'perplexity':
                    dict_entry = {'Grid search': True,
                                  'KL score': score_kl_diver,
                                  'Perplexity': ele,
                                  'Learning rate': forced_learning_rate}
                else:
                    dict_entry = {'Grid search': True,
                                  'KL score': score_kl_diver,
                                  'Perplexity': forced_perplexity,
                                  'Learning rate': ele}

                result_list.append(dict_entry)

                # Computation of the best set of parameters
                if score_kl_diver <= best_kl_score:
                    best_model = model
                    best_kl_score = score_kl_diver
                    best_parameter_set = dict_entry

        else:
            result_list = []
            print('t-SNE aborted. Currently, t-SNE only supports 1-D grid'
                  + ' search.')

    if (op_print & (not result_list)):
        str_to_print = '\n____________Best parameter set is____________\n'
        str_to_print += best_parameter_set
        print str_to_print

    if (op_plot & dim_red_space <= 3 & (not result_list)):
        # Scatter plot
        y_stages = y_labels.flatten()
        print(X_trans)
        print(y_stages)
        df = pd.DataFrame(
            dict(x=X_trans[0::, 0], y=X_trans[0::, 1], z=X_trans[0::, 2],
                 label=y_stages[0::]))

        groups = df.groupby('label')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # fig, ax = plt.subplots()
        # Optional, just adds 5% padding to the autoscaling
        ax.margins(0.05)
        for name, group in groups:

            ax.scatter(group.x, group.y, group.z, marker='.')
            plt.draw()

        ax.legend(numpoints=1, loc='upper left')
        # Get the full name of the test as a string
        plt.title('\nt-SNE reduction\n'
                  + str(best_parameter_set))
        plt.xlabel('1st axis, undefined unit')
        plt.ylabel('2nd axis, undefined unit')
        plt.show()

    return result_list


FILEPATH = 'data/ML/E0_ML_n3.csv'
ODDS_FILEPATH = 'data/ML/E0_home_win_odds.csv'
FEATURES_LOG = ['h_nb_victories', 'h_season_points',
                'a_nb_victories_draws', 'a_season_points']

SELECTED_CLASSIFIER = 'logreg'

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
                    'logreg': ['h_nb_goals_diff', 'h_season_wages', 'a_nb_goals_diff', 'a_season_wages'],
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

PLOT_TSNE_ENABLED = False  # True to plot t-SNE visualization result
PRINT_BEST_TSNE_SCORE = False  # True to print t-SNE parameters after grid search
# Definition of the t-SNE parameter grid - perplexity, learning rate...
TSNE_PARAM_GRID = {'perplexity': np.linspace(5, 50, 2, endpoint=True),
                   'learning_rate': np.linspace(100, 1000, 2, endpoint=True),
                   '': []}
# Definition of the tuning parameter actually used in t-SNE
TSNE_TUNED_PARAMETER = ''
TSNE_OUTER_ITERATIONS = 10  # Number of t-SNE iterations to select best model

if __name__ == '__main__':
    data_odds = pd.read_csv(ODDS_FILEPATH)
    data = pd.read_csv(FILEPATH)

    # store id of games
    id_str = data['id'].values
    data = data.drop('id', 1)

    # if ('Date' in set(data.columns.values)):
    #     data['Date'] = pd.to_datetime(data['Date'])

    # data = data[data['h_nb_games_total']>18]

    # encode categorical data
    # if 'Month' in data.columns.values:
    #     data = pd.get_dummies(data, columns=['Month'])
    # if 'Week' in data.columns.values:
    #     data = pd.get_dummies(data, columns=['Week'])

    y = data['home_win'].values
    data = data.drop('home_win', 1)

    # data = data[FEATURES_TO_KEEP[SELECTED_CLASSIFIER]]

    # for feat in FEATURES_LOG:
    # data[feat] = data[feat].apply(lambda x: np.log10(1+x))
    features = data.columns
    X = data.values

    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    start_time = time.time()

    print('Computation of TSNE...')
    result_list = run_tsne_grid_search(
        X,
        y,
        dim_red_space=3,
        forced_perplexity=50,
        forced_learning_rate=800,
        optimi_parameter=TSNE_TUNED_PARAMETER,
        op_parameter_grid=TSNE_PARAM_GRID[
            TSNE_TUNED_PARAMETER],
        op_plot=PLOT_TSNE_ENABLED,
        op_print=PRINT_BEST_TSNE_SCORE,
        nb_outer_iterations=TSNE_OUTER_ITERATIONS)

    print('Elapsed time:', time.time() - start_time)
