LABEL = 'home_win'

FEAT_TO_KEEP_FOR_ML = [
    # Home Team Features
    'h_nb_victories_total', 'h_nb_draws_total', 'h_season_points', 'h_nb_games_total',
    'h_nb_games_home', 'h_nb_victories_home', 'h_nb_draws_home', 'h_nb_defeats_home',
    'h_nb_goals_scored_home', 'h_mean_nb_goals_scored_home',
    'h_nb_goals_conceded_home', 'h_mean_nb_goals_conceded_home',
    'h_last_n_games_points_home',
    'h_season_wages',

    # Away Team Features
    'a_nb_victories_total', 'a_nb_draws_total', 'a_season_points', 'a_nb_games_total',
    'a_nb_games_away', 'a_nb_victories_away', 'a_nb_draws_away', 'a_nb_defeats_away',
    'a_nb_goals_scored_away', 'a_mean_nb_goals_scored_away',
    'a_nb_goals_conceded_away', 'a_mean_nb_goals_conceded_away',
    'a_last_n_games_points',
    'a_season_wages',

    # Label
    LABEL]


DATA_FOLDER = 'data'
GAMES_FOLDER = DATA_FOLDER + '/Games'
TRAINING_FOLDER = GAMES_FOLDER + '/training'
TESTING_FOLDER = GAMES_FOLDER + '/testing'
WAGES_FOLDER = DATA_FOLDER + '/Wages'
ML_FOLDER = DATA_FOLDER + '/ML'
N = 3 # number of previous games to consider

RANDOM_SEED = 17