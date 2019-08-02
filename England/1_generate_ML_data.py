# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:12:07 2017

@author: AUGUSTE
"""
import pandas as pd
import os
from tqdm import tqdm
from constants import (TRAINING_FOLDER, TESTING_FOLDER, WAGES_FOLDER,
                       ML_FOLDER, N, FEAT_TO_KEEP_FOR_ML)


def get_season_str(filename):
    """
    This method returns a string representing the season (e.g. 2009_2010 for
    the 2009-2010 season) from the name of the file containing the results
    of this season
    
    Parameters
    ----------
    filename: str
        Name of the CSV file containing the results of the season.
        It is expected to end with the season in the following format:
        YYYY1_YYYY2.csv where YYYY1 is the year at which the season starts and 
        YYYY2 is the year at which the season ends
        
    Returns
    -------
    season_str: str
        YYYY1_YYYY2
    """
    first_split = filename.split('.')
    championship_YYY1_YYY2 = first_split[0]
    second_split = championship_YYY1_YYY2.split('_')
    YYYY1 = second_split[1]
    YYYY2 = second_split[2]
    
    season_str = YYYY1 + '_' + YYYY2
    return season_str


def generate_all_season_games_features(one_season, data_wages_season, n=3):
    for index, row in one_season.iterrows():
        date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        home_results = generate_results_hometeam_current_season(home_team,
                                                                date,
                                                                one_season,
                                                                n=n)
        away_results = generate_results_awayteam_current_season(away_team,
                                                                date,
                                                                one_season)

        for res in home_results:
            if index==0:
                one_season[res]=pd.Series()
            one_season.set_value(index, res, home_results[res])
        
        for res in away_results:
            if index==0:
                one_season[res]=pd.Series()
            one_season.set_value(index, res, away_results[res])

        # Team Season Wages Bill
        
        if index==0:
            one_season['h_season_wages']=pd.Series()
            one_season['a_season_wages']=pd.Series()
            
        home_season_wages = data_wages_season[data_wages_season['TeamName']==home_team]['WageBillPounds'].values[0]
        away_season_wages = data_wages_season[data_wages_season['TeamName']==away_team]['WageBillPounds'].values[0]
        one_season.set_value(index, 'h_season_wages', home_season_wages)
        one_season.set_value(index, 'a_season_wages', away_season_wages)
                    
    return one_season


def generate_results_hometeam_current_season(team_name, current_date, 
                                             data_season, n=3):
    """
    This method is meant to generate the features of home team that are related
    to the current season: e.g. Number of victories, number of goals this
    season etc. 
    The features are valid for the current date and computed from previous
    games only
    
    Parameters
    ----------
    team_name: str
        Name of the team
    current_date: numpy.datetime64
        Current date
    data_season: pandas.DataFrame
        data of the current season
    """
    # only look at previous games for this season
    previous_games = data_season[data_season['Date'] < current_date]
    
    selected_results = {}
    results_home = get_home_results_current_season(team_name, previous_games, n=n)
    results_away = get_away_results_current_season(team_name, previous_games, n=n)
    nb_victories = results_home['nb_home_victories'] + results_away['nb_away_victories']
    nb_draws = results_home['nb_home_draws'] + results_away['nb_away_draws']

    selected_results['h_nb_victories_total'] = nb_victories
    selected_results['h_nb_draws_total'] = nb_draws
    selected_results['h_season_points'] = 3 * nb_victories + 1 * nb_draws
    selected_results['h_nb_games_total'] = results_away['nb_away_games'] + results_home['nb_home_games']

    selected_results['h_nb_games_home'] = results_home['nb_home_games']
    selected_results['h_nb_victories_home'] = results_home['nb_home_victories']
    selected_results['h_nb_draws_home'] = results_home['nb_home_draws']
    selected_results['h_nb_defeats_home'] = results_home['nb_home_defeats']
    selected_results['h_nb_goals_scored_home'] = results_home['nb_home_goals_scored']
    selected_results['h_mean_nb_goals_scored_home'] = results_home['mean_nb_home_goals_scored']
    selected_results['h_nb_goals_conceded_home'] = results_home['nb_home_goals_conceded']
    selected_results['h_mean_nb_goals_conceded_home'] = results_home['mean_nb_home_goals_conceded']

    selected_results['h_last_n_games_points_home'] = results_home['last_n_games_home_points']

    return selected_results
   
    
def get_home_results_current_season(team_name, previous_games, n=3):
    """This method computes statistics/features regarding the results of the given team based on the
    specified historical data

    :param team_name:
    :param previous_games:
    :param n:
    :return:
    """
    previous_home_games = previous_games[previous_games['HomeTeam'] == team_name]
    nb_home_games = len(previous_home_games)
    results = {'nb_home_games': nb_home_games,
               'nb_home_victories': 0,
               'nb_home_draws': 0,
               'nb_home_defeats': 0,
               'last_n_games_home_points': 0,
               'nb_home_goals_scored': 0,
               'mean_nb_home_goals_scored': 0,
               'nb_home_goals_conceded': 0,
               'mean_nb_home_goals_conceded': 0}

    if nb_home_games > 0:
        results['nb_home_victories'] = len(previous_home_games[previous_home_games['FTR'] == 'H'])
        results['nb_home_draws'] = len(previous_home_games[previous_home_games['FTR'] == 'D'])
        results['nb_home_defeats'] = len(previous_home_games[previous_home_games['FTR'] == 'A'])

        last_n_home_games = previous_home_games.tail(n)
        last_n_home_victories = len(last_n_home_games[last_n_home_games['FTR'] == 'H'])
        last_n_home_draws = len(last_n_home_games[last_n_home_games['FTR'] == 'D'])
        results['last_n_games_home_points'] = 3*last_n_home_victories+1*last_n_home_draws

        results['nb_home_goals_scored'] = previous_home_games['FTHG'].sum()
        results['mean_nb_home_goals_scored'] = previous_home_games['FTHG'].mean()
        results['nb_home_goals_conceded'] = previous_home_games['FTAG'].sum()
        results['mean_nb_home_goals_conceded'] = previous_home_games['FTAG'].mean()
    
    return results


def get_away_results_current_season(team_name, previous_games, n=3): 
    previous_away_games = previous_games[previous_games['AwayTeam'] == team_name]
    nb_away_games = len(previous_away_games)
    results = {'nb_away_games': nb_away_games,
               'nb_away_victories': 0,
               'nb_away_draws': 0,
               'nb_away_defeats': 0,
               'last_n_games_away_points': 0,
               'nb_away_goals_scored': 0,
               'mean_nb_away_goals_scored': 0,
               'nb_away_goals_conceded': 0,
               'mean_nb_away_goals_conceded': 0}
    
    if nb_away_games > 0:
        results['nb_away_victories'] = len(previous_away_games[previous_away_games['FTR'] == 'A'])
        results['nb_away_draws'] = len(previous_away_games[previous_away_games['FTR'] == 'D'])
        
        last_n_away_games = previous_away_games.tail(n)
        last_n_away_victories = len(last_n_away_games[last_n_away_games['FTR'] == 'A'])
        last_n_away_draws = len(last_n_away_games[last_n_away_games['FTR'] == 'D'])
        results['last_n_games_away_points'] = 3*last_n_away_victories+1*last_n_away_draws

        results['nb_away_goals_scored'] = previous_away_games['FTAG'].sum()
        results['mean_nb_away_goals_scored'] = previous_away_games['FTAG'].mean()
        results['nb_away_goals_conceded'] = previous_away_games['FTHG'].sum()
        results['mean_nb_away_goals_conceded'] = previous_away_games['FTHG'].mean()
    
    return results


def generate_results_awayteam_current_season(team_name, current_date, 
                                             data_season, n=3):
    """
    This method is meant to generate the features of away team that are related
    to the current season: e.g. Number of victories, number of goals this
    season etc. 
    The features are valid for the current date and computed from previous
    games only
    
    Parameters
    ----------
    team_name: str
        Name of the team
    current_date: numpy.datetime64
        Current date
    data_season: pandas.DataFrame
        data of the current season
    """
    # only look at previous games for this season
    previous_games = data_season[data_season['Date'] < current_date]
    
    selected_results = {}
    results_home = get_home_results_current_season(team_name, previous_games, n=n)
    results_away = get_away_results_current_season(team_name, previous_games, n=n)
    nb_victories = results_home['nb_home_victories'] + results_away['nb_away_victories']
    nb_draws = results_home['nb_home_draws'] + results_away['nb_away_draws']

    selected_results['a_nb_victories_total'] = nb_victories
    selected_results['a_nb_draws_total'] = nb_draws
    selected_results['a_season_points'] = 3 * nb_victories + 1 * nb_draws
    selected_results['a_nb_games_total'] = results_away['nb_away_games'] + results_home['nb_home_games']

    selected_results['a_nb_games_away'] = results_away['nb_away_games']
    selected_results['a_nb_victories_away'] = results_away['nb_away_victories']
    selected_results['a_nb_draws_away'] = results_away['nb_away_draws']
    selected_results['a_nb_defeats_away'] = results_away['nb_away_defeats']
    selected_results['a_nb_goals_scored_away'] = results_away['nb_away_goals_scored']
    selected_results['a_mean_nb_goals_scored_away'] = results_away['mean_nb_away_goals_scored']
    selected_results['a_nb_goals_conceded_away'] = results_away['nb_away_goals_conceded']
    selected_results['a_mean_nb_goals_conceded_away'] = results_away['mean_nb_away_goals_conceded']

    selected_results['a_last_n_games_points'] = results_away['last_n_games_away_points']

    return selected_results


DATA_TYPES = {'training': {'folder': TRAINING_FOLDER, 'data_table': pd.DataFrame()},
              'testing': {'folder': TESTING_FOLDER, 'data_table': pd.DataFrame()}}

if __name__=='__main__':
    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()

    for data_type in DATA_TYPES:
        print('TYPE: {}'.format(data_type))
        folder = DATA_TYPES[data_type]['folder']
        data_table = DATA_TYPES[data_type]['data_table']
        for filename in tqdm(os.listdir(folder)):
            print('File: {}'.format(filename))

            season_str = get_season_str(filename)
            filepath = folder + '/' + filename
            one_season = pd.read_csv(filepath)

            wages_filepath = WAGES_FOLDER + '/' + season_str + '.csv'
            data_wages_season = pd.read_csv(wages_filepath)

            # convert dates from string to datetime
            one_season['Date'] = pd.to_datetime(one_season.Date)

            # generate the features of each game of the season
            one_season = generate_all_season_games_features(one_season,
                                                            data_wages_season,
                                                            n=N)
            one_season['home_win'] = one_season['FTR'].apply(lambda x: 1 if x=='H' else 0)

            # add current season to all data
            data_table = pd.concat([data_table, one_season])

        # sort all data by date
        data_table = data_table.sort_values(by='Date')
        division = data_table['Div'].values[0]
        data_table_ML = data_table[FEAT_TO_KEEP_FOR_ML]
        filepath_ML = ML_FOLDER + '/' + data_type + '_' + division + '_ML.csv'
        data_table_ML.to_csv(filepath_ML, index=False)
