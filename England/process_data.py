# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:12:07 2017

@author: AUGUSTE
"""
import pandas as pd
import numpy as np
import os
import math as m

# constants
FEAT_TO_KEEP_FOR_ML = ['h_nb_games_home', 'h_nb_victories',
                       'h_season_points', 'h_nb_games_total',
                       'h_last_n_games_points', 'h_nb_goals_scored_home', 'h_mean_nb_goals_scored_home',
                       'h_nb_goals_conceded_home', 'h_mean_nb_goals_conceded_home', 'h_season_wages',
                       'a_nb_games_away', 'a_nb_victories_draws',
                       'a_season_points', 'a_nb_games_total',
                       'a_last_n_games_points', 'a_nb_goals_scored_away', 'a_mean_nb_goals_scored_away',
                       'a_nb_goals_conceded_away', 'a_mean_nb_goals_conceded_away', 'a_season_wages',
                       'Month', 'Week',
                       'distance_km', 'capacity_home_stadium',
                       'home_win',
                       'id']

FEAT_TO_KEEP_FOR_ML_DATE = FEAT_TO_KEEP_FOR_ML + ['Date']

FEAT_TO_KEEP_HOME_WIN_ODDS = ['id', 'BbAvH']


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
    prep_championship_YYY1_YYY2 = first_split[0]
    second_split = prep_championship_YYY1_YYY2.split('_')
    YYYY1 = second_split[2]
    YYYY2 = second_split[3]

    season_str = YYYY1 + '_' + YYYY2
    return season_str


def generate_all_season_games_features(one_season, data_wages_season,
                                       data_stadium, n=3):
    """
    This method engineers the features of each game of one season

    Parameters
    ----------
    one_season: pandas.DataFrame
        Results of all game of the season
    data_wages_season: pandas.DataFrame
        Payroll for each club for this season
    data_stadium: pandas.DataFrame
        Data related to stadiums of the clubs
    n: int
        Number of previous games to take into account when calculating features
        for one game

    Returns
    -------
    one_season: pandas.DataFrame
        New features and results for all games of the season
    """
    # index stadium data with TeamName
    data_stadium.index = data_stadium['TeamName']

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
            if index == 0:
                one_season[res] = pd.Series()
            one_season.set_value(index, res, home_results[res])

        for res in away_results:
            if index == 0:
                one_season[res] = pd.Series()
            one_season.set_value(index, res, away_results[res])

        if index == 0:
            one_season['h_season_wages'] = pd.Series()
            one_season['a_season_wages'] = pd.Series()
            one_season['distance_km'] = pd.Series()
            one_season['capacity_home_stadium'] = pd.Series()

        # Team Season Wages Bill
        home_season_wages = data_wages_season[data_wages_season[
            'TeamName'] == home_team]['WageBillPounds'].values[0]
        away_season_wages = data_wages_season[data_wages_season[
            'TeamName'] == away_team]['WageBillPounds'].values[0]
        one_season.set_value(index, 'h_season_wages', home_season_wages)
        one_season.set_value(index, 'a_season_wages', away_season_wages)

        # Distance between two teams
        distance_km = get_distance_between_stadiums(away_team, home_team,
                                                    data_stadium)
        one_season.set_value(index, 'distance_km', distance_km)

        # Capacity of the stadium (can be related to how important the support
        # of local supporter towards the home team will be)
        capacity_home_stadium = data_stadium.loc[home_team]['Capacity']
        one_season.set_value(index, 'capacity_home_stadium',
                             capacity_home_stadium)

    # Create features related to time
    one_season = engineer_features_time(one_season, 'Date')

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
    results_home = get_home_results_current_season(
        team_name, previous_games, n=n)
    results_away = get_away_results_current_season(
        team_name, previous_games, n=n)
    nb_victories = results_home[
        'nb_home_victories'] + results_away['nb_away_victories']
    nb_draws = results_home['nb_home_draws'] + results_away['nb_away_draws']

    selected_results['h_nb_games_home'] = results_home['nb_home_games']
    selected_results['h_nb_victories'] = results_home['nb_home_victories']
    selected_results['h_season_points'] = 3 * nb_victories + 1 * nb_draws
    selected_results['h_nb_games_total'] = results_away[
        'nb_away_games'] + results_home['nb_home_games']
    selected_results['h_last_n_games_points'] = results_home[
        'last_n_games_home_points']
    selected_results['h_nb_goals_scored_home'] = results_home[
        'nb_home_goals_scored']
    selected_results['h_mean_nb_goals_scored_home'] = results_home[
        'mean_nb_home_goals_scored']
    selected_results['h_nb_goals_conceded_home'] = results_home[
        'nb_home_goals_conceded']
    selected_results['h_mean_nb_goals_conceded_home'] = results_home[
        'mean_nb_home_goals_conceded']
    return selected_results


def get_home_results_current_season(team_name, previous_games, n=3):
    previous_home_games = previous_games[
        previous_games['HomeTeam'] == team_name]
    nb_home_games = len(previous_home_games.index)
    results = {'nb_home_games': nb_home_games,
               'nb_home_victories': 0,
               'nb_home_draws': 0,
               'last_n_games_home_points': 0,
               'nb_home_goals_scored': 0,
               'mean_nb_home_goals_scored': 0,
               'nb_home_goals_conceded': 0,
               'mean_nb_home_goals_conceded': 0}

    # if the home team already played at least one game
    if nb_home_games > 0:
        results['nb_home_victories'] = len(
            previous_home_games[previous_home_games['FTR'] == 'H'].index)
        results['nb_home_draws'] = len(
            previous_home_games[previous_home_games['FTR'] == 'D'].index)

        last_n_home_games = previous_home_games.tail(n)
        last_n_home_victories = len(
            last_n_home_games[last_n_home_games['FTR'] == 'H'].index)
        last_n_home_draws = len(
            last_n_home_games[last_n_home_games['FTR'] == 'D'].index)
        results['last_n_games_home_points'] = 3 * \
            last_n_home_victories + 1 * last_n_home_draws
        results['nb_home_goals_scored'] = previous_home_games['FTHG'].sum()
        results['mean_nb_home_goals_scored'] = float(
            results['nb_home_goals_scored']) / nb_home_games
        results['nb_home_goals_conceded'] = previous_home_games['FTAG'].sum()
        results['mean_nb_home_goals_conceded'] = float(
            results['nb_home_goals_conceded']) / nb_home_games

    return results


def get_away_results_current_season(team_name, previous_games, n=3):
    previous_away_games = previous_games[
        previous_games['AwayTeam'] == team_name]
    nb_away_games = len(previous_away_games.index)
    results = {'nb_away_games': nb_away_games,
               'nb_away_victories': 0,
               'nb_away_draws': 0,
               'last_n_games_away_points': 0,
               'nb_away_goals_scored': 0,
               'mean_nb_away_goals_scored': 0,
               'nb_away_goals_conceded': 0,
               'mean_nb_away_goals_conceded': 0}

    if nb_away_games > 0:
        results['nb_away_victories'] = len(
            previous_away_games[previous_away_games['FTR'] == 'A'].index)
        results['nb_away_draws'] = len(
            previous_away_games[previous_away_games['FTR'] == 'D'].index)

        last_n_away_games = previous_away_games.tail(n)
        last_n_away_victories = len(
            last_n_away_games[last_n_away_games['FTR'] == 'A'].index)
        last_n_away_draws = len(
            last_n_away_games[last_n_away_games['FTR'] == 'D'].index)
        results['last_n_games_away_points'] = 3 * \
            last_n_away_victories + 1 * last_n_away_draws
        results['nb_away_goals_scored'] = previous_away_games['FTAG'].sum()
        results['mean_nb_away_goals_scored'] = float(
            results['nb_away_goals_scored']) / nb_away_games
        results['nb_away_goals_conceded'] = previous_away_games['FTHG'].sum()
        results['mean_nb_away_goals_conceded'] = float(
            results['nb_away_goals_conceded']) / nb_away_games

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

    Returns
    -------
    selected_results: dict
       Summary of the results from previous games
    """
    # only look at previous games for this season
    previous_games = data_season[data_season['Date'] < current_date]

    selected_results = {}
    results_home = get_home_results_current_season(
        team_name, previous_games, n=n)
    results_away = get_away_results_current_season(
        team_name, previous_games, n=n)
    nb_victories = results_home[
        'nb_home_victories'] + results_away['nb_away_victories']
    nb_draws = results_home['nb_home_draws'] + results_away['nb_away_draws']

    selected_results['a_nb_games_away'] = results_away['nb_away_games']
    selected_results['a_nb_victories_draws'] = results_away[
        'nb_away_victories'] + results_away['nb_away_draws']
    selected_results['a_season_points'] = 3 * nb_victories + 1 * nb_draws
    selected_results['a_nb_games_total'] = results_away[
        'nb_away_games'] + results_home['nb_home_games']
    selected_results['a_last_n_games_points'] = results_away[
        'last_n_games_away_points']
    selected_results['a_nb_goals_scored_away'] = results_away[
        'nb_away_goals_scored']
    selected_results['a_mean_nb_goals_scored_away'] = results_away[
        'mean_nb_away_goals_scored']
    selected_results['a_nb_goals_conceded_away'] = results_away[
        'nb_away_goals_conceded']
    selected_results['a_mean_nb_goals_conceded_away'] = results_away[
        'mean_nb_away_goals_conceded']
    return selected_results


def engineer_features_time(data, date_column_name):
    """
    The purpose of this method is to create new features related to the date

    Parameters
    ----------
    data: pandas.DataFrame
        Input data
    date_column_name: str
        Name of the column that contains dates

    Returns
    -------
    data: pandas.DataFrame
        Output data with new time-related features
    """
    data['Month'] = data[date_column_name].apply(lambda x: x.month)
    data['Week'] = data[date_column_name].apply(lambda x: x.isocalendar()[1])

    return data


def get_distance_between_stadiums(team1, team2, stadium_data):
    """
    The purpose of this method is to compute and return the distance between 
    the GPS coordinates of the stadiums of two teams

    Source: http://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    team1: str
        Name of first team
    team2: str
        Name of second team
    stadium_data: pandas.DataFrame
        Contains gps coordinates of all teams in the database indexed by 
        TeamName

    Returns
    -------
    distance: float
        Distance between the stadiums of the two teams
    """
    lon1 = stadium_data.loc[team1]['Longitude']
    lat1 = stadium_data.loc[team1]['Latitude']

    lon2 = stadium_data.loc[team2]['Longitude']
    lat2 = stadium_data.loc[team2]['Latitude']

    R = 6371  # Radius of the Earth in km
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (m.sin(dlat / 2) * m.sin(dlat / 2) +
         m.sin(dlon / 2) * m.sin(dlon / 2) * m.cos(lat1) * m.cos(lat2))

    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    distance = R * c
    return distance

    # parameters
DATA_FOLDER = 'data'
GAMES_FOLDER = DATA_FOLDER + '/Games/preprocessed'
WAGES_FOLDER = DATA_FOLDER + '/Wages'
STADIUM_FILEPATH = DATA_FOLDER + '/Stadium/stadiums_modified2.csv'
ML_FOLDER = DATA_FOLDER + '/ML'
N = 3  # number of previous games to consider
if __name__ == '__main__':
    all_data = pd.DataFrame()
    stadium_data = pd.read_csv(STADIUM_FILEPATH)

    for filename in os.listdir(GAMES_FOLDER):
        print(GAMES_FOLDER)
        print(filename)
        season_str = get_season_str(filename)
        filepath = GAMES_FOLDER + '/' + filename
        one_season = pd.read_csv(filepath)

        wages_filepath = WAGES_FOLDER + '/' + season_str + '.csv'
        data_wages_season = pd.read_csv(wages_filepath)

        # convert dates from string to datetime
        one_season['Date'] = pd.to_datetime(one_season.Date, format='%d/%m/%y')

        # generate the features of each game of the season
        one_season = generate_all_season_games_features(one_season,
                                                        data_wages_season,
                                                        stadium_data,
                                                        n=N)
        one_season['home_win'] = one_season[
            'FTR'].apply(lambda x: 1 if x == 'H' else 0)
        # break

        # add current season to all data
        all_data = pd.concat([all_data, one_season])

    # sort all data by date
    all_data = all_data.sort_values(by='Date')

    division = all_data['Div'].values[0]

    all_data_ML = all_data[FEAT_TO_KEEP_FOR_ML]
    filepath_ML = ML_FOLDER + '/' + division + '_ML.csv'
    all_data_ML.to_csv(filepath_ML, index=False)

    # Include date
    all_data_ML_date = all_data[FEAT_TO_KEEP_FOR_ML_DATE]
    filepath_ML_date = ML_FOLDER + '/' + division + '_ML_date.csv'
    all_data_ML_date.to_csv(filepath_ML_date, index=False)

    # Home win odds data
    all_data_home_win_odds = all_data[FEAT_TO_KEEP_HOME_WIN_ODDS]
    filepath_home_win_odds = ML_FOLDER + '/' + division + '_home_win_odds.csv'
    all_data_home_win_odds.to_csv(filepath_home_win_odds, index=False)
