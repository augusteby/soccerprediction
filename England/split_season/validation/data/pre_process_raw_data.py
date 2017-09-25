import pandas as pd
import numpy as np
import os


def get_season_str(filename):
    # ALMOST DUPLICATED FUNCTION (process_data.py)
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


def compute_id(row, season_str):
    """
    This method takes a row from a DataFrame containing
    all the games played during a
    season and computes an id.

    Parameters
    ----------
    row: pandas.Series
        Info related to one game
    season_str: str
        Season (e.g. 2009_2010)

    Returns
    -------
    id_str: str
        Generated ID
    """
    id_str = (row['Div'] + '_'
              + season_str + '_'
              + str(row.name))
    return id_str


RAW_DATA_FOLDER = 'raw'
if __name__ == '__main__':

    for filename in os.listdir(RAW_DATA_FOLDER):
        print(filename)

        # read csv file
        season_str = get_season_str(filename)
        filepath = RAW_DATA_FOLDER + '/' + filename
        one_season = pd.read_csv(filepath)

        # preprocess data
        one_season['id'] = one_season.apply(compute_id,
                                            args=(season_str,),
                                            axis=1)

        # save preprocessed csv file
        new_filepath = 'preprocessed/' + 'prep_' + filename
        one_season.to_csv(new_filepath, index=False)
