import pandas as pd
import numpy as np
import os

FOLDER='Games'

all_teams = []

for filename in os.listdir(FOLDER):
    print(filename)
    file = FOLDER+'/'+filename
    data_season=pd.read_csv(file)
    team_names=list(np.unique(data_season['HomeTeam'].values))
    all_teams = all_teams+team_names

all_teams=np.unique(all_teams)    
print(all_teams)
for team in all_teams:
    print(team)