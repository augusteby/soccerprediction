import pandas as pd
import numpy as np
import tweepy as tp 
import settings as s

TEAM_NAMES = {
'Arsenal':'Arsenal',
'Aston Villa':'AVFCOfficial',
'Birmingham':'BCFC',
'Blackburn':'Rovers',
'Blackpool':'BlackpoolFC',
'Bolton':'OfficialBWFC',
'Bournemouth':'afcbournemouth',
'Burnley':'BurnleyOfficial',
'Cardiff':'CardiffCityFC',
'Chelsea':'ChelseaFC',
'Crystal Palace':'CPFC',
'Everton':'Everton',
'Fulham':'FulhamFC',
'Hull':'HullCity',
'Leicester':'LCFC',
'Liverpool':'LFC',
'Man City':'ManCity',
'Man United':'ManUtd',
'Middlesbrough':'Boro',
'Newcastle':'NUFC',
'Norwich':'NorwichCityFC',
'Portsmouth':'officialpompey',
'QPR':'queensparkfc',
'Reading':'ReadingFC',
'Southampton':'SouthamptonFC',
'Stoke':'stokecity',
'Sunderland':'SunderlandAFC',
'Swansea':'SwansOfficial',
'Tottenham':'SpursOfficial',
'Watford':'WatfordFC',
'West Brom':'WBA',
'West Ham':'WestHamUtd',
'Wigan':'LaticsOfficial',
'Wolves':'Wolves'}

auth = tp.OAuthHandler(s.API_KEY, s.API_SECRET)
auth.set_access_token(s.ACCESS_TOKEN, s.ACCESS_TOKEN_SECRET)

api = tp.API(auth)

names = []
tweet_account_names = []
date_creation = []
followers_count = []

for name in TEAM_NAMES:
    print(name)
    names.append(name)
    
    tweet_account=TEAM_NAMES[name]
    tweet_account_names.append(tweet_account)
    
    info_team=api.get_user(tweet_account)
    date_creation.append(info_team.created_at)
    followers_count.append(info_team.followers_count)
    

tweet_info_dict = {'TeamName':names,'TwitterAccount':tweet_account_names,
'CreationDate':date_creation,'NumberFollowers':followers_count}

tweet_info_df = pd.DataFrame(tweet_info_dict)

tweet_info_df.to_csv('tweet_info.csv',index=False)


#if __name__=='__main__':
    