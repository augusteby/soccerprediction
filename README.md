# soccerprediction
The purpose of this project is to use machine learning in order to predict the results of soccer games.

At the end of a soccer game where the home team is playing against the away team, there are 3 possible outcomes:
1. Home team wins
2. Draw
3. Away team wins

Based on historical data, the first option is the most likely. That is why we focus on predicting whether the home team will win or not.
The machine learning problem that we are trying to solve is thus a classification problem with 2 different classes:
* Class 1: `Home Team Wins`
* Class 2: `Home Team Does not Win` (i.e. the away team wins or it is a draw)

## Data sources
### Games results by season
http://www.football-data.co.uk/data.php

### Wages by season
We expect the wage of a team to be correlated with the number of talented players that team has.
Indeed, to afford the best players, a club usually has to put more money than for average players.

* 2009_2010: 
    * https://www.theguardian.com/football/2011/may/19/premier-league-finances-black-hole
    * http://talksport.com/magazine/features/2011-06-10/premier-league-football-clubs-wage-totals-revealed-where-does-your-club-rank?p=1
    * http://www.les-sports.info/football-championnat-d-angleterre-premier-league-2009-2010-resultats-eprd15021.html
* 2010_2011: https://www.theguardian.com/news/datablog/2012/may/24/football-premier-league-club-accounts
* 2011_2012: https://www.theguardian.com/news/datablog/2013/apr/18/premier-league-club-accounts-debt
* 2012_2013: https://www.theguardian.com/news/datablog/2014/may/01/premier-league-club-accounts-debt-wages#data
* 2013_2014: https://www.theguardian.com/football/2015/apr/29/premier-league-finances-club-by-club
* 2014_2015: https://www.theguardian.com/football/2016/may/25/premier-league-finances-club-by-club-breakdown-david-conn
* 2015_2016: https://www.theguardian.com/football/2017/jun/01/premier-league-finances-club-by-club
* 2016_2017: 
    * https://www.theguardian.com/football/2018/jun/06/premier-league-finances-club-guide-2016-17
    * For Crystal Palace only (not in The Guardian data): http://www.totalsportek.com/money/english-premier-league-wage-bills-club-by-club/
* 2017_2018: https://www.theguardian.com/football/2019/may/22/premier-league-finances-club-guide-2017-18-accounts-manchester-united-city
* 2018_2019: 

###

## Strategy

Few definitions:
* FP (False Positive): The model predicts the victory of the home team but it loses
* TP (True Positive): The model predicts the victory of the home team and it wins
* FN (False Negative): The model predicts that the home team won't win but it wins
* TN (True Negative): The model predicts that the home team won't win and it does not win

We absolutely want to minimise the number of FP. Indeed, if the model tells us that we should bet on the
winning team, we don't want it to be wrong, otherwise it means that we are losing money. 
FN are fine since we won't lose money because of them. Obviously we would prefer a model that has
both low FN and low FP.

From the previous definitions, we can define the following metrics:
* Precision = TP/(TP + FP)
* Recall = TP/(TP + FN)

So we want very high precision. As close as possible to 1.

We also want to make as much money as possible. So we should be able to assess how much money 
we would have won or lost in different scenario:
* We bet starting from the beginning of the season
* We start to bet at a strategic time (to be defined) of the season, i.e. a time starting from which
we would optimise the expected income





