# -*- coding: utf-8 -*-
"""
Created on Sun May  7 13:52:15 2017

@author: AUGUSTE
"""

import requests
API_TOKEN = '278750e82d7443efab2b677f36731df4'
HTTP_HEADER = 'X-Auth-Token'
if __name__=='__main__':
    headers = {HTTP_HEADER: API_TOKEN,'X-Response-Control': 'minified'}
    url = 'http://api.football-data.org/v1/competitions/398/teams/?season=2016'
    r = requests.get(url, headers=headers)
    print(r.json().keys())
    print(r.json()['count'])
    print(r.headers)
    print(r.json()['teams'][0])
    