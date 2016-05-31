#!/usr/bin/python

import numpy as np
import pandas as pd
import datetime 
from scipy import cluster
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#read in the contents of the csv files 
physio_cavs = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_cavs.csv')
physio_celtics = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_celtics.csv')
physio_kings = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_kings.csv')
physio_lakers = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_lakers.csv')
physio_sixers = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_76ers.csv')
physio_tblazers = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_tblazers.csv')
physio_warriors = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_warriors.csv')
physio_wizards = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/PhysioData_full_wizards.csv')
roster_boxscore = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/Roster_Boxscore_2_OT.csv')
schedule = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/Schedule_2.csv')
opp_team_elo = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/opp_team_elo.csv')
distance_df = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/arena_distances.csv')
clusters = pd.read_csv('/Users/mgiangreco/Dropbox/acl_spring_2016/data/nba_clusters_for_merging.csv')

#combine physio_ csvs into one dataframe
physio_frames = [physio_cavs, physio_celtics, physio_kings, physio_lakers, physio_lakers, physio_sixers, physio_tblazers, physio_warriors, physio_wizards]
physio = pd.concat(physio_frames)
physio.drop_duplicates(inplace=True)

##Perform operations on roster_boxscore to prepare it for join##
#convert values in the "SPLIT_DESC" column of roster_boxscore to numbers
roster_boxscore['SPLIT_DESC'] = roster_boxscore['SPLIT_DESC'].map({'Total':0, 'Quarter 1':1, 'Quarter 2':2, 'Quarter 3':3, 'Quarter 4':4, 'OT':5})

#rename "SPLIT_DESC" column of roster_boxscore to "PERIOD"
roster_boxscore = roster_boxscore.rename(columns={'SPLIT_DESC':'PERIOD'})

#create "PLAYER" and "OPP_TEAM" fields to match physio
roster_boxscore['PLAYER'] = roster_boxscore['MONIKER'] + " " + roster_boxscore['LAST_NAME']
roster_boxscore['OPP_TEAM'] = roster_boxscore['OPP_TEAM_NAME'] + " " + roster_boxscore['OPP_TEAM_NICKNAME']
roster_boxscore = roster_boxscore.drop('MONIKER', axis=1)
roster_boxscore = roster_boxscore.drop('LAST_NAME', axis=1)

#Drop redundant fields from roster_boxscore that already exist in physio
roster_boxscore = roster_boxscore.drop('TEAM_NAME', axis=1)
roster_boxscore = roster_boxscore.drop('TEAM_NICKNAME', axis=1)
roster_boxscore = roster_boxscore.drop('OPP_TEAM_NAME', axis=1)
roster_boxscore = roster_boxscore.drop('OPP_TEAM_NICKNAME', axis=1)
roster_boxscore = roster_boxscore.drop('GAME_DATE', axis=1)
roster_boxscore = roster_boxscore.drop('MINUTES', axis=1)
roster_boxscore = roster_boxscore.drop('POINTS', axis=1)
roster_boxscore = roster_boxscore.drop('ASSISTS', axis=1)
roster_boxscore = roster_boxscore.drop('STEALS', axis=1)
roster_boxscore = roster_boxscore.drop('BLOCKS', axis=1)
roster_boxscore = roster_boxscore.drop('TURNOVERS', axis=1)
roster_boxscore = roster_boxscore.drop('SEASON', axis=1)
roster_boxscore = roster_boxscore.drop('SPLIT_NUMBER', axis=1)


#Create a new dataframe from schedule that has only GAME_CODE and ARENA_NAME
arena = schedule[['GAME_CODE', 'ARENA_NAME']]

#join physio to roster_boxscore and then to arena
s1 = pd.merge(physio, roster_boxscore, on=['GAME_CODE', 'PLAYER', 'PERIOD'])

s1 = pd.merge(s1, arena)

#join in elo
s1 = pd.merge(s1, opp_team_elo, how='left', on=['OPP_TEAM'])

#Calculate TEAM_ASSISTS and TEAM_FIELD_GOALS_MADE by grouping by GAME_CODE, TEAM, and PERIOD and adding up the total in a new column
s1['TEAM_ASSISTS'] = s1.groupby(by=['GAME_CODE', 'TEAM', 'PERIOD'], as_index=False)['ASSISTS'].transform('sum')
s1['TEAM_FIELD_GOALS_MADE'] = s1.groupby(by=['GAME_CODE', 'TEAM', 'PERIOD'], as_index=False)['FIELD_GOALS_MADE'].transform('sum')

#Calculate PACE_FACTOR by taking league avg. pace per minute (2.04) and dividing that by actual pace per minute
s1['PACE_FACTOR'] = 2.04 / (s1.PACE / s1.MINUTES)
s1['PACE_FACTOR'].replace(0, np.nan, inplace=True)

#Calculate unadjusted PER 
s1['PER_U'] = (1 / s1.MINUTES) * (s1.THREE_POINT_MADE + (s1.ASSISTS * 0.67) + (s1.FIELD_GOALS_MADE * (2 - (s1.TEAM_ASSISTS / s1.TEAM_FIELD_GOALS_MADE * 0.589))) + (s1.FREE_THROWS_MADE * 0.5 * (1 + (1 - (s1.TEAM_ASSISTS / s1.TEAM_FIELD_GOALS_MADE)) + ((s1.TEAM_ASSISTS / s1.TEAM_FIELD_GOALS_MADE) * 0.67))) - (s1.TURNOVERS * 1.04) - ((s1.FGA - s1.FGM) * 1.04 * 0.762) - ((s1.FREE_THROWS_ATT - s1.FREE_THROWS_MADE) * 1.04 * 0.44 * (0.44 + (0.56 * 0.762))) + (s1.DEFENSIVE_REBOUNDS * 1.04 * (1 - 0.762)) + (s1.OFFENSIVE_REBOUNDS * 1.04 * 0.762) + (s1.STEALS * 1.04) + (s1.BLOCKS * 1.04 * 0.762) - (s1.FOULS * (0.872 - (1.158 * 0.44 * 1.04))))
s1['PER_U'].replace(0, np.nan, inplace=True)

#Calculate pace-adjusted PER
s1['PER'] = s1.PACE_FACTOR * s1.PER_U
s1['PER'].replace(0, np.nan, inplace=True)

#Calculate assists ratio
s1['ASST_RATIO'] = (s1.ASSISTS*100) / (s1.FREE_THROWS_ATT*0.44 + s1.FIELD_GOALS_ATT + s1.TURNOVERS + s1.ASSISTS)
s1['ASST_RATIO'].replace(0, np.nan, inplace=True)

#calculate Free Throw Attempts per Field Goals Attempted
s1['FTA_PER_FGA'] = s1.FREE_THROWS_ATT / s1.FIELD_GOALS_ATT
s1['FTA_PER_FGA'].replace(0, np.nan, inplace=True)

#Calculate Steals/Min.
s1['STEALS_PER_MIN'] = s1.STEALS / s1.MINUTES
s1['STEALS_PER_MIN'].replace(0, np.nan, inplace=True)

#Calculate blocks/min
s1['BLOCKS_PER_MIN'] = s1.BLOCKS / s1.MINUTES
s1['BLOCKS_PER_MIN'].replace(0, np.nan, inplace=True)

#calculate Turnover ratio
s1['TURNOVER_RATIO'] = (s1.TURNOVERS*100) / (s1.FREE_THROWS_ATT*0.44 + s1.FIELD_GOALS_ATT + s1.TURNOVERS + s1.ASSISTS)
s1['TURNOVER_RATIO'].replace(0, np.nan, inplace=True)

#calculate Usage Rate
s1['USAGE_RATE'] = s1.PACE_FACTOR * ((s1.FREE_THROWS_ATT*0.44 + s1.FIELD_GOALS_ATT + s1.ASSISTS*0.33 + s1.TURNOVERS) / s1.MINUTES)
s1['USAGE_RATE'].replace(0, np.nan, inplace=True)

#Calculate Points per Shot Attempt
s1['PTS_PER_FGA'] = s1.POINTS / s1.FIELD_GOALS_ATT
s1['PTS_PER_FGA'].replace(0, np.nan, inplace=True)

#Calculate Rebounds per Minute
s1['REB_PER_MIN'] = s1.REBOUNDS / s1.MINUTES
s1['REB_PER_MIN'].replace(0, np.nan, inplace=True)

s1.replace(np.inf, np.nan, inplace=True)


#Convert GAME_DATE strings to datetime objects
s1['GAME_DATE'] = pd.to_datetime(s1['GAME_DATE'])


#days since last game
s2 = s1.iloc[:,[0,3]]

s2 = s2.drop_duplicates()

s2 = s2.sort_values(['PLAYER', 'GAME_DATE'])

s2['GAME_DATE'] = pd.to_datetime(s2['GAME_DATE'])

days_back_to_enter = []
date_last = datetime.datetime(2018,1,1)
for index,row in s2.iterrows():
    player_current=row[0]
    date_current=row[1]
    if date_current<=date_last:
        days_back = 0
    else: 
        days_back = date_current - date_last
        days_back = days_back.days
    date_last = date_current
    player_last = player_current
    days_back_to_enter.append(days_back)
    
s2['DAYS_SINCE_LAST_GAME'] = days_back_to_enter

s2['DAYS_SINCE_LAST_GAME'].replace(0, np.nan, inplace=True)
    
#join in days since last game    
s1 = pd.merge(s1, s2, on=['PLAYER', 'GAME_DATE'])

#distance traveled 
player_arena_subset = s1[['PLAYER','GAME_DATE','ARENA_NAME']]

player_arena = player_arena_subset.drop_duplicates()

player_arena = player_arena.sort_values(['PLAYER', 'GAME_DATE'])

travel_distances = []
player = ''
for index,row in player_arena.iterrows(): 
    if row['PLAYER']!=player:
        arena_from = row['ARENA_NAME']
    else:
        arena_from = arena_to
    arena_to = row['ARENA_NAME']
    player = row['PLAYER']
    distance = int(distance_df[((distance_df['FROM_ARENA']==arena_from) & (distance_df['TO_ARENA']==arena_to))].TRAVEL_DIST)
    travel_distances.append(distance)
    
player_arena['TRAVEL_DISTANCE'] = travel_distances

s1 = pd.merge(s1, player_arena, on=['PLAYER', 'GAME_DATE', 'ARENA_NAME'])


#minutes played in last 5 days
s3 = s1[['PLAYER', 'GAME_DATE', 'PERIOD', 'MINUTES']]
s3 = s3.sort_values(['PLAYER', 'GAME_DATE'])

minutes_to_enter = []
for index,row in s3.iterrows():     
  date = row[1]
  subset = s3[(s3['PLAYER'] == row['PLAYER']) & (s3['GAME_DATE'] < date) & (s3['GAME_DATE'] >= (date - (datetime.timedelta(days=5))))]
  mins_last_five_days = subset.MINUTES.sum()
  minutes_to_enter.append(mins_last_five_days)
  
s3['MINS_LAST_FIVE_DAYS'] = minutes_to_enter
s3.drop('MINUTES', axis=1, inplace=True)

s1 = pd.merge(s1, s3, on=['PLAYER', 'GAME_DATE', 'PERIOD'])


#create high_elevation dummy
def high_elevation (row):
   if row['ARENA_NAME'] == 'Pepsi Center' :
      return 1
   elif row['ARENA_NAME'] == 'Vivint Smart Home Arena' :
      return 1 
   else:
      return 0   

s1['HIGH_ELEVATION'] = s1.apply(high_elevation, axis=1)


#convert all column names to lowercase
s1.columns = map(str.lower, s1.columns)


s1 = pd.merge(s1, clusters, how='left', on=['player'])


#create seperate columns for the physio_loads and pers in every period (0-5)

pivoted = pd.pivot_table(s1, values=['physio_load', 'physio_int', 'per', 'minutes', 
'walk_min', 'walk_max', 'walk_time', 'walk_load', 'jog_min', 'jog_max', 'jog_time', 'jog_load',
'run_min', 'run_max', 'run_time', 'run_load', 'sprint_min', 'sprint_max', 'sprint_time', 'sprint_load',
'max_min', 'max_max', 'max_time', 'max_load', 'time_anaerobic', 'percent_anaerobic', 'accels',
'deccels', 'distance', 'off_distance', 'def_distance', 'average_speed', 'off_average_speed', 'def_average_speed',
'accel_1', 'accel_2', 'accel_3', 'accel_4', 'decel_1', 'decel_2', 'decel_3', 'decel_4', 'mechanical_load',
'mechanical_int', 'accel_decel'], index=['game_date', 'player'], columns=['period']).reset_index()

pivoted.columns = ['game_date', 'player', 
'physio_load_0', 'physio_load_1', 'physio_load_2', 'physio_load_3', 'physio_load_4', 'physio_load_5',
 'physio_int_0', 'physio_int_1', 'physio_int_2', 'physio_int_3', 'physio_int_4', 'physio_int_5', 
 'per_0', 'per_1', 'per_2', 'per_3', 'per_4', 'per_5',
 'minutes_0', 'minutes_1', 'minutes_2', 'minutes_3', 'minutes_4', 'minutes_5',
 'walk_min_0', 'walk_min_1', 'walk_min_2', 'walk_min_3', 'walk_min_4', 'walk_min_5', 
 'walk_max_0', 'walk_max_1', 'walk_max_2', 'walk_max_3', 'walk_max_4', 'walk_max_5',
 'walk_time_0', 'walk_time_1', 'walk_time_2', 'walk_time_3', 'walk_time_4', 'walk_time_5',
 'walk_load_0', 'walk_load_1', 'walk_load_2', 'walk_load_3', 'walk_load_4', 'walk_load_5',
'jog_min_0', 'jog_min_1', 'jog_min_2', 'jog_min_3', 'jog_min_4', 'jog_min_5',
'jog_max_0', 'jog_max_1', 'jog_max_2', 'jog_max_3', 'jog_max_4', 'jog_max_5',
'jog_time_0', 'jog_time_1', 'jog_time_2', 'jog_time_3', 'jog_time_4', 'jog_time_5',
'jog_load_0', 'jog_load_1', 'jog_load_2', 'jog_load_3', 'jog_load_4', 'jog_load_5',
'run_min_0', 'run_min_1', 'run_min_2', 'run_min_3', 'run_min_4', 'run_min_5',
'run_max_0', 'run_max_1', 'run_max_2', 'run_max_3', 'run_max_4', 'run_max_5',
'run_time_0', 'run_time_1', 'run_time_2', 'run_time_3', 'run_time_4', 'run_time_5',
'run_load_0', 'run_load_1', 'run_load_2', 'run_load_3', 'run_load_4', 'run_load_5',
'sprint_min_0', 'sprint_min_1', 'sprint_min_2', 'sprint_min_3', 'sprint_min_4', 'sprint_min_5',
'sprint_max_0', 'sprint_max_1', 'sprint_max_2', 'sprint_max_3', 'sprint_max_4', 'sprint_max_5',
'sprint_time_0', 'sprint_time_1', 'sprint_time_2', 'sprint_time_3', 'sprint_time_4', 'sprint_time_5',
'sprint_load_0', 'sprint_load_1', 'sprint_load_2', 'sprint_load_3', 'sprint_load_4', 'sprint_load_5',
'max_min_0', 'max_min_1', 'max_min_2', 'max_min_3', 'max_min_4', 'max_min_5',
'max_max_0', 'max_max_1', 'max_max_2', 'max_max_3', 'max_max_4', 'max_max_5',
'max_time_0', 'max_time_1', 'max_time_2', 'max_time_3', 'max_time_4', 'max_time_5',
'max_load_0', 'max_load_1', 'max_load_2', 'max_load_3', 'max_load_4', 'max_load_5',
'time_anaerobic_0', 'time_anaerobic_1', 'time_anaerobic_2', 'time_anaerobic_3', 'time_anaerobic_4', 'time_anaerobic_5',
'percent_anaerobic_0', 'percent_anaerobic_1', 'percent_anaerobic_2', 'percent_anaerobic_3', 'percent_anaerobic_4', 'percent_anaerobic_5',
'accels_0', 'accels_1', 'accels_2', 'accels_3', 'accels_4', 'accels_5',
'deccels_0', 'deccels_1', 'deccels_2', 'deccels_3', 'deccels_4', 'deccels_5',
'distance_0', 'distance_1', 'distance_2', 'distance_3', 'distance_4', 'distance_5',
'off_distance_0', 'off_distance_1', 'off_distance_2', 'off_distance_3', 'off_distance_4', 'off_distance_5',
'def_distance_0', 'def_distance_1', 'def_distance_2', 'def_distance_3', 'def_distance_4', 'def_distance_5',
'average_speed_0', 'average_speed_1', 'average_speed_2', 'average_speed_3', 'average_speed_4', 'average_speed_5',
'off_average_speed_0', 'off_average_speed_1', 'off_average_speed_2', 'off_average_speed_3', 'off_average_speed_4', 'off_average_speed_5',
'def_average_speed_0', 'def_average_speed_1', 'def_average_speed_2', 'def_average_speed_3', 'def_average_speed_4', 'def_average_speed_5',
'accel_1_0', 'accel_1_1', 'accel_1_2', 'accel_1_3', 'accel_1_4', 'accel_1_5',
'accel_2_0', 'accel_2_1', 'accel_2_2', 'accel_2_3', 'accel_2_4', 'accel_2_5',
'accel_3_0', 'accel_3_1', 'accel_3_2', 'accel_3_3', 'accel_3_4', 'accel_3_5',
'accel_4_0', 'accel_4_1', 'accel_4_2', 'accel_4_3', 'accel_4_4', 'accel_4_5',
'decel_1_0', 'decel_1_1', 'decel_1_2', 'decel_1_3', 'decel_1_4', 'decel_1_5',
'decel_2_0', 'decel_2_1', 'decel_2_2', 'decel_2_3', 'decel_2_4', 'decel_2_5',
'decel_3_0', 'decel_3_1', 'decel_3_2', 'decel_3_3', 'decel_3_4', 'decel_3_5',
'decel_4_0', 'decel_4_1', 'decel_4_2', 'decel_4_3', 'decel_4_4', 'decel_4_5',
'mechanical_load_0', 'mechanical_load_1', 'mechanical_load_2', 'mechanical_load_3', 'mechanical_load_4', 'mechanical_load_5',
'mechanical_int_0', 'mechanical_int_1', 'mechanical_int_2', 'mechanical_int_3', 'mechanical_int_4', 'mechanical_int_5',
'accel_decel_0', 'accel_decel_1', 'accel_decel_2', 'accel_decel_3', 'accel_decel_4', 'accel_decel_5']

pivoted.fillna(value=0, inplace=True)

s1 = pd.merge(s1, pivoted, how='left', on=['player', 'game_date'])

#total physio_load in last 5 days
s4 = s1[['player', 'game_date', 'period', 'physio_load']]
s4 = s4.sort_values(['player', 'game_date'])

physio_to_enter = []
for index,row in s4.iterrows():     
  date = row[1]
  subset = s4[(s4['player'] == row['player']) & (s4['game_date'] < date) & (s4['game_date'] >= (date - (datetime.timedelta(days=5))))]
  physio_last_five_days = subset.physio_load.sum()
  physio_to_enter.append(physio_last_five_days)
  
s4['physio_load_last_five_days'] = physio_to_enter
s4.drop('physio_load', axis=1, inplace=True)

s1 = pd.merge(s1, s4, on=['player', 'game_date', 'period'])

s1 = s1[s1['period']==0]
s1.reset_index(drop=True, inplace=True)

#cluster based on exertion variables
s1_exertion = s1[['physio_load', 'physio_int', 'minutes', 'walk_max', 'walk_time', 'walk_load', 'jog_min', 'jog_max', 'jog_time', 'jog_load',
'run_min', 'run_max', 'run_time', 'run_load', 'sprint_min', 'sprint_max', 'sprint_time', 'sprint_load',
'max_min', 'max_max', 'max_time', 'max_load', 'time_anaerobic', 'percent_anaerobic', 'accels',
'deccels', 'distance', 'off_distance', 'def_distance', 'average_speed', 'off_average_speed', 'def_average_speed',
'accel_1', 'accel_2', 'accel_3', 'accel_4', 'decel_1', 'decel_2', 'decel_3', 'decel_4', 'mechanical_load',
'mechanical_int', 'accel_decel']]
s1_exertion.reset_index(drop=True, inplace=True)


stdsc = StandardScaler()
s1_exertion_std = stdsc.fit_transform(s1_exertion)

s1_exertion_std = pd.DataFrame(s1_exertion_std, columns = s1_exertion.columns)


#elbow graph to determine number of clusters
initial = [cluster.vq.kmeans(s1_exertion_std,i) for i in range(1,10)]
pyplot.plot([var for (cent,var) in initial])
pyplot.show()


#use k-means to split into four clusters 
km = KMeans (n_clusters=4, init='random', n_init=10, max_iter=300, tol=0.0001, random_state=0)
y_km = km.fit_predict(s1_exertion_std)

y_km = y_km + 1
y_km = pd.DataFrame(y_km)
y_km.rename(columns={0: 'clusters_physio_k4'}, inplace=True)

s1 = pd.merge(s1, y_km, left_index=True, right_index=True)


#drop redundant columns 
s1.drop(['period', 'physio_load', 'physio_int', 'per', 'minutes', 
'walk_min', 'walk_max', 'walk_time', 'walk_load', 'jog_min', 'jog_max', 'jog_time', 'jog_load',
'run_min', 'run_max', 'run_time', 'run_load', 'sprint_min', 'sprint_max', 'sprint_time', 'sprint_load',
'max_min', 'max_max', 'max_time', 'max_load', 'time_anaerobic', 'percent_anaerobic', 'accels',
'deccels', 'distance', 'off_distance', 'def_distance', 'average_speed', 'off_average_speed', 'def_average_speed',
'accel_1', 'accel_2', 'accel_3', 'accel_4', 'decel_1', 'decel_2', 'decel_3', 'decel_4', 'mechanical_load',
'mechanical_int', 'accel_decel'], axis=1, inplace=True)