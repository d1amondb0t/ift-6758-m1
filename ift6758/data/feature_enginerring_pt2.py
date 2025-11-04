import sys, os 
sys.path.append(os.path.abspath('..'))
from ift6758.data import load_game
from ift6758.data.tidying import load_all_games_events
from ift6758.data.tidying  import events_to_dataframe2
import json
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from ift6758.data.acquisition import read_json


def _calculate_rebound(df):
    #not sure again which shots we should be looking at so just including all kinds of shots here 
    #By milestone 2 definition we are just checking if the previous shot was taken or not, if it was a shot then it's a rebound 
    df['rebound'] = df['previous_event_name'].isin(['shot-on-goal', 'missed-shot'])
    return df 

def _lasteventtime(df):
    ''' Takes the previous event time period and converts it to the seconds'''
    minutes = df['previous_event_timeperiod'].str.split(':', expand=True)[0].astype(int)
    seconds = df['previous_event_timeperiod'].str.split(':', expand=True)[1].astype(int)
    period_seconds = minutes * 60 + seconds
    df['previous_event_timeseconds'] = period_seconds
    return df 

def _curenteventtime(df):
    '''takes the current event time period and converts it into seconds compared '''
    minutes = df['period_time'].str.split(':', expand=True)[0].astype(int)
    seconds = df['period_time'].str.split(':', expand=True)[1].astype(int)
    period_seconds = minutes * 60 + seconds
    df['current_event_timeseconds'] = period_seconds
    return df 

def _computetimesincelastevent(df):
    '''Using the lasteventtime - currenteventtime to compute the time since last event'''
    df['time_since_last_event'] =  df['current_event_timeseconds'] - df['previous_event_timeseconds'] 
    return df

def _compute_last_event_distance(df):
    '''Using the last event distance we compute the distance between the previous event and the current event'''
    df['distance_from_last_event'] = np.sqrt( (df['previous_event_y'] - df['coordinates_y'] )**2 +( df['previous_event_x'] - df['coordinates_x'])**2)
    return df

def _compute_speed(df):
    '''from the __compute_last_event_distance__ we just compute the speed with the time since the last event'''
    df.loc[df['time_since_last_event'] != 0, 'speed'] = df['distance_from_last_event'] / df['time_since_last_event']
    return df 

def _compute_angle_change(df):
    same_section = np.sign(df['coordinates_y']) == np.sign(df['previous_event_y'])
    df.loc[(df['rebound'] == True) & (same_section), "angle_change"] =  np.abs(df['angle_shot'] -  df['angle_shot_prev']) #Not entirely sure about this, i would assume it's just a difference in shot angles but the graphic seems to be showing something like this 
    df.loc[(df['rebound'] == True) & (~same_section), "angle_change"]=   (np.abs(df['angle_shot'] +  df['angle_shot_prev'])) #Not entirely sure about this, i would assume it's just a difference in shot angles but the graphic seems to be showing something like this 
    df.loc[(df['rebound'] == False, "angle_change")] = 0.0
    return df 

def _compute_shot_angle(df):
    '''taking the shot angles for this '''
    net_x, net_y = 89, 0
    x_diff  = np.abs( net_x - df['coordinates_x']) 
    y_diff  =np.abs ( df['coordinates_y']) 
    x_diff2 = np.abs( net_x - df['previous_event_x'])
    y_diff2 = np.abs ( df['previous_event_y']) 
    df['angle_shot_prev']  = 0.0
    df['angle_shot'] = np.degrees(np.arctan2(y_diff, x_diff))
    df.loc[df['rebound'] == True, "angle_shot_prev"]  =  np.degrees(np.arctan2(y_diff2, x_diff2))
    return df 

def _compute_shot_distance(df):
    net_x, net_y = 89, 0
    net_x2,net_y2 = -89,0 
    df['distance1'] = np.sqrt((df['coordinates_x'] - net_x)**2 + (df['coordinates_y'] - net_y)**2)
    df['distance2'] = np.sqrt((df['coordinates_x'] - net_x2)**2 + (df['coordinates_y'] - net_y2)**2)
    df.loc[df['zone_code'] == 'O', 'distance_shot'] = df[['distance1', 'distance2']].min(axis=1)
    df.loc[df['zone_code'] != 'O', 'distance_shot'] = df[['distance1', 'distance2']].max(axis=1)
    df.drop(columns=['distance1', 'distance2'], inplace=True)
    return df

def _compute_powerplay_features(df):
    '''TODO need to get all the penalty events from the JSON and then join them figure out how to track them to the goal'''

    return df 

def feature_engineering_two(years):
    csv_path = "../ift6758/data/allshotgoals2.csv"
    if (os.path.exists(csv_path)):
            df = pd.read_csv(csv_path)
    else:
        all_games_events = load_all_games_events(base_path='../ift6758/data/dataStore' , years=years)
        df = events_to_dataframe2(all_games_events)
    if len(df) !=0: 
        df = _curenteventtime(df)
        df  = _lasteventtime(df)
        df = _computetimesincelastevent(df)
        df = _calculate_rebound(df)
        df = _compute_last_event_distance(df)
        df =  _compute_shot_angle(df)
        df = _compute_speed(df)
        df = _compute_angle_change(df)
        df = _compute_shot_distance(df)
    return df



