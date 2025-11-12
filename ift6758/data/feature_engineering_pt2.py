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


def _calculate_rebound(df) -> pd.DataFrame:
    '''
    Calculates rebound, if the previous event was a shot on goal or missed shot then rebound is True otherwise False 
    Returns: df with rebound element included
    '''
    df['rebound'] = df['previous_event_name'].isin(['shot-on-goal', 'missed-shot'])
    return df 

def _lasteventtime(df)-> pd.DataFrame:
    ''' 
    Calculates the previous event time period, splits the time by minutes and seconds, creates a new dataframe with previous_event_timeseconds 
    ReturnsL df with previous_event_timeseconds column
    '''
    minutes = df['previous_event_timeperiod'].str.split(':', expand=True)[0].astype(int)
    seconds = df['previous_event_timeperiod'].str.split(':', expand=True)[1].astype(int)
    period_seconds =( minutes * 60) + seconds
    df['previous_event_timeseconds'] = period_seconds
    return df 

def _curenteventtime(df)-> pd.DataFrame:
    '''
    Calculates the current event time in seconds by splitting the period_time column 
    (formatted as MM:SS) into minutes and seconds, then combining them into total seconds. 
    Returns: df with current_event_timeseconds column included
    '''
    minutes = df['period_time'].str.split(':', expand=True)[0].astype(int)
    seconds = df['period_time'].str.split(':', expand=True)[1].astype(int)
    period_seconds = (minutes * 60) + seconds
    df['current_event_timeseconds'] = period_seconds
    return df 

def _computetimesincelastevent(df)-> pd.DataFrame:
    '''
    Calculates the time difference between the current event and the previous event 
    by subtracting previous_event_timeseconds from current_event_timeseconds. 
    Returns: df with time_since_last_event column included
    '''
    df['time_since_last_event'] =  np.abs(df['current_event_timeseconds'] - df['previous_event_timeseconds']) 
    return df

def _compute_last_event_distance(df)-> pd.DataFrame:
    '''
    Calculates the Euclidean distance between the current event and the previous event 
    using their (x, y) coordinates. 
    Returns: df with distance_from_last_event column included
    '''
    df['distance_from_last_event'] = np.sqrt( (df['previous_event_y'] - df['coordinates_y'] )**2 +( df['previous_event_x'] - df['coordinates_x'])**2)
    return df

def _compute_speed(df)-> pd.DataFrame:
    '''
    Calculates the speed of movement between the previous event and the current event 
    by dividing distance_from_last_event by time_since_last_event (where time is nonzero). 
    Returns: df with speed column included
    '''
    df.loc[df['time_since_last_event'] != 0, 'speed'] = df['distance_from_last_event'] / df['time_since_last_event']
    return df 

def _compute_angle_change(df)-> pd.DataFrame:
    '''
    Calculates the change in shot angle between the current and previous event 
    for rebound situations. If both events are on the same side of the ice, 
    the angle change is the absolute difference; otherwise, it sums the angles. 
    Returns: df with angle_change column included
    '''
    same_section = np.sign(df['coordinates_y']) == np.sign(df['previous_event_y'])
    df.loc[(df['rebound'] == True) & (same_section), "angle_change"] =  np.abs(df['angle_shot'] -  df['angle_shot_prev'])  
    df.loc[(df['rebound'] == True) & (~same_section), "angle_change"]=   (np.abs(df['angle_shot'] +  df['angle_shot_prev']))  
    df.loc[df['rebound'] == False, 'angle_change'] = 0.0
    return df 

def _compute_shot_angle(df)-> pd.DataFrame:
    '''
    Calculates the shot angle relative to the net for both current and previous events. 
    If the event is a rebound, also computes the previous shot angle for comparison. 
    Returns: df with angle_shot and angle_shot_prev columns included
    '''
    net_x, net_y = 89, 0
    x_diff  = np.abs( net_x - df['coordinates_x']) 
    y_diff  =np.abs ( df['coordinates_y']) 
    x_diff2 = np.abs( net_x - df['previous_event_x'])
    y_diff2 = np.abs ( df['previous_event_y']) 
    df['angle_shot_prev']  = 0.0
    df['angle_shot'] = np.degrees(np.arctan2(y_diff, x_diff))
    df.loc[df['rebound'] == True, "angle_shot_prev"]  =  np.degrees(np.arctan2(y_diff2, x_diff2))
    return df 

def _compute_shot_distance(df)-> pd.DataFrame:
    '''
    Calculates the distance from the shot to the net using the Euclidean distance formula. 
    Considers both offensive and defensive zones and selects the appropriate net for distance calculation. 
    Returns: df with distance_shot column included
    '''
    net_x, net_y = 89, 0
    net_x2,net_y2 = -89,0 
    df['distance1'] = np.sqrt((df['coordinates_x'] - net_x)**2 + (df['coordinates_y'] - net_y)**2)
    df['distance2'] = np.sqrt((df['coordinates_x'] - net_x2)**2 + (df['coordinates_y'] - net_y2)**2)
    df.loc[df['zone_code'] == 'O', 'distance_shot'] = df[['distance1', 'distance2']].min(axis=1)
    df.loc[df['zone_code'] != 'O', 'distance_shot'] = df[['distance1', 'distance2']].min(axis=1)
    df.drop(columns=['distance1', 'distance2'], inplace=True)
    return df

def _calculate_is_goal(df) -> pd.DataFrame:
    '''
    Determines whether the event is a goal based on the event_type column. 
    Sets is_goal to 1 if event_type is 'goal', otherwise 0. 
    Returns: df with is_goal column included
    '''
    df['is_goal'] = (df['event_type'] == 'goal').astype(int)
    return df


def feature_engineering_two(years):
    '''
    Performs feature engineering on game event data for specified years.
    This function loads event data either from a preprocessed CSV file 
    (`allshotgoals2.csv`) or directly from raw game data using 
    `load_all_games_events` and `events_to_dataframe2`. 
    It then applies a series of transformations to compute additional 
    analytical features related to hockey shot events.

    Steps performed include:
        - Converting current and previous event times into seconds
        - Calculating time since last event
        - Identifying rebound shots
        - Computing distance and speed between consecutive events
        - Calculating shot angles and changes in angle for rebounds
        - Measuring distance to the net
        - Determining whether the event resulted in a goal

    Params:
        years (list or int): The season year(s) for which to process event data.

    Returns:
        pd.DataFrame: A DataFrame with engineered features for each event.
    '''
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
        df = _calculate_is_goal(df)
    return df



