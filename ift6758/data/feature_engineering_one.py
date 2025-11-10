import sys, os 
import numpy as np
import pandas as pd
from ift6758.data.tidying import load_all_games_events, events_to_dataframe

sys.path.append(os.path.abspath('..'))
csv_path = "../ift6758/data/all_shots_goals.csv"

# def _calculate_distance_to_net(df) -> pd.DataFrame:
#   net_x, net_y = 89, 0
#   df['distance_to_net'] = np.sqrt( (net_x - df['coordinates_x'])**2 + (net_y - df['coordinates_y'])**2)
#   return df

def _calculate_distance_to_net(df) -> pd.DataFrame:
  net_x, net_y = 89, 0
  net_x2, net_y2 = -89, 0

  df["distance1"] = np.sqrt((df["coordinates_x"] - net_x)**2 + (df["coordinates_y"] - net_y)**2)
  df["distance2"] = np.sqrt((df["coordinates_x"] - net_x2)**2 + (df["coordinates_y"] - net_y2)**2)

  df.loc[df["zone_code"] == "D", "distance_to_net"] = df[["distance1", "distance2"]].min(axis=1)
  df.loc[df["zone_code"] != "D", "distance_to_net"] = df[["distance1", "distance2"]].min(axis=1)

  df.drop(columns=["distance1", "distance2"], inplace=True)

  return df
  
def _calculate_shot_angle(df) -> pd.DataFrame:
  net_x, net_y = 89, 0
  x_diff = (net_x - df['coordinates_x'])
  y_diff = (net_y - df['coordinates_y'])
  df['shot_angle'] = np.degrees(np.arctan2(y_diff, x_diff))
  return df

def _calculate_is_goal(df) -> pd.DataFrame:
  df['is_goal'] = (df['event_type'] == 'goal').astype(int)
  return df
  

def _calculate_is_empty_net(df) -> pd.DataFrame:
  mask = df['empty_net'].fillna(False)
  df['is_empty_net'] = mask.astype(int)
  return df

def _load_df(years): 
  if (os.path.exists(csv_path)):
    df = pd.read_csv(csv_path)
  else:
    all_games_events = load_all_games_events(base_path='../ift6758/data/dataStore' , years=years)
    df = events_to_dataframe(all_games_events)
  return df
  
def feature_engineering_one(years) -> pd.DataFrame:
    df = _load_df(years)
    df = _calculate_distance_to_net(df)
    df = _calculate_shot_angle(df)
    df = _calculate_is_goal(df)
    df = _calculate_is_empty_net(df)
    return df 
  