import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
x_bins = np.linspace(0, 100, 10)
y_bins = np.linspace(-42.5, 42.5,10)

def heatmap_data(team_name, season):
    df_aggregate_league = league_stats()
    df = pd.read_csv('all_shots_goals.csv')
    Team_data = df[(df['team_name'] == team_name) ]
    Team_data = Team_data[Team_data['season'] == season]
    Team_data['coordinates_x'] = Team_data['coordinates_x'].apply(abs)
    Team_data['coordinates_x_bins'] = pd.cut(Team_data['coordinates_x'],bins = x_bins  )
    Team_data['coordinates_y_bins'] = pd.cut(Team_data['coordinates_y'], bins = y_bins)
    df_aggregate_counts_team  = Team_data.groupby(['coordinates_x_bins', 'coordinates_y_bins']).size().reset_index(name="counts")
    df_aggregate_counts_team['sr/h'] = df_aggregate_counts_team['counts'] /  ( len ( Team_data['game_id'].unique()) ) #Every game is an hour so we divide by number of gammes 
    merged_df  = pd.merge(
    df_aggregate_counts_team,
    df_aggregate_league,
    on=["coordinates_x_bins", "coordinates_y_bins"])
    merged_df["excess_shot"] =((merged_df["sr/h_x"] - merged_df["sr/h_y"])  )   
    final_dataframe  = merged_df[['coordinates_x_bins', 'coordinates_y_bins', 'excess_shot']]
    final_datafram_pivot = final_dataframe.pivot( index ='coordinates_x_bins', columns='coordinates_y_bins', values= 'excess_shot')
    return final_datafram_pivot
    

def league_stats():
    if (os.path.exists("league_srperh.csv")):
        df_aggregate_counts_league = pd.read_csv("league_sr_per_h.csv")
        return df_aggregate_counts_league
    

    df = pd.read_csv('all_shots_goals.csv')
    df_league_stats =  df[['game_id',  'season', 'team_name', 'coordinates_x', 'coordinates_y']]
    df_league_stats = df_league_stats.dropna() #We have a couple na's in the raw json, I checked a few of them and the (x,y) was just not recorded 
    df_league_stats['coordinates_x'] = df_league_stats['coordinates_x'].apply(abs)
    games_total = len ( df_league_stats['game_id'].unique())
    df_league_stats['coordinates_x_bins'] = pd.cut(df_league_stats['coordinates_x'], bins =  x_bins) # its better to bin first 
    df_league_stats['coordinates_y_bins'] = pd.cut(df_league_stats['coordinates_y'], bins = y_bins )    
    df_aggregate_counts_league  = df_league_stats.groupby(['coordinates_x_bins', 'coordinates_y_bins']).size().reset_index(name="counts")
    df_aggregate_counts_league['sr/h'] = df_aggregate_counts_league['counts'] /  ( 2 * games_total )  #dividing by 2 because there are two teams in a game of hockey, and we need to acount for that 
    df_aggregate_counts_league.to_csv("league_sr_per_h.csv")
    return df_aggregate_counts_league




