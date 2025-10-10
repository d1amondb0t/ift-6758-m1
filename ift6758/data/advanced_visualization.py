import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sci
import plotly.graph_objects as go
import ipywidgets as widgets
from PIL import Image
from IPython.display import display, clear_output
from scipy.ndimage import gaussian_filter

rink = "../../figures/nhl_rink.png"
x_bins = np.linspace(0, 100, 10)
y_bins = np.linspace(-42.5, 42.5,10)

def league_stats(season):
    df = pd.read_csv('all_shots_goals.csv')
    df_league_stats = df[['game_id', 'season', 'team_name', 'coordinates_x', 'coordinates_y']]
    df_league_stats = df_league_stats.dropna()
    df_league_stats = df_league_stats[df_league_stats['season'] == season]
    
    df_league_stats['coordinates_x'] = df_league_stats['coordinates_x'].apply(abs)
    
    df_league_stats['coordinates_x_bins'] = pd.cut(df_league_stats['coordinates_x'], bins=x_bins, ordered=True)
    df_league_stats['coordinates_y_bins'] = pd.cut(df_league_stats['coordinates_y'], bins=y_bins, ordered=True)    
    
    df_aggregate_counts_league  = df_league_stats.groupby(
        ['coordinates_x_bins', 'coordinates_y_bins'],
        observed=False).size().reset_index(name="counts")
    
    #dividing by 2 because there are two teams in a game of hockey, and we need to acount for that 
    games_total = len ( df_league_stats['game_id'].unique())
    df_aggregate_counts_league['sr/h'] = df_aggregate_counts_league['counts'] / (2 * games_total)  
    
    #df_aggregate_counts_league.to_csv("league_sr_per_h.csv")
    
    return df_aggregate_counts_league


def team_data(season, team_name):
    df_aggregate_league = league_stats(season)
    
    df = pd.read_csv('all_shots_goals.csv')
    team_data = df[(df['team_name'] == team_name) ]
    team_data = team_data[team_data['season'] == season]
    
    team_data['coordinates_x'] = team_data['coordinates_x'].apply(abs)
    team_data['coordinates_x_bins'] = pd.cut(team_data['coordinates_x'], bins=x_bins  )
    team_data['coordinates_y_bins'] = pd.cut(team_data['coordinates_y'], bins=y_bins)

    df_aggregate_counts_team  = team_data.groupby(
        ['coordinates_x_bins', 'coordinates_y_bins'],
        observed=False).size().reset_index(name="counts")
    
    #Every game is an hour so we divide by number of gammes 
    df_aggregate_counts_team['sr/h'] = df_aggregate_counts_team['counts'] / (len(team_data['game_id'].unique()))
    
    merged_df = pd.merge(df_aggregate_counts_team, df_aggregate_league,
                         on=["coordinates_x_bins", "coordinates_y_bins"])
    
    merged_df["excess_shot"] =((merged_df["sr/h_x"] - merged_df["sr/h_y"])  )   
    
    final_df  = merged_df[['coordinates_x_bins', 'coordinates_y_bins', 'excess_shot']]
    
    final_df = final_df.pivot(index ='coordinates_x_bins',
                                                 columns='coordinates_y_bins',
                                                 values= 'excess_shot')
    
    return final_df

def plot_data(df):
    X_MIN, X_MAX = -42.5, 42.5
    Y_MIN, Y_MAX = 0, 100
    rink_full = Image.open(rink).convert("RGBA")
    
    W, H = rink_full.size
    rink_half = rink_full.crop((W//2, 0, W, H))
    rink_half_rot = rink_half.transpose(Image.ROTATE_90)

    plt.imshow(rink_half_rot, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], alpha=1)

    Z_raw = df.values.astype(float)
    Z_smooth = gaussian_filter(Z_raw, sigma=0.8, mode="nearest")

    raw_max = np.nanmax(np.abs(Z_raw))
    smooth_max = max(np.nanmax(np.abs(Z_smooth)), 1e-12)
    Z_smooth *= (raw_max / smooth_max)

    lo, hi = np.percentile(Z_smooth, [2, 98])
    max_abs = max(abs(lo), abs(hi))
    vmin, vmax = -max_abs, +max_abs

    plt.imshow(
        Z_smooth, cmap="bwr", interpolation="gaussian", alpha=0.7,
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], vmin=vmin, vmax=vmax
    )

    plt.colorbar(label="Excess shots/hour")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(X_MIN, X_MAX); plt.ylim(Y_MIN, Y_MAX)
    plt.show()


def plot_team_excess_shot_map(season, team_name):
    df_team = team_data(season, team_name)
    plot_data(df_team)
    

def interactive_team_shot_map():
    
    df = pd.read_csv('all_shots_goals.csv')
    teams = sorted(df['team_name'].dropna().unique())
    seasons = sorted(df['season'].dropna().unique())

    team_dropdown = widgets.Dropdown(
        options=teams, value='Sharks', description='Team:'
    )
    season_dropdown = widgets.Dropdown(
        options=seasons, value=20172018, description='Season:'
    )

    output = widgets.Output()

    def update_plot(change=None):
        with output:
            clear_output(wait=True)
            df_team = team_data(season_dropdown.value, team_dropdown.value)
            plot_data(df_team)

    team_dropdown.observe(update_plot, names='value')
    season_dropdown.observe(update_plot, names='value')
    update_plot()

    display(widgets.VBox([widgets.HBox([team_dropdown, season_dropdown]), output]))