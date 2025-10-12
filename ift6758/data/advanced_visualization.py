import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
import tqdm
import os


rink = "../../figures/nhl_rink.png"
rink_half = "../../figures/nhl_rink_half.png"
x_bins = np.linspace(0, 100, 10)
y_bins = np.linspace(-42.5, 42.5,10)
HEATMAP_OPACITY = 0.8
### Helper Functions

def _smooth_gaussian_interpolation_(df, sigma_ft=6.0, upscale=10):
    """
    Apply Gaussian smoothing and optional upscaling to a 2D excess shot grid.
    Inputs:
        df (pd.DataFrame): 2D sr/h matrix 
        sigma_ft (float, optional): Gaussian smoothing in feet.
            Defaults to 6.0.
        upscale (int, optional): Factor to upsample the smoothed grid for
            visualization. Defaults to 10.
    Outputs:
        tuple:
            Z (ndarray): Smoothed 2D sr/h grid.
            x (ndarray): Vertical axis coordinates in feet.
            y (ndarray): Horizontal axis coordinates in feet.
            axes_extent (tuple): (X_MIN, X_MAX, Y_MIN, Y_MAX) rink extents.
    """
    X_MIN, X_MAX = -42.5, 42.5
    Y_MIN, Y_MAX = 0.0, 100.0

    Z = df.to_numpy(dtype=float)
    nx, ny = Z.shape
    dx = (Y_MAX - Y_MIN) / nx  # feet per row along X
    dy = (X_MAX - X_MIN) / ny  # feet per col along Y

    # Gaussian smoothing
    sigma_cells = max(sigma_ft / dx, sigma_ft / dy)
    Z = gaussian_filter(Z, sigma=sigma_cells, mode="nearest")

    #upsampling for smoother look
    if upscale and upscale > 1:
        Z = zoom(Z, zoom=upscale, order=3)

    x = np.linspace(Y_MIN, Y_MAX, Z.shape[0])  # vertical (X feet)
    y = np.linspace(X_MIN, X_MAX, Z.shape[1])  # horizontal (Y feet)
    
    return Z, x, y, (X_MIN, X_MAX, Y_MIN, Y_MAX)

def _rink_trace_(img_rgba, X_MIN, X_MAX, Y_MIN, Y_MAX, H, W):
    """
    Create a Plotly image trace for a hockey rink background.
    Inputs:
        img_rgba (np.ndarray): Rink image .
        X_MIN (float): Minimum horizontal rink coordinate in feet.
        X_MAX (float): Maximum horizontal rink coordinate in feet.
        Y_MIN (float): Minimum vertical rink coordinate in feet.
        Y_MAX (float): Maximum vertical rink coordinate in feet.
        H (int): Image height 
        W (int): Image width.
    Outputs:
        go.Image: Plotly image trace aligned with rink coordinates.
    """
    return go.Image(
        z=img_rgba,
        x0=X_MIN,
        dx=(X_MAX - X_MIN) / W,
        y0=Y_MAX,
        dy=-(Y_MAX - Y_MIN) / H,
        hoverinfo="skip",
        name="rink",
        opacity=1.0,
)
### End Helper Functions



def build_offensive_zone_figure_for_season(
    season: int,
    default_team: str = "Sharks",
    rink_image_path: str = rink_half):

    """
    Builds an interactive offensive zone heatmap figure for a given NHL season.
    Inputs:
        season (int): NHL season year (e.g., 20212022).
        default_team(str, optional): Team selected by default. Defaults to "Sharks".
        rink_image_path(str, optional): Path to half-rink background image.
    Returns:
        go.Figure: Interactive Plotly figure with team heatmaps and rink background.
    """
    
    df_all = pd.read_csv("all_shots_goals.csv")
    teams = sorted(df_all.loc[df_all["season"] == season, "team_name"].dropna().unique())

    if not teams:
        raise ValueError(f"No data found for season {season}.")
    if default_team not in teams:
        default_team = teams[0]

    Z_store, all_vals, axes_extent = {}, [], None
    
    for t in teams:
        df_team = team_data(season, t)
        Z, x, y, axes_extent = _smooth_gaussian_interpolation_(df_team)
        Z_store[(season, t)] = (Z, x, y)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            all_vals.append(finite.ravel())

    #symmetric color scale for different teams
    if all_vals:
        flat = np.concatenate(all_vals)
        lo, hi = np.percentile(flat, [5, 95])
        vmax = max(abs(lo), abs(hi))
        vmin = -vmax if vmax > 0 else None
    else:
        vmin = vmax = None

    X_MIN, X_MAX, Y_MIN, Y_MAX = axes_extent
    img_rgba = np.array(Image.open(rink_image_path).convert("RGBA"))
    H, W = img_rgba.shape[:2]

    #tracing rink as background
    season_traces = [_rink_trace_(img_rgba, X_MIN, X_MAX, Y_MIN, Y_MAX, H, W)]
    for t in teams:
        Z, x, y = Z_store[(season, t)]
        season_traces.append(go.Heatmap(
            z=Z,
            x=y,
            y=x,
            coloraxis="coloraxis",
            visible=False,
            opacity=HEATMAP_OPACITY,
            name=t,
            hovertemplate=f"Team: {t}<br>X: %{{y:.1f}} ft<br>Y: %{{x:.1f}} ft<br>Excess: %{{z:.3f}}<extra></extra>",
        ))
    frames = [go.Frame(name=str(season), data=season_traces)]

    initial_traces = []
    for i, tr in enumerate(frames[0].data):
        if i == 0:
            tr.visible = True
        else:
            tr.visible = (teams[i-1] == default_team)
        initial_traces.append(tr)

    fig = go.Figure(data=initial_traces, frames=frames)

    fig.update_layout(
        coloraxis=dict(
            colorscale="RdBu_r",
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title="Excess shots/hour"),
        ),
        title={
            "text": f"Offensive Zone Excess Shots — Season {season} — {default_team}",
            "x": 0.5},
        width=720,
        height=720,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        uirevision="keep",
        transition={"duration": 0},
    )

    fig.update_xaxes(range=[X_MIN, X_MAX],
                     title="Distance in centre of rink (ft)",
                     scaleanchor="y",
                     autorange=False,
                     fixedrange=True)
    fig.update_yaxes(range=[Y_MIN, Y_MAX],
                     title="Distance from goal line (ft)",
                     autorange=False,
                     fixedrange=True)

    # Team dropdown (background trace index 0 must always be ON)
    n_per_frame = len(teams) + 1
    team_buttons = []
    for ti, team in enumerate(teams):
        mask = [False] * n_per_frame
        mask[0] = True            # background
        mask[1 + ti] = True       # selected team
        team_buttons.append(dict(
            label=team,
            method="update",
            args=[
                {"visible": mask},
                {
                "title": {"text": f"Offensive Zone Excess Shots — Season {season} — {team}"},
                "frame": {"redraw": True},
                "transition": {"duration": 0},
            },
            ],
        ))
    fig.update_layout(
        updatemenus=[
            dict(type="dropdown",
                 direction="down",
                 x=0.01,
                 y=1.3,
                 xanchor="left",
                 yanchor="top",
                 buttons=team_buttons,
                 showactive=True,
                 pad=dict(r=6, t=4, b=4, l=6)),
        ]
    )

    return fig

def export_offensive_zone_figures(
    seasons=None,
    out_dir: str = "plot_figures",
    default_team: str = "Sharks",
    rink_image_path: str = rink_half,
    include_plotlyjs: str = "cdn"):

    """
    Export offensive zone heatmap figures for one or more NHL seasons.
    Inputs:
        seasons (list[int]): List of season years to process. If None,
            all available seasons in the data are used. Defaults to None.
        out_dir (str): Output directory for saving figures.
            Defaults to "plot_figures".
        default_team (str): Team selected by default in each figure.
            Defaults to "Sharks".
        rink_image_path (str): Path to half-rink background image.
            Defaults to `rink_half`.
        include_plotlyjs (str): How to include Plotly.js in HTML export
            ("cdn", "directory", or "inline"). Defaults to "cdn".
    Returns:
        None
    """
    
    if seasons is None:
        seasons = [20162017, 20172018, 20182019, 20192020, 20202021, 20212022, 20222023]

    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for s in (seasons):
        fig = build_offensive_zone_figure_for_season(
            season=s,
            default_team=default_team,
            rink_image_path=rink_image_path,
        )
        out_path = os.path.join(out_dir, f"offensive_zone_{s}.html")
        fig.write_html(out_path, include_plotlyjs=include_plotlyjs, full_html=True)
        saved.append(out_path)

    return saved


def league_stats(season):
    """
    Compute league-wide shot statistics for a given NHL season.
    Inputs:
        season (int): NHL season year (e.g., 2021).
    Ouputs:
        pd.DataFrame: Aggregated league-level shot statistics.
    """
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
    """
    Retrieve shot data for a specific team in a given NHL season.
    Inputs:
        season (int): NHL season year (e.g., 2021).
        team_name (str): Team name to filter.

    Outputs:
        DataFrame: Team-level shot data for the specified season.
    """
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