import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_shots_simple(df, season_year=2016):
    """Show shots vs goals for each shot type"""
    
    # Filter to just one season
    df_season = df[df['game_time'].dt.year == season_year]
    
    # Count shots and goals for each shot type
    shot_counts = df_season.groupby('shot_type').size()
    goal_counts = df_season[df_season['event_type'] == 'goal'].groupby('shot_type').size()
    
    # Calculate goal percentage
    goal_pct = (goal_counts / shot_counts * 100).fillna(0)
    
    # Sort by goal percentage (highest to lowest)
    goal_pct_sorted = goal_pct.sort_values(ascending=False)
    shot_counts_sorted = shot_counts[goal_pct_sorted.index]
    goal_counts_sorted = goal_counts.reindex(goal_pct_sorted.index, fill_value=0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot shots as blue bars
    x = range(len(shot_counts_sorted))
    ax.bar(x, shot_counts_sorted, color='skyblue', label='Shots')
    
    # Plot goals as red bars (on top)
    ax.bar(x, goal_counts_sorted, color='crimson', alpha=0.7, label='Goals')
    
    # Add percentage labels above bars
    for i, (shot_type, pct) in enumerate(goal_pct_sorted.items()):
        ax.text(i, shot_counts_sorted.iloc[i] + 50, f'{pct:.1f}%', 
                ha='center', fontweight='bold')
    
    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(shot_counts_sorted.index, rotation=45, ha='right')
    ax.set_xlabel('Shot Type')
    ax.set_ylabel('Count')
    ax.set_title(f'Shots vs Goals by Type ({season_year} Season) - Sorted by Goal %')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_distance_analysis(df):
    """Show how distance affects goal probability across seasons"""
    
    # Calculate distance from goal for each shot
    df = df.copy()
    goal_x = np.where(df['coordinates_x'] > 0, 89, -89)
    df['distance'] = np.sqrt((df['coordinates_x'] - goal_x)**2 + 
                             df['coordinates_y']**2)
    
    # Mark which shots were goals
    df['is_goal'] = (df['event_type'] == 'goal').astype(int)
    
    # Group shots by distance (every 5 feet)
    df['distance_group'] = (df['distance'] // 5) * 5
    
    # Define seasons to analyze
    seasons = [20182019, 20192020, 20202021]
    season_names = ['2018-19', '2019-20', '2020-21']
    colors = ['#ef4444', '#3b82f6', '#10b981']  # red, blue, green
    
    # Create 4 subplots (3 individual + 1 comparison)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Shot Distance vs Goal Probability (2018-19 to 2020-21)', 
                 fontsize=16, fontweight='bold')
    
    max_goal_pct = 0  # Track highest percentage for consistent y-axis
    
    # Plot each season individually
    for i, (season, season_name, color) in enumerate(zip(seasons, season_names, colors)):
        # Get the right subplot (top-left, top-right, bottom-left)
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Filter data for this season
        season_data = df[df['season'] == season]
        
        # Calculate goal % by distance for this season
        stats = season_data.groupby('distance_group').agg({
            'is_goal': ['mean', 'count']
        })
        stats.columns = ['goal_rate', 'num_shots']
        stats['goal_pct'] = stats['goal_rate'] * 100
        stats = stats[stats['num_shots'] >= 10]  # Only distances with enough shots
        
        # Plot this season
        ax.plot(stats.index, stats['goal_pct'], marker='o', 
                color=color, linewidth=2, label=season_name)
        
        # Add labels and formatting
        ax.set_xlabel('Shot Distance (feet)')
        ax.set_ylabel('Goal Probability (%)')
        ax.set_title(f'{season_name} Season')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 80)
        
        # Add shot/goal counts to the graph
        total_shots = len(season_data)
        total_goals = season_data['is_goal'].sum()
        ax.text(0.95, 0.9, f'Shots: {total_shots:,}\nGoals: {int(total_goals):,}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
        
        # Track max percentage
        max_goal_pct = max(max_goal_pct, stats['goal_pct'].max())
    
    # Set same y-axis limit for all individual plots
    for i in range(3):
        row = i // 2
        col = i % 2
        axes[row, col].set_ylim(0, max_goal_pct + 5)
    
    # Fourth plot: Comparison of all seasons
    ax_comp = axes[1, 1]
    
    for season, season_name, color in zip(seasons, season_names, colors):
        season_data = df[df['season'] == season]
        
        stats = season_data.groupby('distance_group').agg({
            'is_goal': ['mean', 'count']
        })
        stats.columns = ['goal_rate', 'num_shots']
        stats['goal_pct'] = stats['goal_rate'] * 100
        stats = stats[stats['num_shots'] >= 10]
        
        ax_comp.plot(stats.index, stats['goal_pct'], marker='o',
                    color=color, linewidth=2, label=season_name)
    
    ax_comp.set_xlabel('Shot Distance (feet)')
    ax_comp.set_ylabel('Goal Probability (%)')
    ax_comp.set_title('Season Comparison')
    ax_comp.grid(alpha=0.3, linestyle='--')
    ax_comp.legend(loc='upper right')
    ax_comp.set_xlim(0, 80)
    ax_comp.set_ylim(0, max_goal_pct + 5)
    
    plt.tight_layout()
    plt.show()

def plot_distance_by_shot_type(df, season_year=20182019):
    """Compare different shot types at different distances"""
    
    # Filter to one season
    df_season = df[df['season'] == season_year].copy()
    
    # Calculate distance
    goal_x = np.where(df_season['coordinates_x'] > 0, 89, -89)
    df_season['distance'] = np.sqrt(
        (df_season['coordinates_x'] - goal_x)**2 + 
        df_season['coordinates_y']**2
    )
    
    # Mark goals
    df_season['is_goal'] = (df_season['event_type'] == 'goal').astype(int)
    
    # Group by distance (every 5 feet)
    df_season['distance_group'] = (df_season['distance'] // 5) * 5
    
    # Calculate goal % for each shot type and distance
    stats = df_season.groupby(['shot_type', 'distance_group'])['is_goal'].agg([
        ('goal_pct', lambda x: x.mean() * 100),
        ('count', 'count')
    ]).reset_index()
    
    # Plot each shot type as a separate line
    plt.figure(figsize=(10, 6))
    
    for shot_type in stats['shot_type'].unique():
        data = stats[stats['shot_type'] == shot_type]
        plt.plot(data['distance_group'], data['goal_pct'], 
                marker='o', label=shot_type, linewidth=2)
    
    plt.xlabel('Shot Distance (feet)')
    plt.ylabel('Goal Percentage (%)')
    plt.title(f'Goal % by Distance and Shot Type ({season_year})')
    plt.legend(title='Shot Type')
    plt.grid(alpha=0.3)
    plt.xlim(0, 80)
    
    plt.tight_layout()
    plt.show()