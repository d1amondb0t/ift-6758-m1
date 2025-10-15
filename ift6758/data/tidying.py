import json
import os
import pandas as pd
from tqdm import tqdm
from ift6758.data.acquisition import read_json


def load_all_games_events(base_path="./dataStore", years=range(2016, 2024)):
    """
    Load all game events from the given data directory and years.

    Parameters
    ----------
    base_path : str, optional
        Path to the root data folder (default is "./dataStore").
    years : range or list, optional
        Range or list of seasons to load (default is 2016–2023).
    Returns
    -------
        data events for all requested games.
    """

    all_games = {}
    
    for year in tqdm(years):
        for season_type in ["regular", "playoff"]:
            season_path = os.path.join(base_path, str(year), season_type)
            if not os.path.exists(season_path):
                continue
            for file in os.listdir(season_path):
                if file.endswith(".json"):
                    game_id = file.replace(".json", "")
                    file_path = os.path.join(season_path, file)
                    data = read_json(file_path)
                    plays = data.get("plays", [])
                    roster_spots = data.get("rosterSpots", [])
                    game_time = data.get("startTimeUTC")
                    home_team = data.get("homeTeam")
                    away_team = data.get("awayTeam")
                    season = data.get("season")
                    all_games[game_id] = {"season": season, "homeTeam": home_team, "awayTeam": away_team, "gameTime": game_time, "plays" : plays, "rosterSpots": roster_spots}
    return all_games


def __processeventtype_(situationcode, isHome):
    """
    This is a helper function: 

    Determine the strength type of a goal/shot event and whether the net was empty.
    NHL API encodes player counts for each team in the `situationcode`. By 
    comparing the number of skaters for the home and away teams and checking 
    which side took the shot, we can classify the play as:
    
    - **Even Strength**: both teams have the same number of players.
    - **Power Play**: the shooting team has more players on the ice.
    - **Short-Handed Goal**: the shooting team has fewer players on the ice.

    Also this function checks whether a goalie was present in net 
    for the opposing team at the time of the shot (i.e., if it was an empty net situation).

    Params
    ----------
    situationcode : str or list-like
        Encoded player situation, typically a 4-character string where:
        - [0]: number of home skaters (including goalie)
        - [1]: number of home penalties
        - [2]: number of away skaters (including goalie)
        - [3]: number of away penalties
    isHome : bool
        True if the shot was taken by the home team, False if taken by the away team.

    Returns
    -------
    strength : str
        Classification of the play: "Even Strength", "Power Play", or "Short-Handed Goal".
    empty_net : bool
        True if the opposing goalie was not on the ice (empty net), False otherwise.
    Notes
    -----
    - The logic for interpreting `situationcode` is based on the NHL API discussion:
      https://gitlab.com/dword4/nhlapi/-/issues/110#note_1582828385
    """
    away_team_sum = int (situationcode[0]) + int (situationcode[1])
    home_team_sum = int(situationcode[2]) + int (situationcode[3])
    goalie_away = int(situationcode[0])
    goalie_home = int(situationcode[3])
    #Determine the shot type
    if home_team_sum == away_team_sum:
        strength  = "Even Strength"
    elif (home_team_sum > away_team_sum and isHome) or (away_team_sum> home_team_sum and not isHome):
        strength = "Power Play"
    elif (home_team_sum < away_team_sum and isHome) or (away_team_sum< home_team_sum and not isHome):
        strength = "Short-Handed Goal"
    
    #Check if the goalie was present or not
    if (isHome and goalie_away == 0) or (not isHome and goalie_home == 0):
        empty_net = True 
    else:
        empty_net = False
    return strength, empty_net
    

def events_to_dataframe(all_games_events):
    """
    Convert all game events into a pandas DataFrame.

    Parameters
    ----------
    all_games_events : dict or list
        Collection of events from multiple games.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the structured event data.
    """

    records = []
    for game_id, game_data in tqdm( all_games_events.items() ):
        home_team = game_data.get("homeTeam", [])
        away_team= game_data.get("awayTeam", [] )
        season = game_data.get("season", []) 
        id_h, name_h  = home_team.get("id", []), home_team.get("commonName").get("default") if home_team else []
        id_a, name_a  = away_team.get("id"), away_team.get("commonName").get("default") if away_team else [] 


        plays = game_data.get("plays", [])
        players = game_data.get("rosterSpots", [])
        game_time = game_data.get("gameTime")
        for ev in plays:
            ev_type = ev.get("typeDescKey")
            if ev_type not in ["shot-on-goal", "goal"]:
                continue
            details = ev.get("details", {})
            strength, empty_net   = __processeventtype_(ev.get("situationCode"), True if str(id_h) ==  str( details.get("eventOwnerTeamId")) else False )
            shooter_player_id = details.get("shootingPlayerId") 
            scoring_player_id = details.get("scoringPlayerId")
            goalie_in_net_id = details.get("goalieInNetId")
    
            record = {
                "game_id": game_id,
                "season" : season,
                "game_time": pd.to_datetime(game_time),
                "period": ev.get("periodDescriptor", {}).get("number"),
                "period_time": ev.get("timeInPeriod"),
                "event_type": "goal" if ev_type == "goal" else "shot",
                "team_id": details.get("eventOwnerTeamId"),
                "team_name": name_h if id_h == details.get("eventOwnerTeamId") else name_a, 
                "coordinates_x": details.get("xCoord"),
                "coordinates_y": details.get("yCoord"),
                "shooter": shooter_player_id or scoring_player_id,
                "goalie": goalie_in_net_id,
                "shot_type": details.get("shotType"), 
                "empty_net": empty_net,
                "strength": strength,
                "situation_code": ev.get("situationCode"),
            }

            # joueurs impliqués
            for player in players:
                player_id = player.get("playerId")
                if (player_id == shooter_player_id) or (player_id == scoring_player_id):
                    record["shooter"] = player.get("firstName", "").get("default") + " " + player.get("lastName", "").get("default")
                elif player_id == goalie_in_net_id:
                    record["goalie"] = player.get("firstName", "").get("default") + " " + player.get("lastName", "").get("default") 

            records.append(record)

    return pd.DataFrame(records)

if __name__ == "__main__":
    with open("game.json", "w") as f:
        json.dump("./dataStore/2016/playoff/2016030111.json", f, indent=4)
    # Charger tous les événements de tous les matchs
    print("Loading all games events...")
    all_games_events = load_all_games_events()
    # print(f"Total games loaded: {len(all_games_events)}")

    # Convertir en DataFrame
    print("Converting events to DataFrame...")
    df = events_to_dataframe(all_games_events)
    print(df)
    print(f"Total events extracted: {len(df)}")

    # Optionnel : sauvegarder en CSV pour utilisation future
    output_csv = "all_shots_goals.csv"
    df.to_csv(output_csv, index=False)
    print(f"DataFrame saved to {output_csv}")