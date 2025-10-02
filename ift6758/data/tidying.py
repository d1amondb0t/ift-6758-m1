import json
import os
import pandas as pd
from acquisition import read_json

def load_all_games_events(base_path="./dataStore", years=range(2016, 2024)):
    all_games = {}
    
    for year in years:
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
                    all_games[game_id] = {"gameTime": game_time, "plays" : plays, "rosterSpots": roster_spots}
    return all_games

def events_to_dataframe(all_games_events):
    records = []

    for game_id, game_data in all_games_events.items():
        plays = game_data.get("plays", [])
        players = game_data.get("rosterSpots", [])
        game_time = game_data.get("gameTime")
        for ev in plays:
            ev_type = ev.get("typeDescKey")
            if ev_type not in ["shot-on-goal", "goal"]:
                continue

            details = ev.get("details", {})
            shooter_player_id = details.get("shootingPlayerId") 
            scoring_player_id = details.get("scoringPlayerId")
            goalie_in_net_id = details.get("goalieInNetId")

            record = {
                "game_id": game_id,
                "game_time": pd.to_datetime(game_time),
                "period": ev.get("periodDescriptor", {}).get("number"),
                "period_time": ev.get("timeInPeriod"),
                "event_type": "goal" if ev_type == "goal" else "shot",
                "team_id": details.get("eventOwnerTeamId"),
                "coordinates_x": details.get("xCoord"),
                "coordinates_y": details.get("yCoord"),
                "shooter": shooter_player_id or scoring_player_id,
                "goalie": goalie_in_net_id,
                "shot_type": details.get("shotType")
                # See what to put here : 
                # "empty_net": details.get("emptyNet", False),
                #"strength": details.get("strength")
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