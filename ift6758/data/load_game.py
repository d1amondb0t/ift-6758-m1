import os
from acquisition import read_json
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

def __build_game_id(year, season, game_id):
  return f"{int(year):04d}{int(season):02d}{int(game_id):04d}"

def load_json(filepath, year, season, game_id):
  gid = __build_game_id(year, season, game_id)
  folder = "regular" if season == 2 else "playoff"
  load_file = os.path.join(str(filepath), str(year), folder,f"{gid}.json")

  if os.path.exists(load_file):
    return read_json(load_file)
  return None

def load_plays(data):
  if not data:
    return []
  return data.get("plays", [])

def event_id_min_max(plays):
  ids = [p.get('eventId') for p in plays if isinstance(p.get("eventId"), int)]
  return (1, len(ids)-1)

def plot_ring(ring_path, play):

  x = float(play["details"]["xCoord"])
  y = float(play["details"]["yCoord"])
  fig, ax = plt.subplots(figsize=(10, 6))
  img = plt.imread(ring_path)
  ax.imshow(img, extent=[-100, 100, -42.5, 42.5], origin="upper") # Ring limits
  ax.set_xlim(-100, 100)
  ax.set_ylim(-42.5, 42.5)

  ax.scatter([x], [y], s=120)
  ax.annotate(f"#{play.get('eventId')}", (x, y))

  title = f"P{play['periodDescriptor']['number']} {play.get('timeInPeriod','')} - {play.get('typeDescKey','')} (event {play['eventId']})"
  ax.set_title(title)
  ax.set_xlabel("X (feet)"); ax.set_ylabel("Y (feet)")
  plt.show()

def game_summary(data,play):
  start_time = data["startTimeUTC"]
  gid = str(data["id"])[-4:]
  home = data["homeTeam"] if "homeTeam" in data else {}
  away = data["awayTeam"] if "awayTeam" in data else {}
  pd_num = play["periodDescriptor"]["number"] if "periodDescriptor" in play and "number" in play[
    "periodDescriptor"] else "?"
  time_in_period = play["timeInPeriod"] if "timeInPeriod" in play else ""
  type_key = play["typeDescKey"] if "typeDescKey" in play else ""
  event_id = play["eventId"] if "eventId" in play else ""
  home_abbr = home["abbrev"] if "abbrev" in home else None
  away_abbr = away["abbrev"] if "abbrev" in away else None

  home_goals = home["score"] if "score" in home else home.get("goals", None)
  away_goals = away["score"] if "score" in away else away.get("goals", None)

  home_sog = home["sog"] if "sog" in home else home.get("shotsOnGoal", None)
  away_sog = away["sog"] if "sog" in away else away.get("shotsOnGoal", None)

  home_so_goals = home["shootoutGoals"] if "shootoutGoals" in home else home.get("soGoals", None)
  away_so_goals = away["shootoutGoals"] if "shootoutGoals" in away else away.get("soGoals", None)

  home_so_att = home["shootoutAttempts"] if "shootoutAttempts" in home else home.get("soAttempts", None)
  away_so_att = away["shootoutAttempts"] if "shootoutAttempts" in away else away.get("soAttempts", None)


  print(f"{start_time} | Game ID: {gid}")
  print(f"{home_abbr} (home) vs {away_abbr} (away)")
  print(f"P{pd_num} {time_in_period} - {type_key} (event {event_id})")
  print(f"{'':12}{'Home':<12}{'Away':<12}")
  print(f"{'Teams:':12}{home_abbr!s:<12}{away_abbr!s:<12}")
  print(f"{'Goals:':12}{home_goals!s:<12}{away_goals!s:<12}")
  print(f"{'SoG:':12}{home_sog!s:<12}{away_sog!s:<12}")
  print(f"{'SO Goals:':12}{home_so_goals!s:<12}{away_so_goals!s:<12}")
  print(f"{'SO Attempts:':12}{home_so_att!s:<12}{away_so_att!s:<12}")


def on_game_change(root, ring_path, year_widget, season_widget, game_widget, event_widget, out=None):
  def _handler(change):
    data = load_json(root, year_widget.value, season_widget.value, game_widget.value)
    plays = load_plays(data)

    if plays:
      min, max = event_id_min_max(plays)
      event_widget.min = min
      event_widget.max = max

    if out is not None:
      with out:
        clear_output(wait=True)
        if plays:
          idx = event_widget.value
          play = plays[idx]
          d = play.get("details", {}) or {}
          if isinstance(d.get("xCoord"), (int, float)) and isinstance(d.get("yCoord"), (int, float)):
            game_summary(data, play)
            plot_ring(ring_path, play)
          else:
            print(f"Play index {idx} with eventId {play.get('eventId')} has no x/y coordinates to plot")

  return _handler