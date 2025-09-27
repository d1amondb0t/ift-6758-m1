import os
from acquisition import read_json
import matplotlib.pyplot as plt
from IPython.display import clear_output

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
  ax.annotate(f"#{play.get("eventId")}", (x, y))

  title = f"P{play['periodDescriptor']['number']} {play.get('timeInPeriod','')} - {play.get('typeDescKey','')} (event {play['eventId']})"
  ax.set_title(title)
  ax.set_xlabel("X (feet)"); ax.set_ylabel("Y (feet)")
  plt.show()

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
            plot_ring(ring_path, play)
          else:
            print(f"Play index {idx} with eventId {play.get('eventId')} has no x/y coordinates to plot")
    
  return _handler