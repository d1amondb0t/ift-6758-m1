import os
from acquisition import read_json


def __build_game_id(year, season, game_id):
  return f"{int(year):04d}{int(season):02d}{int(game_id):04d}"

def load_json(filepath, year, season, game_id):
  gid = __build_game_id(year, season, game_id)
  folder = "regular" if season == 2 else "playoff"
  load_file = os.path.join(str(filepath), str(year), folder,f"{gid}.json")

  if os.path.exists(load_file):
    return read_json(load_file)
  return None