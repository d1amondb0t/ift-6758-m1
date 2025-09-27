import os
import json
import requests

BASE_URL = "https://api-web.nhle.com/v1/gamecenter"


def write_json(data, file_path):
    """
    Save the data as a JSON file.

    :param data: The data to save.
    :type data: dict
    :param file_path: Path where the JSON file will be stored.
    :type file_path: str
    :return: None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)


def read_json(file_path):
    """
    Load a JSON file and return its contents.

    :param file_path: Path of the JSON file.
    :type file_path: str
    :return: Data loaded from the JSON file.
    :rtype: dict
    """
    with open(file_path, "r") as f:
        return json.load(f)


def get_game(game_id, out_dir):
    """
    Retrieve a single game's data.

    If the file already exists locally, it loads and returns the saved content.
    If not, it downloads from the NHL Web API and saves it.

    :param game_id: The NHL game ID
    :type game_id: str | int
    :param out_dir: Directory where the file  be stored.
    :type out_dir: str
    :return: The JSON payload if successful; "fail" if request error; None if already exist.
    :rtype: dict | str | None
    """
    file_path = os.path.join(out_dir, f"{game_id}.json")
    if os.path.exists(file_path):
        return read_json(file_path)
    url = f"{BASE_URL}/{game_id}/play-by-play"
    try:
        r = requests.get(url)
    except requests.RequestException:
        return "fail"

    if r.status_code == 200:
        payload = r.json()
        write_json(payload, file_path)
        return payload
    return None


def download_regular(start_year, output_path):
    """
    Download all  regular games for a season.

    There are 32 teams at most, so we assume there are 1312 games maximum (32 teams * 82 games / 2).

    :param start_year: Start year of the season
    :type start_year: int
    :param output_path: Directory where data will be stored.
    :type output_path: str
    :return: None
    """
    game_type = "02"
    start_id = int(f"{start_year}{game_type}0001")
    end_id = int(f"{start_year}{game_type}1312")
    game_id = start_id
    output_path = os.path.join(output_path, str(start_year), "regular")
    while game_id <= end_id:
        get_game(game_id, output_path)
        game_id += 1


def download_playoff(start_year, output_path):
    """
    Download all playoff games for a season.

    Playoff Game ID format uses the last 3 digits :
      - R: Round of the playoff
      - M: Matchup in that round (R1: up to 8, R2: up to 4, R3: up to 2, R4: 1)
      - G: Game number in series (out of 7)

    :param start_year: Start year of the season
    :type start_year: int
    :param output_path: Directory where data will be stored.
    :type output_path: str
    :return: None
    """
    game_type = "03"
    max_matchups = {1: 8, 2: 4, 3: 2, 4: 1}
    output_path = os.path.join(output_path, str(start_year), "playoff")
    for rnd in (1, 2, 3, 4):
        for matchup in range(1, max_matchups[rnd] + 1):
            for g in range(1, 8):
                game_id = f"{start_year}{game_type}0{rnd}{matchup}{g}"
                get_game(game_id, output_path)


def download_all(start_year, output_path):
    """
    Download both regular-season and playoff games for a  season.

    :param start_year: Start year of the season.
    :type start_year: int
    :param output_path: Directory where data will be stored.
    :type output_path: str
    :return: None
    """
    download_regular(start_year, output_path)
    download_playoff(start_year, output_path)
