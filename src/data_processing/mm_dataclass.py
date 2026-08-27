from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class MarchMadnessData:
    """
    Load and provide access to the March Madness source datasets.

    Each CSV file is loaded into:
    1. The `files` dictionary
    2. A corresponding class attribute

    Example
    -------
    data = MarchMadnessData()
    kenpom = data.kenpom_barttorvik
    """

    FILE_MAP = {
        "538 Ratings.csv": "ratings_538",
        "AP Poll Data.csv": "ap_poll_data",
        "Barttorvik Away-Neutral.csv": "barttorvik_away_neutral",
        "Barttorvik Away.csv": "barttorvik_away",
        "Barttorvik Home.csv": "barttorvik_home",
        "Barttorvik Neutral.csv": "barttorvik_neutral",
        "Coach Results.csv": "coach_results",
        "Conference Results.csv": "conference_results",
        "Conference Stats Away Neutral.csv": "conference_stats_away_neutral",
        "Conference Stats Away.csv": "conference_stats_away",
        "Conference Stats Home.csv": "conference_stats_home",
        "Conference Stats Neutral.csv": "conference_stats_neutral",
        "Conference Stats.csv": "conference_stats",
        "EvanMiya.csv": "evanmiya",
        "Heat Check Ratings.csv": "heat_check_ratings",
        "Heat Check Tournament Index.csv": "heat_check_tournament_index",
        "KenPom Barttorvik.csv": "kenpom_barttorvik",
        "KenPom Preseason.csv": "kenpom_preseason",
        "Public Picks.csv": "public_picks",
        "Resumes.csv": "resumes",
        "RPPF Conference Ratings.csv": "rppf_conference_ratings",
        "RPPF Preseason Ratings.csv": "rppf_preseason_ratings",
        "RPPF Ratings.csv": "rppf_ratings",
        "Seed Results.csv": "seed_results",
        "Shooting Splits.csv": "shooting_splits",
        "Team Results.csv": "team_results",
        "TeamRankings Away.csv": "teamrankings_away",
        "TeamRankings Home.csv": "teamrankings_home",
        "TeamRankings Neutral.csv": "teamrankings_neutral",
        "TeamRankings.csv": "teamrankings",
        "Teamsheet Ranks.csv": "teamsheet_ranks",
        "Tournament Locations.csv": "tournament_locations",
        "Tournament Matchups.csv": "tournament_matchups",
        "Tournament Simulation.csv": "tournament_simulation",
        "Upset Count.csv": "upset_count",
        "Upset Seed Info.csv": "upset_seed_info",
        "Z Rating Cumulative.csv": "z_rating_cumulative",
        "Z Rating Teams.csv": "z_rating_teams",
    }

    def __init__(self, data_folder=DEFAULT_DATA_DIR):
        self.data_folder = Path(data_folder)
        self.files = {}

        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_folder}\n"
                "Run update_mm_data.py first to download the source data."
            )

        self.load_data()

    def load_data(self):
        """
        Load all available CSV files defined in FILE_MAP.
        """
        for filename, attr_name in self.FILE_MAP.items():
            file_path = self.data_folder / filename

            if file_path.exists():
                df = pd.read_csv(file_path)
                self.files[attr_name] = df
                setattr(self, attr_name, df)

    def list_datasets(self):
        """
        Print the names of all successfully loaded datasets.
        """
        print("Loaded datasets:")
        for name in sorted(self.files):
            print(f"- {name}")

    def show_shapes(self):
        """
        Print the shape of each successfully loaded dataset.
        """
        print("Loaded datasets and shapes:")
        for name in sorted(self.files):
            print(f"{name}: {self.files[name].shape}")

    def get_dataset(self, name):
        """
        Return a loaded dataset by its mapped name.

        Parameters
        ----------
        name : str
            Dataset name from FILE_MAP, such as 'kenpom_barttorvik'.

        Returns
        -------
        pandas.DataFrame
        """
        if name not in self.files:
            raise KeyError(
                f"Dataset '{name}' is not loaded. "
                f"Available datasets: {sorted(self.files)}"
            )

        return self.files[name]