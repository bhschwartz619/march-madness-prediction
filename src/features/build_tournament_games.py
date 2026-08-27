from pathlib import Path

import pandas as pd

from src.data_processing.mm_dataclass import MarchMadnessData


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tournament_matchups_game_level.csv"
)

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "2026_tournament_matchups_game_level.csv"
)


def build_tournament_games(save_path=None):
    """
    Build the historical game-level tournament dataset.

    This preserves the logic of the original build_games_df.py:
    - excludes 2026
    - pairs every two rows into one game
    - creates outcome variables
    - creates seed-based descriptive variables
    """
    data = MarchMadnessData()
    df = data.files["tournament_matchups"].copy()

    # Original project excluded 2026 from historical games
    df = df[df["YEAR"] != 2026].copy()

    if len(df) % 2 != 0:
        raise ValueError(
            "Tournament Matchups file has an odd number of rows, "
            "so it cannot be paired cleanly."
        )

    df = df.reset_index(drop=True)

    team1 = df.iloc[0::2].reset_index(drop=True)
    team2 = df.iloc[1::2].reset_index(drop=True)

    games = pd.DataFrame(
        {
            "year": team1["YEAR"],
            "current_round": team1["CURRENT ROUND"],

            "team1_by_year_no": team1["BY YEAR NO"],
            "team1_round_fin": team1["ROUND"],
            "team1_no": team1["TEAM NO"],
            "team1": team1["TEAM"],
            "seed1": team1["SEED"],
            "score1": team1["SCORE"],

            "team2_by_year_no": team2["BY YEAR NO"],
            "team2_round_fin": team2["ROUND"],
            "team2_no": team2["TEAM NO"],
            "team2": team2["TEAM"],
            "seed2": team2["SEED"],
            "score2": team2["SCORE"],
        }
    )

    # Game outcomes
    games["team1_win"] = (
        games["score1"] > games["score2"]
    ).astype(int)

    games["winner"] = games["team1"].where(
        games["team1_win"] == 1,
        games["team2"],
    )

    games["loser"] = games["team2"].where(
        games["team1_win"] == 1,
        games["team1"],
    )

    games["score_diff"] = (
        games["score1"] - games["score2"]
    )

    # Seed features
    games["better_seed"] = games[
        ["seed1", "seed2"]
    ].min(axis=1)

    games["worse_seed"] = games[
        ["seed1", "seed2"]
    ].max(axis=1)

    games["seed_gap"] = (
        games["worse_seed"]
        - games["better_seed"]
    )

    games["better_seed_team"] = games["team1"].where(
        games["seed1"] < games["seed2"],
        games["team2"],
    )

    games["worse_seed_team"] = games["team1"].where(
        games["seed1"] > games["seed2"],
        games["team2"],
    )

    games["better_seed_score"] = games["score1"].where(
        games["seed1"] < games["seed2"],
        games["score2"],
    )

    games["worse_seed_score"] = games["score1"].where(
        games["seed1"] > games["seed2"],
        games["score2"],
    )

    equal_seed_mask = (
        games["seed1"] == games["seed2"]
    )

    games.loc[
        equal_seed_mask,
        "better_seed_team",
    ] = None

    games.loc[
        equal_seed_mask,
        "worse_seed_team",
    ] = None

    games.loc[
        equal_seed_mask,
        "better_seed_score",
    ] = None

    games.loc[
        equal_seed_mask,
        "worse_seed_score",
    ] = None

    games["better_seed_win"] = (
        games["winner"]
        == games["better_seed_team"]
    ).astype("Int64")

    games.loc[
        equal_seed_mask,
        "better_seed_win",
    ] = pd.NA

    games["upset"] = (
        games["winner"]
        == games["worse_seed_team"]
    ).astype("Int64")

    games.loc[
        equal_seed_mask,
        "upset",
    ] = pd.NA

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        games.to_csv(
            save_path,
            index=False,
        )

    return games


def build_tournament_games_for_prediction(
    save_path=None,
):
    """
    Build the 2026 Round of 64 matchup dataset.

    This preserves the logic of the original games_for_modeling.py.
    """
    data = MarchMadnessData()
    df = data.files["tournament_matchups"].copy()

    # Original project filtered specifically to:
    # Round of 64 AND 2026
    df = df[
        df["CURRENT ROUND"] == 64
    ].copy()

    df = df[
        df["YEAR"] == 2026
    ].copy()

    if len(df) % 2 != 0:
        raise ValueError(
            "Tournament Matchups file has an odd number of rows, "
            "so it cannot be paired cleanly."
        )

    df = df.reset_index(drop=True)

    team1 = df.iloc[0::2].reset_index(drop=True)
    team2 = df.iloc[1::2].reset_index(drop=True)

    games = pd.DataFrame(
        {
            "year": team1["YEAR"],
            "current_round": team1["CURRENT ROUND"],

            "team1_by_year_no": team1["BY YEAR NO"],
            "team1_round_fin": team1["ROUND"],
            "team1_no": team1["TEAM NO"],
            "team1": team1["TEAM"],
            "seed1": team1["SEED"],

            "team2_by_year_no": team2["BY YEAR NO"],
            "team2_round_fin": team2["ROUND"],
            "team2_no": team2["TEAM NO"],
            "team2": team2["TEAM"],
            "seed2": team2["SEED"],
        }
    )

    games["better_seed"] = games[
        ["seed1", "seed2"]
    ].min(axis=1)

    games["worse_seed"] = games[
        ["seed1", "seed2"]
    ].max(axis=1)

    games["seed_gap"] = (
        games["worse_seed"]
        - games["better_seed"]
    )

    games["better_seed_team"] = games["team1"].where(
        games["seed1"] < games["seed2"],
        games["team2"],
    )

    games["worse_seed_team"] = games["team1"].where(
        games["seed1"] > games["seed2"],
        games["team2"],
    )

    equal_seed_mask = (
        games["seed1"] == games["seed2"]
    )

    games.loc[
        equal_seed_mask,
        "better_seed_team",
    ] = None

    games.loc[
        equal_seed_mask,
        "worse_seed_team",
    ] = None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        games.to_csv(
            save_path,
            index=False,
        )

    return games


def save_historical_tournament_games(
    output_path=HISTORICAL_OUTPUT_PATH,
):
    games = build_tournament_games(
        save_path=output_path
    )

    print(
        f"Tournament games shape: "
        f"{games.shape}"
    )

    print(
        f"Tournament games saved to: "
        f"{output_path}"
    )

    return games


def save_2026_prediction_games(
    output_path=PREDICTION_OUTPUT_PATH,
):
    games = build_tournament_games_for_prediction(
        save_path=output_path
    )

    print(
        f"2026 Round of 64 shape: "
        f"{games.shape}"
    )

    print(
        f"2026 tournament games saved to: "
        f"{output_path}"
    )

    return games


if __name__ == "__main__":
    save_historical_tournament_games()