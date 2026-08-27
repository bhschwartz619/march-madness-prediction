from pathlib import Path

import pandas as pd

from src.data_processing.mm_dataclass import MarchMadnessData


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tournament_matchups_game_level.csv"
)


def build_tournament_games(
    year=None,
    round_number=None,
    exclude_year=None,
):
    """
    Convert the tournament matchup dataset from two rows per game
    into one row per game.

    Parameters
    ----------
    year : int, optional
        Keep only games from a specific tournament year.

    round_number : int, optional
        Keep only games from a specific tournament round
        (64, 32, 16, 8, 4, or 2).

    exclude_year : int, optional
        Exclude games from a specific tournament year.

    Returns
    -------
    pandas.DataFrame
        Game-level tournament matchup dataset.
    """
    data = MarchMadnessData()
    df = data.tournament_matchups.copy()

    if year is not None:
        df = df[df["YEAR"] == year].copy()

    if exclude_year is not None:
        df = df[df["YEAR"] != exclude_year].copy()

    if round_number is not None:
        df = df[df["CURRENT ROUND"] == round_number].copy()

    df = df.reset_index(drop=True)

    if len(df) % 2 != 0:
        raise ValueError(
            "Tournament matchup data contains an odd number of rows "
            "and cannot be paired into games."
        )

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

    # Historical outcomes
    if games["score1"].notna().all() and games["score2"].notna().all():
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

        games["score_diff"] = games["score1"] - games["score2"]

    # Seed-based features
    games["better_seed"] = games[["seed1", "seed2"]].min(axis=1)
    games["worse_seed"] = games[["seed1", "seed2"]].max(axis=1)
    games["seed_gap"] = games["worse_seed"] - games["better_seed"]

    games["better_seed_team"] = games["team1"].where(
        games["seed1"] < games["seed2"],
        games["team2"],
    )

    games["worse_seed_team"] = games["team1"].where(
        games["seed1"] > games["seed2"],
        games["team2"],
    )

    equal_seed_mask = games["seed1"] == games["seed2"]

    games.loc[equal_seed_mask, "better_seed_team"] = None
    games.loc[equal_seed_mask, "worse_seed_team"] = None

    if "winner" in games.columns:
        games["better_seed_win"] = (
            games["winner"] == games["better_seed_team"]
        ).astype("Int64")

        games["upset"] = (
            games["winner"] == games["worse_seed_team"]
        ).astype("Int64")

        games.loc[equal_seed_mask, "better_seed_win"] = pd.NA
        games.loc[equal_seed_mask, "upset"] = pd.NA

    return games


def save_historical_tournament_games(
    output_path=DEFAULT_OUTPUT_PATH,
    exclude_year=2026,
):
    """
    Build and save the historical game-level tournament dataset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    games = build_tournament_games(exclude_year=exclude_year)
    games.to_csv(output_path, index=False)

    print(f"Tournament games shape: {games.shape}")
    print(f"Tournament games saved to: {output_path}")

    return games


if __name__ == "__main__":
    save_historical_tournament_games()