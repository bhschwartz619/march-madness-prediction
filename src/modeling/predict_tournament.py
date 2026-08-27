from pathlib import Path

import pandas as pd

from src.features.build_team_features import (
    build_team_features,
    clean_team_name,
)

from src.modeling.train_models import (
    build_merged_games_dataset,
    build_modeling_dataset,
    split_train_test_by_year,
    scale_features,
    train_logistic_regression,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROUND_OF_64_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "2026_round_of_64.csv"
)


def train_primary_model(
    split_year=2022,
    drop_rppf=True,
    drop_evan=True,
):
    """
    Train the primary logistic regression model on
    historical tournament games.
    """
    games, teams = build_merged_games_dataset(
        exclude_2026=True
    )

    X, y = build_modeling_dataset(
        games,
        drop_rppf=drop_rppf,
        drop_evan=drop_evan,
        impute=True,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test_by_year(
        games,
        X,
        y,
        split_year=split_year,
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_test,
    )

    model = train_logistic_regression(
        X_train_scaled,
        y_train,
    )

    train_accuracy = model.score(
        X_train_scaled,
        y_train,
    )

    test_accuracy = model.score(
        X_test_scaled,
        y_test,
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
        "historical_games": games,
        "historical_X": X,
        "historical_y": y,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def load_2026_master_df():
    """
    Build the master team feature dataset and
    retain only 2026 teams.
    """
    master_df = build_team_features(
        verbose=False
    )

    master_2026 = master_df[
        master_df["year"] == 2026
    ].copy()

    master_2026["team"] = (
        master_2026["team"].apply(
            clean_team_name
        )
    )

    return master_2026


def load_2026_round_of_64():
    """
    Load the frozen 2026 Round of 64 bracket used
    for the original project.

    Using a repository snapshot keeps the prediction
    reproducible even if the upstream Kaggle dataset
    changes later.
    """
    df = pd.read_csv(
        ROUND_OF_64_PATH
    )

    df = df[
        (df["YEAR"] == 2026)
        & (df["CURRENT ROUND"] == 64)
    ].copy()

    if len(df) != 64:
        raise ValueError(
            "Expected 64 team rows for the "
            "2026 Round of 64, but found "
            f"{len(df)}."
        )

    df = df.reset_index(
        drop=True
    )

    team1 = (
        df.iloc[0::2]
        .reset_index(drop=True)
    )

    team2 = (
        df.iloc[1::2]
        .reset_index(drop=True)
    )

    games = pd.DataFrame(
        {
            "year": team1["YEAR"],
            "current_round": team1[
                "CURRENT ROUND"
            ],
            "team1_by_year_no": team1[
                "BY YEAR NO"
            ],
            "team1_round_fin": team1[
                "ROUND"
            ],
            "team1_no": team1[
                "TEAM NO"
            ],
            "team1": team1["TEAM"],
            "seed1": team1["SEED"],
            "team2_by_year_no": team2[
                "BY YEAR NO"
            ],
            "team2_round_fin": team2[
                "ROUND"
            ],
            "team2_no": team2[
                "TEAM NO"
            ],
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

    games["better_seed_team"] = (
        games["team1"].where(
            games["seed1"]
            < games["seed2"],
            games["team2"],
        )
    )

    games["worse_seed_team"] = (
        games["team2"].where(
            games["seed1"]
            < games["seed2"],
            games["team1"],
        )
    )

    equal_seed_mask = (
        games["seed1"]
        == games["seed2"]
    )

    games.loc[
        equal_seed_mask,
        "better_seed_team",
    ] = None

    games.loc[
        equal_seed_mask,
        "worse_seed_team",
    ] = None

    games["team1"] = (
        games["team1"].apply(
            clean_team_name
        )
    )

    games["team2"] = (
        games["team2"].apply(
            clean_team_name
        )
    )

    if len(games) != 32:
        raise ValueError(
            "Expected 32 Round of 64 games, "
            f"but found {len(games)}."
        )

    return games


def merge_prediction_features(
    matchups_df,
    master_2026_df,
):
    """
    Merge team-level 2026 features onto both teams
    in each matchup.
    """
    games = matchups_df.copy()
    teams = master_2026_df.copy()

    games = games.merge(
        teams,
        left_on=["year", "team1"],
        right_on=["year", "team"],
        how="left",
    )

    team1_cols = {
        col: f"{col}_t1"
        for col in teams.columns
        if col not in ["year", "team"]
    }

    games = (
        games.rename(
            columns=team1_cols
        )
        .drop(
            columns=["team"]
        )
    )

    games = games.merge(
        teams,
        left_on=["year", "team2"],
        right_on=["year", "team"],
        how="left",
    )

    team2_cols = {
        col: f"{col}_t2"
        for col in teams.columns
        if col not in ["year", "team"]
    }

    games = (
        games.rename(
            columns=team2_cols
        )
        .drop(
            columns=["team"]
        )
    )

    return games


def create_prediction_diffs(
    merged_matchups_df,
    master_2026_df,
):
    """
    Create Team 1 minus Team 2 feature differences.
    """
    games = (
        merged_matchups_df.copy()
    )

    feature_cols = [
        col
        for col in master_2026_df.columns
        if col not in ["year", "team"]
    ]

    for col in feature_cols:
        team1_col = f"{col}_t1"
        team2_col = f"{col}_t2"

        if (
            team1_col in games.columns
            and team2_col in games.columns
        ):
            games[f"{col}_diff"] = (
                games[team1_col]
                - games[team2_col]
            )

    return games


def build_prediction_matrix(
    games_with_diffs_df,
    feature_columns,
    drop_rppf=True,
    drop_evan=True,
):
    """
    Build prediction features and align them to the
    historical training feature set.
    """
    games = (
        games_with_diffs_df.copy()
    )

    X_pred = games[
        [
            col
            for col in games.columns
            if col.endswith("_diff")
        ]
    ].copy()

    if drop_rppf:
        X_pred = X_pred.drop(
            columns=[
                col
                for col in X_pred.columns
                if (
                    "rppf" in col
                    or "npb" in col
                )
            ],
            errors="ignore",
        )

    if drop_evan:
        X_pred = X_pred.drop(
            columns=[
                col
                for col in X_pred.columns
                if "evan" in col
            ],
            errors="ignore",
        )

    for col in feature_columns:
        if col not in X_pred.columns:
            X_pred[col] = pd.NA

    X_pred = X_pred[
        feature_columns
    ].copy()

    X_pred = X_pred.fillna(
        X_pred.mean()
    )

    return X_pred


def score_matchups(
    matchups_df,
    X_pred,
    model,
    scaler,
):
    """
    Score tournament matchups using the trained
    logistic regression model.
    """
    X_scaled = scaler.transform(
        X_pred
    )

    probabilities = (
        model.predict_proba(
            X_scaled
        )[:, 1]
    )

    predictions = model.predict(
        X_scaled
    )

    results = matchups_df.copy()

    results[
        "team1_win_prob"
    ] = probabilities

    results[
        "pred_team1_win"
    ] = predictions

    results[
        "predicted_winner"
    ] = results["team1"].where(
        results["pred_team1_win"] == 1,
        results["team2"],
    )

    results[
        "predicted_loser"
    ] = results["team2"].where(
        results["pred_team1_win"] == 1,
        results["team1"],
    )

    results[
        "winner_seed"
    ] = results["seed1"].where(
        results["pred_team1_win"] == 1,
        results["seed2"],
    )

    results[
        "loser_seed"
    ] = results["seed2"].where(
        results["pred_team1_win"] == 1,
        results["seed1"],
    )

    return results


def advance_round(
    scored_round_df,
    next_round_value,
):
    """
    Advance predicted winners into the next round
    based on bracket order.
    """
    round_df = (
        scored_round_df
        .copy()
        .reset_index(drop=True)
    )

    required_cols = {
        "year",
        "predicted_winner",
        "winner_seed",
    }

    missing_cols = (
        required_cols
        - set(round_df.columns)
    )

    if missing_cols:
        raise ValueError(
            "advance_round is missing "
            "required columns: "
            f"{missing_cols}"
        )

    if "game_slot" not in round_df.columns:
        round_df["game_slot"] = range(
            1,
            len(round_df) + 1,
        )

    round_df = (
        round_df
        .sort_values("game_slot")
        .reset_index(drop=True)
    )

    if len(round_df) % 2 != 0:
        raise ValueError(
            "Round has an odd number of games; "
            "cannot build next round."
        )

    winners = round_df[
        [
            "year",
            "game_slot",
            "predicted_winner",
            "winner_seed",
        ]
    ].copy()

    winners = winners.rename(
        columns={
            "predicted_winner": "team",
            "winner_seed": "seed",
        }
    )

    team1 = (
        winners
        .iloc[0::2]
        .reset_index(drop=True)
    )

    team2 = (
        winners
        .iloc[1::2]
        .reset_index(drop=True)
    )

    next_round = pd.DataFrame(
        {
            "year": team1["year"],
            "current_round": (
                next_round_value
            ),
            "game_slot": range(
                1,
                len(team1) + 1,
            ),
            "source_slot_team1": (
                team1["game_slot"]
            ),
            "source_slot_team2": (
                team2["game_slot"]
            ),
            "team1": team1["team"],
            "seed1": team1["seed"],
            "team2": team2["team"],
            "seed2": team2["seed"],
        }
    )

    return next_round


def predict_round(
    matchups_df,
    master_2026_df,
    model,
    scaler,
    feature_columns,
):
    """
    Predict all games in a tournament round.
    """
    merged = merge_prediction_features(
        matchups_df,
        master_2026_df,
    )

    merged = create_prediction_diffs(
        merged,
        master_2026_df,
    )

    X_pred = build_prediction_matrix(
        merged,
        feature_columns,
    )

    scored = score_matchups(
        merged,
        X_pred,
        model,
        scaler,
    )

    return scored


def simulate_full_bracket():
    """
    Predict the 2026 tournament bracket from the
    Round of 64 through the championship.
    """
    trained = train_primary_model()

    model = trained["model"]
    scaler = trained["scaler"]
    feature_columns = (
        trained["feature_columns"]
    )

    master_2026 = load_2026_master_df()

    round64 = (
        load_2026_round_of_64()
    )

    scored_64 = predict_round(
        round64,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    round32 = advance_round(
        scored_64,
        next_round_value=32,
    )

    scored_32 = predict_round(
        round32,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    sweet16 = advance_round(
        scored_32,
        next_round_value=16,
    )

    scored_16 = predict_round(
        sweet16,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    elite8 = advance_round(
        scored_16,
        next_round_value=8,
    )

    scored_8 = predict_round(
        elite8,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    final4 = advance_round(
        scored_8,
        next_round_value=4,
    )

    scored_4 = predict_round(
        final4,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    championship = advance_round(
        scored_4,
        next_round_value=2,
    )

    scored_2 = predict_round(
        championship,
        master_2026,
        model,
        scaler,
        feature_columns,
    )

    champion = (
        scored_2[
            "predicted_winner"
        ].iloc[0]
    )

    final_four_teams = (
        scored_8[
            "predicted_winner"
        ].tolist()
    )

    return {
        "train_accuracy": (
            trained["train_accuracy"]
        ),
        "test_accuracy": (
            trained["test_accuracy"]
        ),
        "round64": scored_64,
        "round32": scored_32,
        "sweet16": scored_16,
        "elite8": scored_8,
        "final4": scored_4,
        "championship": scored_2,
        "final_four_teams": (
            final_four_teams
        ),
        "champion": champion,
    }


def print_round(
    round_name,
    round_df,
):
    """
    Print a tournament round in a readable format.
    """
    print(
        f"\n=== {round_name} ==="
    )

    columns_to_show = [
        "team1",
        "seed1",
        "team2",
        "seed2",
        "team1_win_prob",
        "predicted_winner",
    ]

    print(
        round_df[
            columns_to_show
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    results = simulate_full_bracket()

    print(
        "Train accuracy:",
        results["train_accuracy"],
    )

    print(
        "Test accuracy:",
        results["test_accuracy"],
    )

    print_round(
        "ROUND OF 64",
        results["round64"],
    )

    print_round(
        "ROUND OF 32",
        results["round32"],
    )

    print_round(
        "SWEET 16",
        results["sweet16"],
    )

    print_round(
        "ELITE 8",
        results["elite8"],
    )

    print_round(
        "FINAL FOUR",
        results["final4"],
    )

    print_round(
        "CHAMPIONSHIP",
        results["championship"],
    )

    print(
        "\nPredicted Final Four:"
    )

    for team in results[
        "final_four_teams"
    ]:
        print(
            "-",
            team,
        )

    print(
        "\nPredicted Champion:"
    )

    print(
        results["champion"]
    )