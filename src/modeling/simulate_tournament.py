from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from src.modeling.predict_tournament import (
    train_primary_model,
    load_2026_master_df,
    load_2026_round_of_64,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIMULATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mc_2026_summary.csv"
)


def choose_winner(
    team1,
    team2,
    team1_win_prob,
    method="deterministic",
    rng=None,
):
    """
    Choose a winner either deterministically
    or via Monte Carlo sampling.
    """
    if method == "deterministic":
        return (
            team1
            if team1_win_prob >= 0.5
            else team2
        )

    if method == "monte_carlo":
        if rng is None:
            rng = np.random.default_rng()

        return (
            team1
            if rng.random() < team1_win_prob
            else team2
        )

    raise ValueError(
        "method must be 'deterministic' "
        "or 'monte_carlo'"
    )


def prepare_team_feature_lookup(
    master_2026,
    feature_columns,
):
    """
    Precompute the model feature vector for every
    2026 tournament team.

    feature_columns contains names such as:
        kp_adj_em_diff

    The corresponding team-level feature is:
        kp_adj_em

    Storing each team's feature vector once avoids
    repeated pandas merges during Monte Carlo
    simulation.
    """
    base_features = [
        col.removesuffix("_diff")
        for col in feature_columns
    ]

    missing_features = [
        col
        for col in base_features
        if col not in master_2026.columns
    ]

    if missing_features:
        raise ValueError(
            "The following training features are "
            "missing from the 2026 master dataset: "
            f"{missing_features}"
        )

    team_lookup = {}

    for _, row in master_2026.iterrows():
        team = row["team"]

        values = pd.to_numeric(
            row[base_features],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        team_lookup[team] = values

    return team_lookup


def prepare_simulation_objects():
    """
    Train the model once and prepare all objects
    needed for repeated tournament simulations.
    """
    trained = train_primary_model()

    master_2026 = (
        load_2026_master_df()
    )

    round64 = (
        load_2026_round_of_64()
        .copy()
        .reset_index(drop=True)
    )

    round64["game_slot"] = range(
        1,
        len(round64) + 1,
    )

    feature_columns = (
        trained["feature_columns"]
    )

    team_feature_lookup = (
        prepare_team_feature_lookup(
            master_2026,
            feature_columns,
        )
    )

    tournament_teams = set(
        round64["team1"]
    ) | set(
        round64["team2"]
    )

    missing_teams = (
        tournament_teams
        - set(team_feature_lookup)
    )

    if missing_teams:
        raise ValueError(
            "The following tournament teams are "
            "missing from the 2026 feature lookup: "
            f"{sorted(missing_teams)}"
        )

    return {
        "model": trained["model"],
        "scaler": trained["scaler"],
        "feature_columns": (
            feature_columns
        ),
        "team_feature_lookup": (
            team_feature_lookup
        ),
        "round64": round64,
        "train_accuracy": (
            trained["train_accuracy"]
        ),
        "test_accuracy": (
            trained["test_accuracy"]
        ),
    }


def build_round_feature_matrix(
    matchups_df,
    team_feature_lookup,
):
    """
    Build Team 1 minus Team 2 feature differences
    for every matchup in a round.

    This replaces repeated pandas merges with direct
    NumPy subtraction.
    """
    team1_vectors = np.vstack(
        [
            team_feature_lookup[team]
            for team in matchups_df["team1"]
        ]
    )

    team2_vectors = np.vstack(
        [
            team_feature_lookup[team]
            for team in matchups_df["team2"]
        ]
    )

    X = (
        team1_vectors
        - team2_vectors
    )

    return X


def impute_round_means(
    X,
):
    """
    Apply the same round-level mean imputation used
    by the original prediction workflow.

    Each missing feature value is replaced with the
    mean of that feature across the current round.
    """
    X = X.copy()

    if not np.isnan(X).any():
        return X

    with np.errstate(
        all="ignore"
    ):
        column_means = np.nanmean(
            X,
            axis=0,
        )

    all_nan_columns = np.isnan(
        column_means
    )

    if all_nan_columns.any():
        indexes = np.where(
            all_nan_columns
        )[0]

        raise ValueError(
            "At least one prediction feature is "
            "missing for every matchup in this "
            "round. Feature indexes: "
            f"{indexes.tolist()}"
        )

    missing_rows, missing_cols = (
        np.where(
            np.isnan(X)
        )
    )

    X[
        missing_rows,
        missing_cols,
    ] = column_means[
        missing_cols
    ]

    return X


def score_matchups_fast(
    matchups_df,
    prepared,
    method="monte_carlo",
    rng=None,
):
    """
    Score one tournament round without repeated
    dataframe merging or feature construction.
    """
    X = build_round_feature_matrix(
        matchups_df,
        prepared[
            "team_feature_lookup"
        ],
    )

    X = impute_round_means(
        X
    )

    X_df = pd.DataFrame(
        X,
        columns=prepared[
            "feature_columns"
        ],
    )

    X_scaled = prepared[
        "scaler"
    ].transform(
        X_df
    )

    probabilities = prepared[
        "model"
    ].predict_proba(
        X_scaled
    )[:, 1]

    results = (
        matchups_df
        .copy()
        .reset_index(drop=True)
    )

    results[
        "team1_win_prob"
    ] = probabilities

    predicted_winners = []
    predicted_losers = []
    pred_team1_win = []

    for i, row in results.iterrows():
        winner = choose_winner(
            row["team1"],
            row["team2"],
            probabilities[i],
            method=method,
            rng=rng,
        )

        if winner == row["team1"]:
            loser = row["team2"]
            team1_win = 1
        else:
            loser = row["team1"]
            team1_win = 0

        predicted_winners.append(
            winner
        )

        predicted_losers.append(
            loser
        )

        pred_team1_win.append(
            team1_win
        )

    results[
        "pred_team1_win"
    ] = pred_team1_win

    results[
        "predicted_winner"
    ] = predicted_winners

    results[
        "predicted_loser"
    ] = predicted_losers

    results[
        "winner_seed"
    ] = results["seed1"].where(
        results[
            "pred_team1_win"
        ] == 1,
        results["seed2"],
    )

    results[
        "loser_seed"
    ] = results["seed2"].where(
        results[
            "pred_team1_win"
        ] == 1,
        results["seed1"],
    )

    return results


def advance_round(
    scored_round_df,
    next_round_value,
):
    """
    Build next-round matchups from simulated
    winners using bracket order.
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

    missing = (
        required_cols
        - set(round_df.columns)
    )

    if missing:
        raise ValueError(
            "advance_round is missing "
            f"required columns: {missing}"
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
            "predicted_winner": (
                "team"
            ),
            "winner_seed": (
                "seed"
            ),
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
            "year": (
                team1["year"]
            ),
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
            "team1": (
                team1["team"]
            ),
            "seed1": (
                team1["seed"]
            ),
            "team2": (
                team2["team"]
            ),
            "seed2": (
                team2["seed"]
            ),
        }
    )

    return next_round


def simulate_one_tournament_prepared(
    prepared,
    rng=None,
):
    """
    Run one complete Monte Carlo tournament using
    preloaded model and team-feature objects.
    """
    round64 = (
        prepared["round64"]
        .copy()
    )

    scored_64 = score_matchups_fast(
        round64,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    round32 = advance_round(
        scored_64,
        next_round_value=32,
    )

    scored_32 = score_matchups_fast(
        round32,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    sweet16 = advance_round(
        scored_32,
        next_round_value=16,
    )

    scored_16 = score_matchups_fast(
        sweet16,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    elite8 = advance_round(
        scored_16,
        next_round_value=8,
    )

    scored_8 = score_matchups_fast(
        elite8,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    final4 = advance_round(
        scored_8,
        next_round_value=4,
    )

    scored_4 = score_matchups_fast(
        final4,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    title_game = advance_round(
        scored_4,
        next_round_value=2,
    )

    scored_2 = score_matchups_fast(
        title_game,
        prepared,
        method="monte_carlo",
        rng=rng,
    )

    return {
        "round64": scored_64,
        "round32": scored_32,
        "sweet16": scored_16,
        "elite8": scored_8,
        "final4": scored_4,
        "championship": scored_2,
        "round32_teams": (
            scored_64[
                "predicted_winner"
            ].tolist()
        ),
        "sweet16_teams": (
            scored_32[
                "predicted_winner"
            ].tolist()
        ),
        "elite8_teams": (
            scored_16[
                "predicted_winner"
            ].tolist()
        ),
        "final_four_teams": (
            scored_8[
                "predicted_winner"
            ].tolist()
        ),
        "title_game_teams": (
            scored_2[
                "team1"
            ].iloc[0],
            scored_2[
                "team2"
            ].iloc[0],
        ),
        "champion": (
            scored_2[
                "predicted_winner"
            ].iloc[0]
        ),
    }


def run_monte_carlo_simulation(
    n_sims=10000,
    random_seed=42,
):
    """
    Run repeated tournament simulations and
    summarize advancement probabilities.
    """
    prepared = (
        prepare_simulation_objects()
    )

    rng = np.random.default_rng(
        random_seed
    )

    champion_counter = Counter()
    title_game_counter = Counter()
    final_four_counter = Counter()
    elite8_counter = Counter()
    sweet16_counter = Counter()
    round32_counter = Counter()

    for i in range(n_sims):
        sim = (
            simulate_one_tournament_prepared(
                prepared,
                rng=rng,
            )
        )

        champion_counter.update(
            [sim["champion"]]
        )

        title_game_counter.update(
            sim["title_game_teams"]
        )

        final_four_counter.update(
            sim["final_four_teams"]
        )

        elite8_counter.update(
            sim["elite8_teams"]
        )

        sweet16_counter.update(
            sim["sweet16_teams"]
        )

        round32_counter.update(
            sim["round32_teams"]
        )

        if (i + 1) % 500 == 0:
            print(
                f"Completed {i + 1} "
                "simulations..."
            )

    all_teams = (
        set(round32_counter)
        | set(sweet16_counter)
        | set(elite8_counter)
        | set(final_four_counter)
        | set(title_game_counter)
        | set(champion_counter)
    )

    summary = pd.DataFrame(
        {
            "team": sorted(
                all_teams
            )
        }
    )

    summary["round32_prob"] = (
        summary["team"].map(
            lambda team:
            round32_counter[team]
            / n_sims
        )
    )

    summary["sweet16_prob"] = (
        summary["team"].map(
            lambda team:
            sweet16_counter[team]
            / n_sims
        )
    )

    summary["elite8_prob"] = (
        summary["team"].map(
            lambda team:
            elite8_counter[team]
            / n_sims
        )
    )

    summary["final4_prob"] = (
        summary["team"].map(
            lambda team:
            final_four_counter[team]
            / n_sims
        )
    )

    summary["title_game_prob"] = (
        summary["team"].map(
            lambda team:
            title_game_counter[team]
            / n_sims
        )
    )

    summary["champion_prob"] = (
        summary["team"].map(
            lambda team:
            champion_counter[team]
            / n_sims
        )
    )

    summary = (
        summary
        .sort_values(
            [
                "champion_prob",
                "final4_prob",
                "elite8_prob",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "prepared": prepared,
        "summary": summary,
        "champion_counter": (
            champion_counter
        ),
        "title_game_counter": (
            title_game_counter
        ),
        "final_four_counter": (
            final_four_counter
        ),
        "elite8_counter": (
            elite8_counter
        ),
        "sweet16_counter": (
            sweet16_counter
        ),
        "round32_counter": (
            round32_counter
        ),
    }


def save_monte_carlo_results(
    summary_df,
    output_path=SIMULATION_OUTPUT_PATH,
):
    """
    Save Monte Carlo summary results.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved Monte Carlo results to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    mc = run_monte_carlo_simulation(
        n_sims=10000,
        random_seed=42,
    )

    print(
        "Train accuracy:",
        mc["prepared"][
            "train_accuracy"
        ],
    )

    print(
        "Test accuracy:",
        mc["prepared"][
            "test_accuracy"
        ],
    )

    print(
        "\nTop 10 teams by "
        "championship probability:"
    )

    print(
        mc["summary"][
            [
                "team",
                "round32_prob",
                "sweet16_prob",
                "elite8_prob",
                "final4_prob",
                "title_game_prob",
                "champion_prob",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    save_monte_carlo_results(
        mc["summary"]
    )