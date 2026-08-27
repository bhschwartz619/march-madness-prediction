import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from src.features.build_team_features import (
    build_team_features,
    clean_team_name,
)

from src.features.build_tournament_games import (
    build_tournament_games,
)


def load_datasets(exclude_2026=True):
    """
    Build the historical tournament game dataset and
    team-season feature dataset.

    This replaces the hard-coded CSV paths used in the original
    project while preserving the same modeling inputs.
    """
    games = build_tournament_games()
    teams = build_team_features()

    if exclude_2026:
        games = games[
            games["year"] != 2026
        ].copy()

    return games, teams


def drop_leakage_columns(games_df):
    """
    Drop the same columns excluded in the original modeling project.
    """
    games = games_df.copy()

    leakage_cols = [
        "score1",
        "score2",
        "winner",
        "loser",
        "score_diff",
        "team1_round_fin",
        "team2_round_fin",
        "better_seed",
        "worse_seed",
        "better_seed_team",
        "worse_seed_team",
        "better_seed_score",
        "worse_seed_score",
        "better_seed_win",
        "upset",
    ]

    games = games.drop(
        columns=[
            col
            for col in leakage_cols
            if col in games.columns
        ]
    )

    return games


def clean_dataset_team_names(
    games_df,
    teams_df,
):
    """
    Standardize team names in both datasets.
    """
    games = games_df.copy()
    teams = teams_df.copy()

    games["team1"] = games["team1"].apply(
        clean_team_name
    )

    games["team2"] = games["team2"].apply(
        clean_team_name
    )

    teams["team"] = teams["team"].apply(
        clean_team_name
    )

    return games, teams


def merge_team_features(
    games_df,
    teams_df,
):
    """
    Merge team-season features onto Team 1 and Team 2.
    """
    games = games_df.copy()
    teams = teams_df.copy()

    # Merge Team 1
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

    games = games.rename(
        columns=team1_cols
    )

    games = games.drop(
        columns=["team"]
    )

    # Merge Team 2
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

    games = games.rename(
        columns=team2_cols
    )

    games = games.drop(
        columns=["team"]
    )

    return games


def create_difference_features(
    games_with_features_df,
    teams_df,
):
    """
    Create Team 1 minus Team 2 differential features.
    """
    games = games_with_features_df.copy()

    feature_cols = [
        col
        for col in teams_df.columns
        if col not in ["year", "team"]
    ]

    for col in feature_cols:
        t1_col = f"{col}_t1"
        t2_col = f"{col}_t2"

        if (
            t1_col in games.columns
            and t2_col in games.columns
        ):
            games[f"{col}_diff"] = (
                games[t1_col]
                - games[t2_col]
            )

    return games


def build_modeling_dataset(
    games_with_diffs_df,
    drop_rppf=True,
    drop_evan=True,
    impute=True,
):
    """
    Build X and y using the original project's feature logic.
    """
    games = games_with_diffs_df.copy()

    y = games["team1_win"].copy()

    X = games[
        [
            col
            for col in games.columns
            if col.endswith("_diff")
        ]
    ].copy()

    if drop_rppf:
        X = X.drop(
            columns=[
                col
                for col in X.columns
                if (
                    "rppf" in col
                    or "npb" in col
                )
            ],
            errors="ignore",
        )

    if drop_evan:
        X = X.drop(
            columns=[
                col
                for col in X.columns
                if "evan" in col
            ],
            errors="ignore",
        )

    if impute:
        X = X.fillna(
            X.mean()
        )

    return X, y


def split_train_test_by_year(
    games_df,
    X,
    y,
    split_year=2022,
):
    """
    Use tournaments before 2022 for training and
    tournaments from 2022 onward for testing.
    """
    train = games_df[
        games_df["year"] < split_year
    ].copy()

    test = games_df[
        games_df["year"] >= split_year
    ].copy()

    X_train = X.loc[
        train.index
    ].copy()

    y_train = y.loc[
        train.index
    ].copy()

    X_test = X.loc[
        test.index
    ].copy()

    y_test = y.loc[
        test.index
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def scale_features(
    X_train,
    X_test,
):
    """
    Standardize features using the training data.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


def train_logistic_regression(
    X_train,
    y_train,
    max_iter=1000,
):
    """
    Train logistic regression.
    """
    model = LogisticRegression(
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=6,
    random_state=42,
):
    """
    Train random forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_gradient_boosting(
    X_train,
    y_train,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
):
    """
    Train gradient boosting classifier.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Calculate training and testing accuracy.
    """
    train_accuracy = model.score(
        X_train,
        y_train,
    )

    test_accuracy = model.score(
        X_test,
        y_test,
    )

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def get_linear_feature_importance(
    model,
    feature_names,
):
    """
    Return logistic regression coefficients
    sorted by absolute magnitude.
    """
    importance = pd.Series(
        model.coef_[0],
        index=feature_names,
    )

    importance = importance.sort_values(
        key=abs,
        ascending=False,
    )

    return importance


def get_tree_feature_importance(
    model,
    feature_names,
):
    """
    Return tree-based feature importance scores.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    )

    importance = importance.sort_values(
        ascending=False
    )

    return importance


def build_merged_games_dataset(
    exclude_2026=True,
):
    """
    Build the historical tournament modeling table.
    """
    games, teams = load_datasets(
        exclude_2026=exclude_2026
    )

    games = drop_leakage_columns(
        games
    )

    games, teams = clean_dataset_team_names(
        games,
        teams,
    )

    games = merge_team_features(
        games,
        teams,
    )

    games = create_difference_features(
        games,
        teams,
    )

    return games, teams


def run_full_modeling_pipeline(
    exclude_2026=True,
    split_year=2022,
    drop_rppf=True,
    drop_evan=True,
):
    """
    Run the original model comparison workflow.
    """
    games, teams = build_merged_games_dataset(
        exclude_2026=exclude_2026
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

    # Logistic Regression
    log_model = train_logistic_regression(
        X_train_scaled,
        y_train,
    )

    log_eval = evaluate_model(
        log_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    log_importance = (
        get_linear_feature_importance(
            log_model,
            X.columns,
        )
    )

    # Random Forest
    rf_model = train_random_forest(
        X_train_scaled,
        y_train,
    )

    rf_eval = evaluate_model(
        rf_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    rf_importance = (
        get_tree_feature_importance(
            rf_model,
            X.columns,
        )
    )

    # Gradient Boosting
    gb_model = train_gradient_boosting(
        X_train_scaled,
        y_train,
    )

    gb_eval = evaluate_model(
        gb_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    gb_importance = (
        get_tree_feature_importance(
            gb_model,
            X.columns,
        )
    )

    return {
        "games": games,
        "teams": teams,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "log_model": log_model,
        "rf_model": rf_model,
        "gb_model": gb_model,
        "log_eval": log_eval,
        "rf_eval": rf_eval,
        "gb_eval": gb_eval,
        "log_importance": log_importance,
        "rf_importance": rf_importance,
        "gb_importance": gb_importance,
    }


if __name__ == "__main__":
    results = run_full_modeling_pipeline()

    print("LOGISTIC REGRESSION")
    print(
        "Train accuracy:",
        results["log_eval"][
            "train_accuracy"
        ],
    )
    print(
        "Test accuracy:",
        results["log_eval"][
            "test_accuracy"
        ],
    )
    print(
        results["log_importance"]
    )

    print("\nRANDOM FOREST")
    print(
        "Train accuracy:",
        results["rf_eval"][
            "train_accuracy"
        ],
    )
    print(
        "Test accuracy:",
        results["rf_eval"][
            "test_accuracy"
        ],
    )
    print(
        results["rf_importance"]
    )

    print("\nGRADIENT BOOSTING")
    print(
        "Train accuracy:",
        results["gb_eval"][
            "train_accuracy"
        ],
    )
    print(
        "Test accuracy:",
        results["gb_eval"][
            "test_accuracy"
        ],
    )
    print(
        results["gb_importance"]
    )