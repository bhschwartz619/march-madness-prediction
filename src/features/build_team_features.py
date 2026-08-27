from pathlib import Path
import re

import pandas as pd

from src.data_processing.mm_dataclass import MarchMadnessData


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_team_season_features.csv"

MISSINGNESS_THRESHOLD = 0.60


def clean_team_name(name):
    """
    Standardize team names across data sources.
    """
    if pd.isna(name):
        return pd.NA

    name = str(name).strip().lower()
    name = name.replace(".", "")
    name = name.replace("'", "")
    name = name.replace("-", " ")
    name = re.sub(r"\s+", " ", name)

    team_map = {
        "uconn": "connecticut",
        "st marys": "saint marys",
        "st johns": "saint johns",
        "usc": "southern california",
        "byu": "brigham young",
        "florida st": "florida state",
        "miami fl": "miami",
        "ole miss": "mississippi",
        "texas am": "texas a&m",
        "saint peters": "st peters",
        "vcu": "virginia commonwealth",
        "unc": "north carolina",
        "smu": "southern methodist",
        "lsu": "louisiana state",
        "pitt": "pittsburgh",
    }

    return team_map.get(name, name)


def prep_team_df(df, year_col, team_col, keep_cols, rename_map=None):
    """
    Standardize a team-level source dataset to one row per team-season.
    """
    temp = df.copy()

    if rename_map:
        temp = temp.rename(columns=rename_map)

    temp = temp.rename(columns={year_col: "year", team_col: "team"})

    cols = ["year", "team"] + keep_cols
    temp = temp[cols].copy()

    temp["year"] = pd.to_numeric(temp["year"], errors="coerce").astype("Int64")
    temp["team"] = temp["team"].apply(clean_team_name)

    temp = temp.dropna(subset=["year", "team"])
    temp = temp.drop_duplicates(subset=["year", "team"])

    return temp


def build_team_features(
    missingness_threshold=MISSINGNESS_THRESHOLD,
    drop_sparse=True,
    verbose=False,
):
    """
    Build the master team-season feature dataset from selected source files.
    """
    data = MarchMadnessData()

    # 1. KenPom + Barttorvik
    kenpom = prep_team_df(
        df=data.kenpom_barttorvik,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "kp_adj_em",
            "kp_adj_o",
            "kp_adj_d",
            "kp_adj_t",
            "kp_barthag",
            "kp_efg",
            "kp_ftr",
            "kp_tov",
            "kp_oreb",
            "kp_dreb",
            "kp_three_pct",
            "kp_ft_pct",
            "kp_exp",
            "kp_talent",
            "kp_elite_sos",
            "kp_wab",
        ],
        rename_map={
            "KADJ EM": "kp_adj_em",
            "KADJ O": "kp_adj_o",
            "KADJ D": "kp_adj_d",
            "KADJ T": "kp_adj_t",
            "BARTHAG": "kp_barthag",
            "EFG%": "kp_efg",
            "FTR": "kp_ftr",
            "TOV%": "kp_tov",
            "OREB%": "kp_oreb",
            "DREB%": "kp_dreb",
            "3PT%": "kp_three_pct",
            "FT%": "kp_ft_pct",
            "EXP": "kp_exp",
            "TALENT": "kp_talent",
            "ELITE SOS": "kp_elite_sos",
            "WAB": "kp_wab",
        },
    )

    # 2. Barttorvik Away-Neutral
    bart_an = prep_team_df(
        df=data.barttorvik_away_neutral,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "bart_an_adj_em",
            "bart_an_adj_o",
            "bart_an_adj_d",
            "bart_an_adj_t",
            "bart_an_efg",
            "bart_an_ftr",
            "bart_an_tov",
            "bart_an_oreb",
            "bart_an_dreb",
            "bart_an_three_pct",
            "bart_an_ft_pct",
            "bart_an_elite_sos",
            "bart_an_wab",
        ],
        rename_map={
            "BADJ EM": "bart_an_adj_em",
            "BADJ O": "bart_an_adj_o",
            "BADJ D": "bart_an_adj_d",
            "BADJ T": "bart_an_adj_t",
            "EFG%": "bart_an_efg",
            "FTR": "bart_an_ftr",
            "TOV%": "bart_an_tov",
            "OREB%": "bart_an_oreb",
            "DREB%": "bart_an_dreb",
            "3PT%": "bart_an_three_pct",
            "FT%": "bart_an_ft_pct",
            "ELITE SOS": "bart_an_elite_sos",
            "WAB": "bart_an_wab",
        },
    )

    # 3. EvanMiya
    evan = prep_team_df(
        df=data.evanmiya,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "evan_o_rate",
            "evan_d_rate",
            "evan_relative_rating",
            "evan_opponent_adjust",
            "evan_pace_adjust",
            "evan_true_tempo",
        ],
        rename_map={
            "O RATE": "evan_o_rate",
            "D RATE": "evan_d_rate",
            "RELATIVE RATING": "evan_relative_rating",
            "OPPONENT ADJUST": "evan_opponent_adjust",
            "PACE ADJUST": "evan_pace_adjust",
            "TRUE TEMPO": "evan_true_tempo",
        },
    )

    # 4. RPPF Ratings
    rppf = prep_team_df(
        df=data.rppf_ratings,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "rppf_rating",
            "npb_rating",
            "rppf_adj_o",
            "rppf_adj_d",
            "rppf_adj_em",
            "rppf_pace",
            "rppf_sos",
        ],
        rename_map={
            "RPPF RATING": "rppf_rating",
            "NPB RATING": "npb_rating",
            "RADJ O": "rppf_adj_o",
            "RADJ D": "rppf_adj_d",
            "RADJ EM": "rppf_adj_em",
            "R PACE": "rppf_pace",
            "R SOS": "rppf_sos",
        },
    )

    # 5. Resumes
    resumes = prep_team_df(
        df=data.resumes,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "net_rpi",
            "resume_score",
            "resume_wab_rank",
            "resume_elo",
            "resume_b_power",
            "q1_wins",
            "q2_wins",
            "q1_q2_wins",
            "q3_q4_losses",
            "plus_500",
            "resume_r_score",
        ],
        rename_map={
            "NET RPI": "net_rpi",
            "RESUME": "resume_score",
            "WAB RANK": "resume_wab_rank",
            "ELO": "resume_elo",
            "B POWER": "resume_b_power",
            "Q1 W": "q1_wins",
            "Q2 W": "q2_wins",
            "Q1 PLUS Q2 W": "q1_q2_wins",
            "Q3 Q4 L": "q3_q4_losses",
            "PLUS 500": "plus_500",
            "R SCORE": "resume_r_score",
        },
    )

    # 6. Shooting Splits
    shooting = prep_team_df(
        df=data.shooting_splits,
        year_col="YEAR",
        team_col="TEAM",
        keep_cols=[
            "dunks_fg_pct",
            "dunks_share",
            "close_twos_fg_pct",
            "close_twos_share",
            "farther_twos_fg_pct",
            "farther_twos_share",
            "threes_fg_pct",
            "threes_share",
        ],
        rename_map={
            "DUNKS FG%": "dunks_fg_pct",
            "DUNKS SHARE": "dunks_share",
            "CLOSE TWOS FG%": "close_twos_fg_pct",
            "CLOSE TWOS SHARE": "close_twos_share",
            "FARTHER TWOS FG%": "farther_twos_fg_pct",
            "FARTHER TWOS SHARE": "farther_twos_share",
            "THREES FG%": "threes_fg_pct",
            "THREES SHARE": "threes_share",
        },
    )

    master_df = kenpom.copy()

    for df_part in [bart_an, evan, rppf, resumes, shooting]:
        master_df = master_df.merge(
            df_part,
            on=["year", "team"],
            how="left",
        )

    if drop_sparse:
        cols_to_keep = [
            col
            for col in master_df.columns
            if col in ["year", "team"]
            or master_df[col].isna().mean() < missingness_threshold
        ]
        master_df = master_df[cols_to_keep].copy()

    master_df = master_df.sort_values(["year", "team"]).reset_index(drop=True)

    if verbose:
        print(f"Master dataset shape: {master_df.shape}")
        print(
            f"Duplicate team-season rows: "
            f"{master_df.duplicated(subset=['year', 'team']).sum()}"
        )

        missing_summary = (
            master_df.isna()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .rename("missing_pct")
        )

        print("\nHighest missingness:")
        print(missing_summary.head(15))

    return master_df


def save_team_features(output_path=DEFAULT_OUTPUT_PATH, **build_kwargs):
    """
    Build and save the master team-season feature dataset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master_df = build_team_features(**build_kwargs)
    master_df.to_csv(output_path, index=False)

    print(f"Team features saved to: {output_path}")

    return master_df


if __name__ == "__main__":
    save_team_features(verbose=True)