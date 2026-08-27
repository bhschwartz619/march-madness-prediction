from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "march_madness_data_audit.xlsx"


def export_data_audit_to_excel(data, output_path=DEFAULT_OUTPUT_PATH):
    """
    Export a data-quality audit for all loaded March Madness datasets.

    The workbook includes:
    - A dataset-level summary
    - A column-level audit sheet for each dataset

    Parameters
    ----------
    data : MarchMadnessData
        Loaded March Madness datasets.

    output_path : pathlib.Path or str
        Destination path for the Excel audit workbook.

    Returns
    -------
    pathlib.Path
        Path to the generated audit workbook.
    """
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        for name, df in data.files.items():

            summary_rows.append(
                {
                    "dataset": name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "duplicate_rows": df.duplicated().sum(),
                    "missing_cells": df.isna().sum().sum(),
                    "missing_pct": round(
                        df.isna().sum().sum() / df.size * 100, 2
                    )
                    if df.size > 0
                    else 0.0,
                }
            )

            column_summary = pd.DataFrame(
                {
                    "column": df.columns,
                    "dtype": df.dtypes.astype(str).values,
                    "missing_values": df.isna().sum().values,
                    "missing_pct": (df.isna().mean() * 100).round(2).values,
                    "n_unique": df.nunique(dropna=True).values,
                }
            )

            # Excel sheet names are limited to 31 characters
            sheet_name = name[:31]
            column_summary.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("dataset")

        summary_df.to_excel(
            writer,
            sheet_name="dataset_summary",
            index=False,
        )

    print(f"Data audit exported to: {output_path}")

    return output_path