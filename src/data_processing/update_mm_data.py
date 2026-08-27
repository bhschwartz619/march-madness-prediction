from pathlib import Path
import shutil
import kagglehub

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Kaggle dataset
kaggle_dataset = "nishaanamin/march-madness-data"

def update_mm_data(data_dir=DATA_DIR):
    """
    Download the most up-to-date March Madness dataset from Kaggle and copy the CSV files
    into the project's data/raw directory

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Destination directory for the downloaded dataset

    Returns
    ----------
    pathlib.Path
        Path to the local raw-data directory
    """
    data_dir = Path(data_dir)

    print("Downloading March Madness data from Kaggle...")
    download_path = Path(kagglehub.dataset_download(kaggle_dataset))

    # Create destination directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously downloaded CSV files
    for file_path in data_dir.glob("*.csv"):
        file_path.unlink()

    # Copy downloaded CSV files into data/raw
    csv_files = list(download_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the downloaded dataset at {download_path}")

    for file_path in csv_files:
        shutil.copy2(file_path, data_dir / file_path.name)

    print (f"March Madness data updated: {len(csv_files)} CSV files copied to {data_dir}")

    return data_dir

if __name__ == "__main__":
    update_mm_data()