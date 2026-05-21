"""Download NYC TLC yellow taxi trip data and save to data/raw.parquet."""

import os
import pathlib
import requests
import pandas as pd


DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
RAW_PATH = pathlib.Path(__file__).parent.parent / "data" / "raw.parquet"


def download(url: str, dest: pathlib.Path) -> None:
    """Stream-download a file from url to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved to {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def print_stats(df: pd.DataFrame) -> None:
    """Print basic DataFrame statistics."""
    print(f"\nShape:   {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nSample:\n{df.head(3)}")


def main() -> None:
    if not RAW_PATH.exists():
        download(DATA_URL, RAW_PATH)
    else:
        print(f"Already exists: {RAW_PATH}")

    df = pd.read_parquet(RAW_PATH)
    print_stats(df)


if __name__ == "__main__":
    main()
