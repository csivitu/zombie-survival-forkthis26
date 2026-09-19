import pandas as pd
from pathlib import Path


MORTALITY_DIR = Path("data/raw/mortality")


MORTALITY_FILES = sorted(
    MORTALITY_DIR.glob("NHANES_*_MORT_2019_PUBLIC.dat")
)


def read_mortality_file(file_path):
    """
    Read one NHANES 2019 public-use linked mortality file.

    The CDC fixed-width layout uses:
        SEQN      positions 1-14
        ELIGSTAT  position 15
        MORTSTAT  position 16
    """

    df = pd.read_fwf(
        file_path,
        colspecs=[
            (0, 14),   # SEQN
            (14, 15),  # ELIGSTAT
            (15, 16),  # MORTSTAT
        ],
        names=[
            "SEQN",
            "ELIGSTAT",
            "MORTSTAT",
        ],
        dtype=str,
    )

    return df


def main():
    print("=" * 70)
    print("NHANES LINKED MORTALITY DATA")
    print("=" * 70)

    all_data = []

    for file_path in MORTALITY_FILES:
        print(f"\nLoading: {file_path.name}")

        df = read_mortality_file(file_path)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("\nFirst 5 rows:")
        print(df.head())

        all_data.append(df)

    mortality = pd.concat(
        all_data,
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("COMBINED MORTALITY DATA")
    print("=" * 70)

    print(f"Total rows: {len(mortality)}")

    print("\nColumns:")
    print(mortality.columns.tolist())

    print("\nMORTSTAT values:")
    print(mortality["MORTSTAT"].value_counts(dropna=False))

    print("\nELIGSTAT values:")
    print(mortality["ELIGSTAT"].value_counts(dropna=False))

    print("\nMissing values:")
    print(mortality.isna().sum())

    print("\nUnique SEQN:")
    print(mortality["SEQN"].nunique())

    print("\nDuplicate SEQN:")
    print(mortality["SEQN"].duplicated().sum())


if __name__ == "__main__":
    main()