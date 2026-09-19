from data_loader import build_dataset


def main():
    df = build_dataset()

    print("=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nAll columns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:3}. {column}")

    print("\n" + "=" * 70)
    print("MISSINGNESS")
    print("=" * 70)

    missing = df.isna().mean() * 100

    missing = (
        missing
        .sort_values()
        .to_frame("missing_percent")
    )

    print(missing.to_string())

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()