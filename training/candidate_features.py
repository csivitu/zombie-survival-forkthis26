from data_loader import build_dataset, CANDIDATE_FEATURES


def main():
    df = build_dataset()

    print("=" * 70)
    print("CURRENT CANDIDATE FEATURES")
    print("=" * 70)

    for feature in CANDIDATE_FEATURES:

        if feature not in df.columns:
            print(f"\n{feature}: NOT AVAILABLE")
            continue

        missing = df[feature].isna().sum()
        missing_pct = df[feature].isna().mean() * 100
        unique = df[feature].nunique(dropna=True)

        print(f"\n{feature}")
        print(f"  Missing: {missing:,} ({missing_pct:.2f}%)")
        print(f"  Unique values: {unique}")

        print("  Values:")
        print(df[feature].value_counts(dropna=False).head(15))


if __name__ == "__main__":
    main()