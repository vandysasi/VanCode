"""
Data Registration script.

Validates that data/tourism.csv has the expected schema before it is
allowed to flow into the rest of the pipeline, and prints a summary.
Run from the repository root: python tourism_project/model_building/data_register.py
"""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: could not find dataset at {DATA_PATH}")
        sys.exit(1)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("Schema validation PASSED.")
    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
