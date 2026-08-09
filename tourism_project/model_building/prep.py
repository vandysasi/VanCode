"""
Data Preparation script.

- Loads the dataset from the repository data folder.
- Removes unnecessary columns and fixes label inconsistencies found during EDA.
- Splits into train/test sets and saves them locally so the next GitHub
  Actions job can pick them up as a workflow artifact.
Run from the repository root: python tourism_project/model_building/prep.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
OUT_DIR = "tourism_project/data"
TARGET_COL = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded raw data: {df.shape}")

    # Remove unnecessary identifier columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Handle missing values defensively (dataset has none, but this keeps
    # the pipeline robust to future data refreshes)
    numerical_cols = ["Age", "DurationOfPitch", "NumberOfFollowups", "PreferredPropertyStar",
                       "NumberOfTrips", "PitchSatisfactionScore", "NumberOfChildrenVisiting",
                       "MonthlyIncome"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["TypeofContact", "Occupation", "Gender", "MaritalStatus",
                         "ProductPitched", "Designation"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # Fix data inconsistencies found during EDA
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows (only visible once identifier columns are removed)
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {before - df.shape[0]} duplicate rows")

    df = df.dropna(subset=[TARGET_COL])
    print(f"Cleaned shape: {df.shape}")

    # NOTE: categorical columns are kept as raw strings on purpose. They are
    # one-hot encoded *inside* the model pipeline in train.py, so the
    # Streamlit app can submit human-readable values directly without
    # needing to reproduce any manual label-encoding scheme at inference time.
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df = X_train.copy(); train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy(); test_df[TARGET_COL] = y_test.values

    train_df.to_csv(os.path.join(OUT_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test_data.csv"), index=False)

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")


if __name__ == "__main__":
    main()
