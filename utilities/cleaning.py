"""
Utilities for inspecting and cleaning the Superstore dataset.

This module provides reusable functions that handle the standard data-quality
workflow: report on a DataFrame's state, check categorical consistency, find
outliers, standardise text columns, and run the full cleaning pipeline.
"""
import pandas as pd

def report(df):
    """
    Print a quick diagnostic summary of a DataFrame.

    Shows shape, duplicate count, missing-value counts per column,
    the first/last 5 rows, and the column dtypes via df.info().
    Useful as a "before and after" snapshot when cleaning.
    """
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    total_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {total_dupes}")

    null_values = df.isnull().sum()
    null_values = null_values[null_values != 0]
    if len(null_values) > 0:
        print("\nMissing values:")
        for col, val in null_values.items():
                print(f"{col}: {val} missing values")
    else:
        print("\nNo missing values")

    print("\nThe first and last 5 rows of the dataset:")
    print(df.head(5))
    print(df.tail(5))

    print()
    df.info()

def find_outliers(df, col):
    """
    Identify outliers in a numeric column using the IQR method.

    An outlier is any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    Returns the subset of rows that are outliers (callers can decide what
    to do with them) and prints mean/median/count for quick comparison.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].reset_index(drop=True)

    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Outliers found: {len(outliers)}")

def check_categories(df):
    """
    Print value_counts() for each categorical column we care about.

    Useful for spotting casing/whitespace inconsistencies — e.g. 'West'
    and ' west ' showing up as separate categories. Skips numeric columns
    silently.
    """
    # cols = [col for col in df.columns if col in CATEGORICAL_TEXT_COLS]
    cols = [col for col in df.columns if df[col].dtype == object]

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
          continue
        print(f"Categories in {col}")
        print(df[col].value_counts())
        print()

def standardize_text(df):
    """
    Strip whitespace and title-case the categorical text columns.

    Returns the modified DataFrame. Only touches the allowlist defined in
    CATEGORICAL_TEXT_COLS — IDs (Order ID, Product ID) and free-text fields
    (Product Name) are left alone, since title-casing them would mangle
    real formatting like 'VoIP' or 'CA-2016-152156'.
    """
    cols_to_clean = ["Region", "Ship Mode", "Category", "Segment", "Sub-Category"]

    for col in cols_to_clean:
        df[col] = df[col].str.strip().str.title()

def clean_data(df):
    """
    Run the full cleaning pipeline on the Superstore DataFrame.

    Order matters:
    1. Standardise text first — so duplicates and groupbys see consistent values.
    2. Drop duplicates — now reliable because text has been normalised.
    3. Handle missing values — drop missing Sales (a measure we sum, so filling
       would invent revenue); fill Postal Code with 'Unknown' (an identifier
       we never do math on).
    4. Parse dates last — the rest of the data is sound before we change types.

    Returns the cleaned DataFrame.
    """
    standardize_text(df)

    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values
    df = df.dropna(subset=["Sales"]).reset_index(drop=True)
    df["Postal Code"] = df["Postal Code"].fillna("Unknown")

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=False)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed", dayfirst=False)

    return df








