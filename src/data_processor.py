"""
Expense Tracker App - Data Processor Module
Handles data cleaning, schema validation, type casting, missing value imputation,
and robust feature engineering for financial analytics and predictive modeling.
"""

import os
import pandas as pd
import numpy as np


def clean_and_process_expenses(input_source, output_path: str = None) -> pd.DataFrame:
    """
    Cleans raw expense dataset, removes anomalies, formats dates, and enriches
    the dataframe with temporal and behavioral features.
    
    Args:
        input_source: Filepath string (CSV) or an existing pd.DataFrame.
        output_path: Optional path to save processed CSV.
        
    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset.
    """
    if isinstance(input_source, str):
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Input file not found: {input_source}")
        df = pd.read_csv(input_source)
    elif isinstance(input_source, pd.DataFrame):
        df = input_source.copy()
    else:
        raise ValueError("input_source must be a valid file path string or pandas DataFrame.")

    # 1. Deduplication
    initial_count = len(df)
    df = df.drop_duplicates(subset=["Transaction_ID"])
    dedup_dropped = initial_count - len(df)
    if dedup_dropped > 0:
        print(f"[INFO] Dropped {dedup_dropped} duplicate transactions.")

    # 2. Date parsing and validation
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # 3. Numeric sanitization
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    df = df[df["Amount"] > 0]  # Filter out non-positive transactions

    # 4. Categorical normalization
    categorical_cols = ["Category", "Sub_Category", "Payment_Mode", "Expense_Type", "Location"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 5. Feature Engineering: Temporal dimensions
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Year_Month"] = df["Date"].dt.strftime("%Y-%m")
    df["Day"] = df["Date"].dt.day
    df["Day_Name"] = df["Date"].dt.strftime("%A")
    df["Day_Of_Week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["Is_Weekend"] = df["Day_Of_Week"].apply(lambda x: 1 if x >= 5 else 0)
    df["Weekend_Label"] = df["Is_Weekend"].map({1: "Weekend", 0: "Weekday"})
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["Week_Of_Year"] = df["Date"].dt.isocalendar().week

    # 6. Sorting & Cumulative Running Total per Month
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Monthly_Cumulative_Spend"] = df.groupby("Year_Month")["Amount"].cumsum()

    # 7. Expense Classification Safeguard
    if "Expense_Type" not in df.columns or df["Expense_Type"].isnull().any():
        need_keywords = ["Rent", "Groceries", "Utilities", "Bills", "Medical", "Healthcare", "Fuel", "Transport"]
        want_keywords = ["Dining", "Restaurant", "Shopping", "Entertainment", "Electronics", "Lifestyle"]
        
        def infer_type(category):
            cat_lower = str(category).lower()
            if any(k.lower() in cat_lower for k in need_keywords):
                return "Need"
            if any(k.lower() in cat_lower for k in want_keywords):
                return "Want"
            if "invest" in cat_lower or "saving" in cat_lower or "sip" in cat_lower:
                return "Investment"
            return "Want"

        df["Expense_Type"] = df["Category"].apply(infer_type)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Processed {len(df)} transactions saved to: {output_path}")

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    raw_file = os.path.join(project_dir, "data", "expenses_dataset.csv")
    proc_file = os.path.join(project_dir, "data", "processed_expenses.csv")
    
    if os.path.exists(raw_file):
        clean_and_process_expenses(raw_file, proc_file)
    else:
        print("[WARNING] Raw file not found. Run data_generator.py first.")
