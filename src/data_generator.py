"""
Expense Tracker App - Data Generator Module
Generates realistic, industry-standard synthetic personal finance transaction records.
Includes seasonal trends, recurring monthly commitments, weekend spending spikes,
and realistic payment channel distributions.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for reproducibility
np.random.seed(42)
random.seed(42)

CATEGORIES_MAP = {
    "Rent & Housing": {
        "subcategories": ["House Rent", "Maintenance", "Property Tax"],
        "min_amt": 12000,
        "max_amt": 25000,
        "type": "Need",
        "recurring": True,
        "day_of_month": 1,
    },
    "Groceries & Supermarket": {
        "subcategories": ["Vegetables & Fruits", "Dairy & Essentials", "Packaged Food", "Beverages"],
        "min_amt": 250,
        "max_amt": 3500,
        "type": "Need",
        "recurring": False,
    },
    "Utilities & Bills": {
        "subcategories": ["Electricity Bill", "Broadband Wi-Fi", "Mobile Recharge", "Water Bill", "Piped Gas"],
        "min_amt": 300,
        "max_amt": 2200,
        "type": "Need",
        "recurring": True,
        "day_of_month": 5,
    },
    "Food & Dining Out": {
        "subcategories": ["Restaurant Dining", "Zomato / Swiggy", "Cafe / Coffee", "Fast Food"],
        "min_amt": 150,
        "max_amt": 2500,
        "type": "Want",
        "recurring": False,
    },
    "Transportation": {
        "subcategories": ["Fuel (Petrol/Diesel)", "Uber / Ola", "Metro / Bus", "Auto Rickshaw"],
        "min_amt": 80,
        "max_amt": 2000,
        "type": "Need",
        "recurring": False,
    },
    "Shopping & Lifestyle": {
        "subcategories": ["Clothing & Apparel", "Footwear", "Home Decor", "Personal Care"],
        "min_amt": 500,
        "max_amt": 8500,
        "type": "Want",
        "recurring": False,
    },
    "Entertainment & OTT": {
        "subcategories": ["Netflix / Prime", "Movie Tickets", "Gaming & Console", "Concerts / Events"],
        "min_amt": 199,
        "max_amt": 1800,
        "type": "Want",
        "recurring": False,
    },
    "Healthcare & Medical": {
        "subcategories": ["Doctor Consultation", "Pharmacy Medicines", "Health Diagnostic Test"],
        "min_amt": 200,
        "max_amt": 4500,
        "type": "Need",
        "recurring": False,
    },
    "Electronics & Gadgets": {
        "subcategories": ["Smartphone Accessories", "Laptop Upgrades", "Smartwatch / Headphones"],
        "min_amt": 1500,
        "max_amt": 35000,
        "type": "Want",
        "recurring": False,
    },
    "Investments & Savings": {
        "subcategories": ["Mutual Fund SIP", "PPF Deposit", "Gold Sovereign Bond", "Recurring Deposit"],
        "min_amt": 2000,
        "max_amt": 15000,
        "type": "Investment",
        "recurring": True,
        "day_of_month": 10,
    }
}

PAYMENT_MODES = ["UPI (PhonePe/GPay)", "Credit Card", "Debit Card", "Net Banking", "Cash"]
PAYMENT_WEIGHTS = [0.45, 0.28, 0.12, 0.10, 0.05]

LOCATIONS = ["Online", "Offline / Retail Store"]


def generate_synthetic_expenses(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    num_random_records: int = 1100,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset of financial expenses across a date range.
    Includes recurring monthly obligations, realistic discretionary spending, and outliers.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    records = []
    tx_counter = 10001

    # 1. Generate Recurring Monthly Expenses (Rent, SIP, Bills)
    curr_date = start
    while curr_date <= end:
        # First day of each month: Rent
        rent_date = curr_date.replace(day=1)
        if start <= rent_date <= end:
            records.append({
                "Transaction_ID": f"TXN{tx_counter}",
                "Date": rent_date.strftime("%Y-%m-%d"),
                "Category": "Rent & Housing",
                "Sub_Category": "House Rent",
                "Amount": float(random.choice([16000, 18000, 20000])),
                "Payment_Mode": "Net Banking",
                "Description": "Monthly Apartment Rental",
                "Expense_Type": "Need",
                "Location": "Online"
            })
            tx_counter += 1

        # 5th of each month: Utilities
        util_date = curr_date.replace(day=5)
        if start <= util_date <= end:
            records.append({
                "Transaction_ID": f"TXN{tx_counter}",
                "Date": util_date.strftime("%Y-%m-%d"),
                "Category": "Utilities & Bills",
                "Sub_Category": "Electricity Bill",
                "Amount": round(random.uniform(900, 2100), 2),
                "Payment_Mode": "UPI (PhonePe/GPay)",
                "Description": "State Electricity Board Payment",
                "Expense_Type": "Need",
                "Location": "Online"
            })
            tx_counter += 1

            records.append({
                "Transaction_ID": f"TXN{tx_counter}",
                "Date": util_date.strftime("%Y-%m-%d"),
                "Category": "Utilities & Bills",
                "Sub_Category": "Broadband Wi-Fi",
                "Amount": 999.00,
                "Payment_Mode": "Credit Card",
                "Description": "Fiber Broadband Monthly Subscription",
                "Expense_Type": "Need",
                "Location": "Online"
            })
            tx_counter += 1

        # 10th of each month: SIP Investment
        sip_date = curr_date.replace(day=10)
        if start <= sip_date <= end:
            records.append({
                "Transaction_ID": f"TXN{tx_counter}",
                "Date": sip_date.strftime("%Y-%m-%d"),
                "Category": "Investments & Savings",
                "Sub_Category": "Mutual Fund SIP",
                "Amount": float(random.choice([5000, 8000, 10000])),
                "Payment_Mode": "Net Banking",
                "Description": "Auto-debit Index Mutual Fund SIP",
                "Expense_Type": "Investment",
                "Location": "Online"
            })
            tx_counter += 1

        # Advance to next month
        # Compute next month safely
        if curr_date.month == 12:
            curr_date = curr_date.replace(year=curr_date.year + 1, month=1, day=1)
        else:
            curr_date = curr_date.replace(month=curr_date.month + 1, day=1)

    # 2. Generate Daily Variable & Discretionary Expenses
    variable_categories = [
        "Groceries & Supermarket",
        "Food & Dining Out",
        "Transportation",
        "Shopping & Lifestyle",
        "Entertainment & OTT",
        "Healthcare & Medical",
        "Electronics & Gadgets"
    ]
    var_weights = [0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]

    for _ in range(num_random_records):
        random_day_offset = random.randint(0, total_days - 1)
        tx_date = start + timedelta(days=random_day_offset)
        is_weekend = tx_date.weekday() >= 5

        # Weekend bias towards dining out and shopping
        if is_weekend and random.random() < 0.45:
            cat = random.choice(["Food & Dining Out", "Shopping & Lifestyle", "Entertainment & OTT"])
        else:
            cat = random.choices(variable_categories, weights=var_weights, k=1)[0]

        cat_info = CATEGORIES_MAP[cat]
        subcat = random.choice(cat_info["subcategories"])
        
        # Log-normal distribution for spending amounts to create realistic long-tail
        base_min = cat_info["min_amt"]
        base_max = cat_info["max_amt"]
        
        # Add slight inflation / weekend spike
        multiplier = 1.25 if is_weekend and cat in ["Food & Dining Out", "Entertainment & OTT"] else 1.0
        amount = round(random.uniform(base_min, base_max) * multiplier, 2)

        # Inject realistic occasional outlier (e.g., Festival Diwali sale in Oct/Nov or holiday travel)
        if tx_date.month in [10, 11] and random.random() < 0.03:
            amount = round(amount * 2.5, 2)

        pay_mode = random.choices(PAYMENT_MODES, weights=PAYMENT_WEIGHTS, k=1)[0]
        location = random.choice(LOCATIONS)
        
        # Contextual description
        desc = f"{subcat} payment via {pay_mode}"

        records.append({
            "Transaction_ID": f"TXN{tx_counter}",
            "Date": tx_date.strftime("%Y-%m-%d"),
            "Category": cat,
            "Sub_Category": subcat,
            "Amount": amount,
            "Payment_Mode": pay_mode,
            "Description": desc,
            "Expense_Type": cat_info["type"],
            "Location": location
        })
        tx_counter += 1

    df = pd.DataFrame(records)
    # Sort chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Generated {len(df)} synthetic expense records saved to: {output_path}")

    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    out_file = os.path.join(project_dir, "data", "expenses_dataset.csv")
    generate_synthetic_expenses(output_path=out_file)
