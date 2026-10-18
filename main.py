"""
Expense Tracker App - Main Pipeline Runner (CLI)
Automates the full data science lifecycle:
1. Synthetic Data Generation
2. Schema Cleaning & Feature Engineering
3. Financial Analytics & Anomaly Detection
4. Machine Learning Spending Forecasting
5. Automated Publication-Quality Visualizations
6. Executive Report Generation
"""

import os
import sys
from datetime import datetime

# Path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Force UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_generator import generate_synthetic_expenses
from src.data_processor import clean_and_process_expenses
from src.analytics import ExpenseAnalytics
from src.ml_predictor import SpendingPredictor
from src.visualizer import generate_all_static_charts


def print_banner():
    banner = """
    ======================================================================
    📊 FINTRACK PRO: EXPENSE TRACKER DATA SCIENCE PIPELINE 
    ----------------------------------------------------------------------
    Advanced Analytics, Anomaly Detection & ML Spend Forecasting
    ======================================================================
    """
    print(banner)


def run_pipeline():
    print_banner()

    data_dir = os.path.join(BASE_DIR, "data")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    images_dir = os.path.join(BASE_DIR, "images")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    raw_data_path = os.path.join(data_dir, "expenses_dataset.csv")
    proc_data_path = os.path.join(data_dir, "processed_expenses.csv")

    # Step 1: Data Generation
    print("[STEP 1/6] Generating / Verifying Synthetic Financial Dataset...")
    if not os.path.exists(raw_data_path):
        generate_synthetic_expenses(output_path=raw_data_path)
    else:
        print(f" -> Existing dataset found at: {raw_data_path}")

    # Step 2: Data Cleaning & Feature Engineering
    print("\n[STEP 2/6] Executing Cleaning & Feature Engineering Pipeline...")
    df = clean_and_process_expenses(raw_data_path, output_path=proc_data_path)
    print(f" -> Total Cleaned Transactions: {len(df)}")
    print(f" -> Date Horizon: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    # Step 3: Financial Analytics & Business Intelligence
    print("\n[STEP 3/6] Computing Financial Analytics & Budget Benchmarks...")
    analytics = ExpenseAnalytics(df)
    target_budget = 75000.0
    kpis = analytics.get_summary_kpis(monthly_budget=target_budget)
    cat_df = analytics.get_category_breakdown()
    split_df = analytics.get_need_vs_want_split()
    anomalies_df = analytics.detect_anomalies_iqr()

    print("\n" + "-" * 55)
    print(f"💰 Total Outflow:         ₹{kpis['Total_Expenditure']:,.2f}")
    print(f"💳 Total Transactions:    {kpis['Total_Transactions']}")
    print(f"📅 Active Months:         {kpis['Active_Months']}")
    print(f"📉 Daily Burn Rate:       ₹{kpis['Daily_Burn_Rate']:,.2f} / day")
    print(f"🏆 Top Cost Driver:       {kpis['Top_Spending_Category']} (₹{kpis['Top_Category_Amount']:,.2f})")
    print(f"🎯 Latest Month Spend:    ₹{kpis['Latest_Month_Spend']:,.2f} vs Budget ₹{target_budget:,.2f}")
    print(f"⚠️ Budget Utilization:    {kpis['Budget_Utilization_Pct']}% ({'OVER BUDGET' if kpis['Is_Over_Budget'] else 'SAFE'})")
    print("-" * 55)

    # Step 4: Machine Learning Spend Forecasting
    print("\n[STEP 4/6] Training ML Spending Forecast Model...")
    predictor = SpendingPredictor(df)
    forecast = predictor.forecast_next_month_spend(target_budget=target_budget)
    
    print(f" -> Model: {forecast['Model_Name']}")
    print(f" -> Next Month Predicted Outflow: ₹{forecast['Forecasted_Spend']:,.2f}")
    print(f" -> 90% Confidence Interval:     ₹{forecast['Lower_Bound_90']:,.2f} - ₹{forecast['Upper_Bound_90']:,.2f}")
    print(f" -> Monthly Trajectory Trend:     {forecast['Trend_Direction']} ({forecast.get('Monthly_Slope_INR', 0):+,.2f} INR/mo)")
    print(f" -> Budget Overrun Probability:   {forecast['Budget_Overrun_Risk_Pct']}%")

    # Step 5: Visualizations
    print("\n[STEP 5/6] Generating Publication-Quality Visualizations...")
    # Save into outputs and copy to images
    generate_all_static_charts(df, analytics, outputs_dir)
    generate_all_static_charts(df, analytics, images_dir)
    print(f" -> Charts saved to: {outputs_dir} and {images_dir}")

    # Step 6: Executive Report Generation
    print("\n[STEP 6/6] Generating Executive Insights Report...")
    insights = analytics.generate_executive_insights(monthly_budget=target_budget)
    report_path = os.path.join(outputs_dir, "executive_summary.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("FINTRACK PRO - EXECUTIVE FINANCIAL AUDIT & INSIGHTS REPORT\n")
        f.write(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("======================================================================\n\n")
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write(f"- Total Expenditure: ₹{kpis['Total_Expenditure']:,.2f}\n")
        f.write(f"- Active Period: {kpis['Active_Months']} months ({kpis['Total_Transactions']} transactions)\n")
        f.write(f"- Average Monthly Spend: ₹{kpis['Average_Monthly_Spend']:,.2f}\n")
        f.write(f"- Target Monthly Budget: ₹{target_budget:,.2f}\n")
        f.write(f"- Projected Next Month Spend: ₹{forecast['Forecasted_Spend']:,.2f} (Risk: {forecast['Budget_Overrun_Risk_Pct']}%)\n\n")
        f.write("2. KEY BUSINESS & BEHAVIORAL INSIGHTS\n")
        for i, ins in enumerate(insights, 1):
            clean_ins = ins.replace('**', '').replace('🚨', '').replace('✅', '').replace('📊', '').replace('⚠️', '').replace('🎯', '').replace('🔍', '').replace('📈', '').replace('📉', '').strip()
            f.write(f"[{i}] {clean_ins}\n")
        f.write("\n3. STATISTICAL OUTLIER ANALYSIS (IQR METHOD)\n")
        f.write(f"- Detected {len(anomalies_df)} unusual transactions.\n")
        if not anomalies_df.empty:
            for _, row in anomalies_df.head(5).iterrows():
                f.write(f"   * {row['Date'].strftime('%Y-%m-%d')} | {row['Category']} | ₹{row['Amount']:,.2f} | {row['Description']}\n")

    print(f" -> Report generated: {report_path}")

    print("\n" + "=" * 70)
    print("🎉 FULL DATA SCIENCE PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)
    print("\nTo launch the interactive Web Dashboard, run:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    run_pipeline()
