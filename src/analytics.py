"""
Expense Tracker App - Analytics & Insights Engine
Calculates executive financial KPIs, monthly burn rates, 50/30/20 budget allocations,
payment mode preferences, statistical anomaly detection (IQR & Z-score), and budget risk alerts.
"""

import pandas as pd
import numpy as np


class ExpenseAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def get_summary_kpis(self, monthly_budget: float = 75000.0) -> dict:
        """
        Calculates high-level executive financial metrics.
        """
        total_spent = float(self.df["Amount"].sum())
        total_tx = int(len(self.df))
        
        # Unique active months
        unique_months = self.df["Year_Month"].nunique() or 1
        avg_monthly_spent = total_spent / unique_months
        
        # Active date range in days
        min_date = self.df["Date"].min()
        max_date = self.df["Date"].max()
        total_days = max((max_date - min_date).days + 1, 1)
        daily_burn_rate = total_spent / total_days
        
        # Top category
        cat_group = self.df.groupby("Category")["Amount"].sum()
        top_category = cat_group.idxmax() if not cat_group.empty else "N/A"
        top_category_amt = float(cat_group.max()) if not cat_group.empty else 0.0

        # Most frequent payment mode
        top_payment_mode = self.df["Payment_Mode"].mode()[0] if not self.df.empty else "N/A"

        # Budget utilization for latest month
        latest_month = self.df["Year_Month"].max()
        latest_month_spend = float(self.df[self.df["Year_Month"] == latest_month]["Amount"].sum())
        budget_utilization_pct = (latest_month_spend / monthly_budget) * 100 if monthly_budget > 0 else 0.0

        return {
            "Total_Expenditure": round(total_spent, 2),
            "Total_Transactions": total_tx,
            "Active_Months": unique_months,
            "Average_Monthly_Spend": round(avg_monthly_spent, 2),
            "Daily_Burn_Rate": round(daily_burn_rate, 2),
            "Top_Spending_Category": top_category,
            "Top_Category_Amount": round(top_category_amt, 2),
            "Top_Payment_Mode": top_payment_mode,
            "Latest_Month": latest_month,
            "Latest_Month_Spend": round(latest_month_spend, 2),
            "Monthly_Budget": monthly_budget,
            "Budget_Utilization_Pct": round(budget_utilization_pct, 1),
            "Is_Over_Budget": latest_month_spend > monthly_budget
        }

    def get_category_breakdown(self) -> pd.DataFrame:
        """
        Computes total amount, percentage share, count, and average spend per category.
        """
        cat_df = self.df.groupby("Category").agg(
            Total_Amount=("Amount", "sum"),
            Transaction_Count=("Amount", "count"),
            Average_Amount=("Amount", "mean")
        ).reset_index()

        total = cat_df["Total_Amount"].sum()
        cat_df["Percentage_Share"] = (cat_df["Total_Amount"] / total * 100).round(2)
        cat_df["Average_Amount"] = cat_df["Average_Amount"].round(2)
        cat_df = cat_df.sort_values(by="Total_Amount", ascending=False).reset_index(drop=True)
        return cat_df

    def get_monthly_trends(self) -> pd.DataFrame:
        """
        Aggregates monthly expenditure with Month-on-Month (MoM) growth velocity.
        """
        monthly = self.df.groupby("Year_Month").agg(
            Total_Spend=("Amount", "sum"),
            Transaction_Count=("Amount", "count"),
            Average_Ticket_Size=("Amount", "mean")
        ).reset_index()

        monthly["MoM_Growth_Pct"] = monthly["Total_Spend"].pct_change() * 100
        monthly["MoM_Growth_Pct"] = monthly["MoM_Growth_Pct"].fillna(0).round(2)
        monthly["Average_Ticket_Size"] = monthly["Average_Ticket_Size"].round(2)
        return monthly

    def get_need_vs_want_split(self) -> pd.DataFrame:
        """
        Analyzes spending distribution according to the 50/30/20 financial rule
        (Needs / Wants / Savings & Investments).
        """
        split_df = self.df.groupby("Expense_Type").agg(
            Total_Amount=("Amount", "sum"),
            Count=("Amount", "count")
        ).reset_index()

        total = split_df["Total_Amount"].sum()
        split_df["Actual_Pct"] = (split_df["Total_Amount"] / total * 100).round(2)
        
        # Benchmark mapping based on 50/30/20 rule
        benchmark_map = {"Need": 50.0, "Want": 30.0, "Investment": 20.0}
        split_df["Benchmark_Pct"] = split_df["Expense_Type"].map(benchmark_map).fillna(0.0)
        split_df["Variance"] = (split_df["Actual_Pct"] - split_df["Benchmark_Pct"]).round(2)

        return split_df

    def get_payment_mode_distribution(self) -> pd.DataFrame:
        """
        Aggregates volume and value by payment method.
        """
        pay_df = self.df.groupby("Payment_Mode").agg(
            Total_Spent=("Amount", "sum"),
            Count=("Amount", "count")
        ).reset_index()

        total = pay_df["Total_Spent"].sum()
        pay_df["Share_Pct"] = (pay_df["Total_Spent"] / total * 100).round(2)
        pay_df = pay_df.sort_values(by="Total_Spent", ascending=False).reset_index(drop=True)
        return pay_df

    def detect_anomalies_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Identifies statistical spending anomalies and luxury outliers using the
        Interquartile Range (IQR) method within each category.
        """
        anomalies_list = []
        for cat, group in self.df.groupby("Category"):
            q1 = group["Amount"].quantile(0.25)
            q3 = group["Amount"].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + (multiplier * iqr)
            
            outliers = group[group["Amount"] > upper_bound].copy()
            if not outliers.empty:
                outliers["Threshold"] = round(upper_bound, 2)
                outliers["Excess_Amount"] = round(outliers["Amount"] - upper_bound, 2)
                anomalies_list.append(outliers)

        if anomalies_list:
            anomalies_df = pd.concat(anomalies_list).sort_values(by="Amount", ascending=False).reset_index(drop=True)
            return anomalies_df
        return pd.DataFrame(columns=self.df.columns)

    def generate_executive_insights(self, monthly_budget: float = 75000.0) -> list:
        """
        Generates actionable financial business recommendations in plain English.
        """
        kpis = self.get_summary_kpis(monthly_budget=monthly_budget)
        cat_df = self.get_category_breakdown()
        split_df = self.get_need_vs_want_split()
        trends = self.get_monthly_trends()
        anomalies = self.detect_anomalies_iqr()

        insights = []

        # 1. Budget Health
        if kpis["Is_Over_Budget"]:
            over_val = kpis["Latest_Month_Spend"] - monthly_budget
            insights.append(
                f"🚨 **Budget Alert**: In {kpis['Latest_Month']}, spending reached ₹{kpis['Latest_Month_Spend']:,.2f}, "
                f"exceeding your target budget of ₹{monthly_budget:,.2f} by ₹{over_val:,.2f} ({kpis['Budget_Utilization_Pct']}% utilized)."
            )
        else:
            saved_val = monthly_budget - kpis["Latest_Month_Spend"]
            insights.append(
                f"✅ **Budget On Track**: In {kpis['Latest_Month']}, you spent ₹{kpis['Latest_Month_Spend']:,.2f} "
                f"({kpis['Budget_Utilization_Pct']}% of budget), leaving a healthy buffer of ₹{saved_val:,.2f}."
            )

        # 2. Dominant Cost Driver
        if not cat_df.empty:
            top_cat = cat_df.iloc[0]
            insights.append(
                f"📊 **Top Expense Driver**: **{top_cat['Category']}** constitutes **{top_cat['Percentage_Share']}%** "
                f"of total outgoing cash flow (₹{top_cat['Total_Amount']:,.2f} across {top_cat['Transaction_Count']} transactions)."
            )

        # 3. 50/30/20 Discipline
        want_row = split_df[split_df["Expense_Type"] == "Want"]
        if not want_row.empty:
            want_pct = want_row.iloc[0]["Actual_Pct"]
            if want_pct > 30.0:
                insights.append(
                    f"⚠️ **Discretionary Leakage**: Your 'Wants' (Dining, Shopping, Entertainment) account for "
                    f"**{want_pct}%** of spending, higher than the recommended 30% rule. Curtailing dining and impulsive shopping could save ₹{(want_pct - 30) * kpis['Total_Expenditure'] / 100:,.2f}."
                )
            else:
                insights.append(
                    f"🎯 **Disciplined Lifestyle**: Discretionary wants are well maintained at **{want_pct}%** (below the 30% ceiling)."
                )

        # 4. Outliers & Anomalies
        if len(anomalies) > 0:
            top_anomaly = anomalies.iloc[0]
            insights.append(
                f"🔍 **Statistical Outlier Detected**: Found {len(anomalies)} transactions significantly above standard spending habits. "
                f"Largest: ₹{top_anomaly['Amount']:,.2f} on {top_anomaly['Date']} for '{top_anomaly['Description']}'."
            )

        # 5. MoM Spending Momentum
        if len(trends) >= 2:
            last_mom = trends.iloc[-1]["MoM_Growth_Pct"]
            if last_mom > 10.0:
                insights.append(
                    f"📈 **Spike Alert**: Spending increased by **+{last_mom}%** compared to the previous month. Review discretionary transactions."
                )
            elif last_mom < -5.0:
                insights.append(
                    f"📉 **Cost Reduction**: Monthly spending decreased by **{abs(last_mom)}%**, reflecting improved financial restraint."
                )

        return insights
