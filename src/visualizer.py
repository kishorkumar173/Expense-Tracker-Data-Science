"""
Expense Tracker App - Visualization Generator Module
Generates production-grade static graphics using Matplotlib and Seaborn.
Saves high-resolution PNGs directly to outputs/ and images/ for GitHub proof and reporting.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set standard clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11


def plot_category_donut(cat_df: pd.DataFrame, output_path: str):
    """
    Renders a clean donut chart of category-wise expenditure share.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))
    
    top_categories = cat_df.head(7).copy()
    if len(cat_df) > 7:
        others_amt = cat_df.iloc[7:]["Total_Amount"].sum()
        others_row = pd.DataFrame([{
            "Category": "Others",
            "Total_Amount": others_amt,
            "Percentage_Share": round(others_amt / cat_df["Total_Amount"].sum() * 100, 2)
        }])
        top_categories = pd.concat([top_categories, others_row], ignore_index=True)

    colors = sns.color_palette("Set2", len(top_categories))
    wedges, texts, autotexts = ax.pie(
        top_categories["Total_Amount"],
        labels=top_categories["Category"],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_weight("bold")

    ax.set_title("Expenditure Breakdown by Category", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED CHART] Category Donut -> {output_path}")


def plot_monthly_trend(monthly_df: pd.DataFrame, output_path: str, target_budget: float = 75000.0):
    """
    Renders a bar & line combo chart displaying monthly expense momentum and budget threshold.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    x_labels = monthly_df["Year_Month"].tolist()
    y_values = monthly_df["Total_Spend"].tolist()

    bars = ax.bar(x_labels, y_values, color="#4F46E5", alpha=0.85, label="Monthly Spend (INR)")
    ax.axhline(target_budget, color="#EF4444", linestyle="--", linewidth=2, label=f"Budget Cap (₹{target_budget:,.0f})")

    # Add trend line
    ax.plot(x_labels, y_values, color="#F59E0B", marker="o", linewidth=2.5, label="Trendline")

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"₹{int(height/1000)}k",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    ax.set_title("Monthly Spending Trajectory vs Target Budget", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spend (₹)")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED CHART] Monthly Trend -> {output_path}")


def plot_need_vs_want_split(split_df: pd.DataFrame, output_path: str):
    """
    Renders grouped bar chart comparing actual allocation against the 50/30/20 guideline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    types = split_df["Expense_Type"].tolist()
    actual = split_df["Actual_Pct"].tolist()
    benchmark = split_df["Benchmark_Pct"].tolist()

    x = np.arange(len(types))
    width = 0.35

    ax.bar(x - width/2, actual, width, label="Actual Allocation (%)", color="#06B6D4")
    ax.bar(x + width/2, benchmark, width, label="50/30/20 Target (%)", color="#94A3B8")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Financial Health: Needs vs Wants vs Investments", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()

    for i in x:
        ax.annotate(f"{actual[i]:.1f}%", xy=(i - width/2, actual[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, weight="bold")
        ax.annotate(f"{benchmark[i]:.0f}%", xy=(i + width/2, benchmark[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color="#475569")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED CHART] Need vs Want -> {output_path}")


def plot_payment_modes(pay_df: pd.DataFrame, output_path: str):
    """
    Renders horizontal bar chart for payment mode distribution.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.barplot(data=pay_df, x="Share_Pct", y="Payment_Mode", hue="Payment_Mode", palette="viridis", legend=False, ax=ax)
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1f}%", xy=(width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, weight="bold")

    ax.set_title("Transaction Volume Share by Payment Channel", pad=15)
    ax.set_xlabel("Expenditure Share (%)")
    ax.set_ylabel("Payment Channel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED CHART] Payment Modes -> {output_path}")


def plot_category_boxplots(df: pd.DataFrame, output_path: str):
    """
    Renders boxplot of spending amounts across major categories to expose outliers.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Exclude Rent for clearer scale on variable expenses
    var_df = df[df["Category"] != "Rent & Housing"].copy()

    sns.boxplot(
        data=var_df,
        y="Category",
        x="Amount",
        hue="Category",
        palette="pastel",
        legend=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "6"},
        ax=ax
    )

    ax.set_title("Spending Spread & Outlier Detection by Category (Excl. Rent)", pad=15)
    ax.set_xlabel("Transaction Amount (₹)")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED CHART] Boxplot Outliers -> {output_path}")


def generate_all_static_charts(df: pd.DataFrame, analytics_obj, output_dir: str):
    """
    Orchestrates the generation of the complete chart portfolio for presentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    cat_df = analytics_obj.get_category_breakdown()
    monthly_df = analytics_obj.get_monthly_trends()
    split_df = analytics_obj.get_need_vs_want_split()
    pay_df = analytics_obj.get_payment_mode_distribution()

    plot_category_donut(cat_df, os.path.join(output_dir, "category_spending_donut.png"))
    plot_monthly_trend(monthly_df, os.path.join(output_dir, "monthly_spending_trend.png"))
    plot_need_vs_want_split(split_df, os.path.join(output_dir, "need_vs_want_split.png"))
    plot_payment_modes(pay_df, os.path.join(output_dir, "payment_mode_distribution.png"))
    plot_category_boxplots(df, os.path.join(output_dir, "category_outliers_boxplot.png"))
