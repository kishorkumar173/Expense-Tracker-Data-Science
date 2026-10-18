"""
Expense Tracker App - Interactive Streamlit Analytics & ML Dashboard
Industry-grade web application for personal finance intelligence, budget monitoring,
and predictive spending analytics.
"""

import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add current project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_generator import generate_synthetic_expenses, CATEGORIES_MAP, PAYMENT_MODES
from src.data_processor import clean_and_process_expenses
from src.analytics import ExpenseAnalytics
from src.ml_predictor import SpendingPredictor

# Streamlit Page Config
st.set_page_config(
    page_title="Expense Tracker | Data Science Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


DATA_FILE = os.path.join(BASE_DIR, "data", "expenses_dataset.csv")


@st.cache_data(show_spinner=False)
def load_and_prep_data():
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        generate_synthetic_expenses(output_path=DATA_FILE)
    df = clean_and_process_expenses(DATA_FILE)
    return df


# Load data
df_master = load_and_prep_data()

# ==========================================
# SIDEBAR CONTROLS & NEW TRANSACTION FORM
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=65)
st.sidebar.title("FinTrack Pro")
st.sidebar.caption("Data-Driven Personal Finance Intelligence")
st.sidebar.markdown("---")

# Monthly Budget Target
st.sidebar.subheader("🎯 Monthly Budget Target")
monthly_budget = st.sidebar.number_input(
    "Set Monthly Limit (₹)",
    min_value=20000,
    max_value=300000,
    value=75000,
    step=5000
)

st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("🔍 Filter Transactions")
min_d = df_master["Date"].min().date()
max_d = df_master["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Window",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

all_categories = sorted(df_master["Category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Filter Categories",
    options=all_categories,
    default=all_categories
)

all_payment_modes = sorted(df_master["Payment_Mode"].unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment Mode",
    options=all_payment_modes,
    default=all_payment_modes
)

# Apply filters
if len(date_range) == 2:
    start_filter, end_filter = date_range
    mask = (
        (df_master["Date"].dt.date >= start_filter) &
        (df_master["Date"].dt.date <= end_filter) &
        (df_master["Category"].isin(selected_categories)) &
        (df_master["Payment_Mode"].isin(selected_payments))
    )
    filtered_df = df_master[mask].copy()
else:
    filtered_df = df_master.copy()

st.sidebar.markdown("---")

# Form: Add New Real-time Expense
with st.sidebar.expander("➕ Log New Expense", expanded=False):
    with st.form("new_expense_form", clear_on_submit=True):
        new_date = st.date_input("Transaction Date", value=date.today())
        new_cat = st.selectbox("Category", list(CATEGORIES_MAP.keys()))
        new_subcat = st.selectbox("Sub-Category", CATEGORIES_MAP[new_cat]["subcategories"])
        new_amt = st.number_input("Amount (₹)", min_value=1.0, value=500.0, step=50.0)
        new_pay = st.selectbox("Payment Mode", PAYMENT_MODES)
        new_desc = st.text_input("Description", value="Manual Expense Entry")
        new_type = CATEGORIES_MAP[new_cat]["type"]
        
        submitted = st.form_submit_button("Save Transaction")
        if submitted:
            new_record = {
                "Transaction_ID": f"TXN{len(df_master) + 10001}",
                "Date": new_date.strftime("%Y-%m-%d"),
                "Category": new_cat,
                "Sub_Category": new_subcat,
                "Amount": new_amt,
                "Payment_Mode": new_pay,
                "Description": new_desc,
                "Expense_Type": new_type,
                "Location": "Offline / Retail Store"
            }
            # Append to CSV
            temp_df = pd.DataFrame([new_record])
            temp_df.to_csv(DATA_FILE, mode="a", header=not os.path.exists(DATA_FILE), index=False)
            st.cache_data.clear()
            st.success("Transaction recorded! Refreshing dashboard...")
            st.rerun()

# Button to regenerate demo synthetic dataset
if st.sidebar.button("🔄 Reset / Regenerate Sample Data"):
    generate_synthetic_expenses(output_path=DATA_FILE)
    st.cache_data.clear()
    st.success("Fresh 12-month synthetic financial history generated!")
    st.rerun()

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.markdown('<div class="main-title">💳 Personal Expense & Budget Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Data Science Application | Financial Analytics, Anomaly Detection & ML Forecasting</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No transactions match the selected filters. Please adjust date range or categories.")
    st.stop()

# Instantiate Analytics Engine
analytics = ExpenseAnalytics(filtered_df)
kpis = analytics.get_summary_kpis(monthly_budget=monthly_budget)

# Budget Alert Banner
if kpis["Is_Over_Budget"]:
    st.error(
        f"🚨 **Monthly Overspending Alert**: In **{kpis['Latest_Month']}**, total expenditure reached "
        f"**₹{kpis['Latest_Month_Spend']:,.2f}**, which exceeds your monthly limit of **₹{monthly_budget:,.2f}** "
        f"by **₹{kpis['Latest_Month_Spend'] - monthly_budget:,.2f}** ({kpis['Budget_Utilization_Pct']}% utilized)."
    )
else:
    st.success(
        f"✅ **Budget Health Good**: In **{kpis['Latest_Month']}**, you spent "
        f"**₹{kpis['Latest_Month_Spend']:,.2f}** of your **₹{monthly_budget:,.2f}** budget "
        f"({kpis['Budget_Utilization_Pct']}% utilized). You have **₹{monthly_budget - kpis['Latest_Month_Spend']:,.2f}** remaining."
    )

# Executive KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Outflow", f"₹{kpis['Total_Expenditure']:,.0f}", f"{kpis['Total_Transactions']} Txns")
col2.metric("Monthly Avg Spend", f"₹{kpis['Average_Monthly_Spend']:,.0f}", f"{kpis['Active_Months']} Active Months")
col3.metric("Daily Burn Rate", f"₹{kpis['Daily_Burn_Rate']:,.0f}/day")
col4.metric("Top Cost Driver", kpis["Top_Spending_Category"][:16], f"₹{kpis['Top_Category_Amount']:,.0f}")
col5.metric("Budget Utilization", f"{kpis['Budget_Utilization_Pct']}%", delta=f"{round(100 - kpis['Budget_Utilization_Pct'], 1)}% Buffer", delta_color="inverse" if kpis['Is_Over_Budget'] else "normal")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Spending Distribution",
    "📈 Monthly Momentum & Trends",
    "🤖 Predictive ML & Outliers",
    "📑 Transaction Explorer & Reports"
])

# ==========================================
# TAB 1: SPENDING DISTRIBUTION
# ==========================================
with tab1:
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.subheader("Category & Sub-Category Composition")
        sunburst_fig = px.sunburst(
            filtered_df,
            path=["Category", "Sub_Category"],
            values="Amount",
            color="Amount",
            color_continuous_scale="Viridis",
            title="Interactive Expense Drilldown (Click rings to zoom)"
        )
        sunburst_fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), height=460)
        st.plotly_chart(sunburst_fig, use_container_width=True)

    with c2:
        st.subheader("50/30/20 Financial Health Split")
        split_df = analytics.get_need_vs_want_split()
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Actual Spend %', x=split_df['Expense_Type'], y=split_df['Actual_Pct'], marker_color='#4F46E5'),
            go.Bar(name='50/30/20 Rule Target %', x=split_df['Expense_Type'], y=split_df['Benchmark_Pct'], marker_color='#94A3B8')
        ])
        fig_bar.update_layout(
            barmode='group',
            yaxis_title="Percentage (%)",
            height=340,
            margin=dict(t=20, l=10, r=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption("Standard Rule: **50% Needs** (Rent, Groceries, Utilities), **30% Wants** (Dining, Shopping), **20% Savings/Investments**.")

    st.markdown("---")
    
    # Category table & Payment Channel
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Category Expenditure Ranking")
        cat_df = analytics.get_category_breakdown()
        st.dataframe(
            cat_df.style.format({
                "Total_Amount": "₹{:,.2f}",
                "Average_Amount": "₹{:,.2f}",
                "Percentage_Share": "{:.1f}%"
            }),
            use_container_width=True,
            height=300
        )

    with c4:
        st.subheader("Payment Channel Preference")
        pay_df = analytics.get_payment_mode_distribution()
        fig_pay = px.pie(
            pay_df,
            names="Payment_Mode",
            values="Total_Spent",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_pay.update_layout(margin=dict(t=20, l=0, r=0, b=0), height=300)
        st.plotly_chart(fig_pay, use_container_width=True)


# ==========================================
# TAB 2: MONTHLY MOMENTUM & TRENDS
# ==========================================
with tab2:
    st.subheader("Monthly Outflow vs Budget Threshold")
    monthly_df = analytics.get_monthly_trends()

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(
        x=monthly_df["Year_Month"],
        y=monthly_df["Total_Spend"],
        name="Actual Spend (₹)",
        marker_color="#6366F1",
        text=[f"₹{int(v/1000)}k" for v in monthly_df["Total_Spend"]],
        textposition="outside"
    ))
    trend_fig.add_trace(go.Scatter(
        x=monthly_df["Year_Month"],
        y=monthly_df["Total_Spend"],
        name="Momentum Trend",
        line=dict(color="#F59E0B", width=3)
    ))
    trend_fig.add_hline(
        y=monthly_budget,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Monthly Budget (₹{monthly_budget:,.0f})",
        annotation_position="top left"
    )
    trend_fig.update_layout(
        yaxis_title="Total Spend (₹)",
        xaxis_title="Month",
        height=420,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.subheader("Weekday vs Weekend Spending Velocity")
        dow_df = filtered_df.groupby("Weekend_Label").agg(
            Total=("Amount", "sum"),
            Avg_Per_Tx=("Amount", "mean"),
            Count=("Amount", "count")
        ).reset_index()
        fig_dow = px.bar(
            dow_df,
            x="Weekend_Label",
            y="Avg_Per_Tx",
            color="Weekend_Label",
            text_auto=".2f",
            title="Average Ticket Size: Weekdays vs Weekends",
            color_discrete_map={"Weekend": "#EC4899", "Weekday": "#3B82F6"}
        )
        fig_dow.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    with c_t2:
        st.subheader("Month-on-Month (MoM) Growth Analysis")
        st.dataframe(
            monthly_df.style.format({
                "Total_Spend": "₹{:,.2f}",
                "Average_Ticket_Size": "₹{:,.2f}",
                "MoM_Growth_Pct": "{:+.1f}%"
            }),
            use_container_width=True,
            height=320
        )


# ==========================================
# TAB 3: PREDICTIVE ML & ANOMALY DETECTION
# ==========================================
with tab3:
    st.subheader("🤖 Machine Learning Spending Forecast (Next Month)")
    predictor = SpendingPredictor(filtered_df)
    forecast_results = predictor.forecast_next_month_spend(target_budget=monthly_budget)
    cat_projections = predictor.forecast_category_breakdown()

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Predicted Spend (Next Month)", f"₹{forecast_results['Forecasted_Spend']:,.0f}", f"Model: {forecast_results['Model_Name']}")
    p_col2.metric("90% Confidence Interval", f"₹{forecast_results['Lower_Bound_90']:,.0f} - ₹{forecast_results['Upper_Bound_90']:,.0f}")
    p_col3.metric("Spending Trajectory", forecast_results["Trend_Direction"], f"₹{forecast_results.get('Monthly_Slope_INR', 0):+,.0f} / mo")
    
    # Overrun risk metric
    risk_val = forecast_results["Budget_Overrun_Risk_Pct"]
    p_col4.metric(
        "Budget Breach Probability",
        f"{risk_val}%",
        delta="High Risk" if risk_val > 50 else "Safe",
        delta_color="inverse"
    )

    st.markdown("---")

    col_proj, col_outliers = st.columns([1, 1.2])

    with col_proj:
        st.subheader("Projected Category Expenses")
        fig_proj = px.bar(
            cat_projections,
            x="Projected_Spend",
            y="Category",
            orientation="h",
            color="Projected_Spend",
            color_continuous_scale="Blues",
            text=[f"₹{int(v/1000)}k" for v in cat_projections["Projected_Spend"]]
        )
        fig_proj.update_layout(height=380, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_outliers:
        st.subheader("🔍 Statistical Anomaly Detection (IQR Method)")
        anomalies_df = analytics.detect_anomalies_iqr()
        
        if not anomalies_df.empty:
            st.info(f"Identified **{len(anomalies_df)}** transactions exceeding the statistical upper bound (1.5x IQR) for their respective category.")
            st.dataframe(
                anomalies_df[["Date", "Category", "Amount", "Threshold", "Description"]].style.format({
                    "Amount": "₹{:,.2f}",
                    "Threshold": "₹{:,.2f}"
                }),
                use_container_width=True,
                height=320
            )
        else:
            st.success("No significant anomalous transactions detected! Spending remains within standard bounds.")


# ==========================================
# TAB 4: TRANSACTION EXPLORER & REPORTS
# ==========================================
with tab4:
    st.subheader("📑 Raw Transaction Ledger")
    
    search_query = st.text_input("Quick Search (by Description, Subcategory or Category):", "")
    view_df = filtered_df.copy()
    if search_query:
        view_df = view_df[
            view_df["Description"].str.contains(search_query, case=False, na=False) |
            view_df["Sub_Category"].str.contains(search_query, case=False, na=False) |
            view_df["Category"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(
        view_df[["Transaction_ID", "Date", "Category", "Sub_Category", "Amount", "Payment_Mode", "Expense_Type", "Description"]].sort_values(by="Date", ascending=False),
        use_container_width=True,
        height=380
    )

    # Downloads & Reports
    st.markdown("---")
    st.subheader("📥 Export Financial Intelligence Reports")
    
    rep_col1, rep_col2 = st.columns(2)
    
    with rep_col1:
        csv_data = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Transactions (CSV)",
            data=csv_data,
            file_name=f"Expenses_Export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with rep_col2:
        # Generate automated Executive Brief
        insights_list = analytics.generate_executive_insights(monthly_budget=monthly_budget)
        brief_text = f"# FinTrack Pro - Executive Spending Intelligence Report\n"
        brief_text += f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        brief_text += f"## Financial Summary\n"
        brief_text += f"- Total Expenditure: ₹{kpis['Total_Expenditure']:,.2f}\n"
        brief_text += f"- Total Transactions: {kpis['Total_Transactions']}\n"
        brief_text += f"- Average Monthly Spend: ₹{kpis['Average_Monthly_Spend']:,.2f}\n"
        brief_text += f"- Daily Burn Rate: ₹{kpis['Daily_Burn_Rate']:,.2f}/day\n"
        brief_text += f"- Target Monthly Budget: ₹{monthly_budget:,.2f}\n"
        brief_text += f"- Forecasted Next Month Spend: ₹{forecast_results['Forecasted_Spend']:,.2f}\n\n"
        brief_text += f"## Key Observations & Recommendations\n"
        for ins in insights_list:
            brief_text += f"- {ins.replace('**', '').replace('🚨', '').replace('✅', '').replace('📊', '').replace('⚠️', '').replace('🎯', '').replace('🔍', '').replace('📈', '').replace('📉', '').strip()}\n"

        st.download_button(
            label="📄 Download Executive Intelligence Brief (TXT)",
            data=brief_text,
            file_name=f"Expense_Audit_Report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
