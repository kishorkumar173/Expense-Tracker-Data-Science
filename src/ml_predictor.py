"""
Expense Tracker App - Machine Learning & Predictive Forecasting Module
Forecasts future monthly spending using regression modeling with time-series lags,
trend features, and risk probability estimation for budget overruns.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class SpendingPredictor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            
    def prepare_monthly_series(self) -> pd.DataFrame:
        """
        Transforms transaction-level records into an aggregated monthly time series.
        """
        monthly = self.df.groupby("Year_Month").agg(
            Total_Spend=("Amount", "sum"),
            Tx_Count=("Amount", "count"),
            Avg_Ticket=("Amount", "mean")
        ).reset_index()

        monthly["Month_Index"] = np.arange(len(monthly))
        return monthly

    def forecast_next_month_spend(self, target_budget: float = 75000.0) -> dict:
        """
        Trains a predictive model on historical monthly expenses to forecast
        spending for the forthcoming month, calculating prediction intervals and budget breach risk.
        """
        monthly = self.prepare_monthly_series()
        n_obs = len(monthly)

        if n_obs < 3:
            # Fallback if insufficient historical months
            recent_mean = monthly["Total_Spend"].mean() if n_obs > 0 else 50000.0
            return {
                "Forecasted_Spend": round(recent_mean, 2),
                "Lower_Bound_90": round(recent_mean * 0.9, 2),
                "Upper_Bound_90": round(recent_mean * 1.1, 2),
                "Historical_Months_Count": n_obs,
                "Model_Name": "Simple Moving Average Fallback",
                "Budget_Overrun_Risk_Pct": 50.0 if recent_mean > target_budget else 10.0,
                "Trend_Direction": "Neutral"
            }

        # Build feature matrix: Time trend index + Moving Average Lag
        X = monthly[["Month_Index"]].values
        y = monthly["Total_Spend"].values

        # Fit Ridge regression with light regularization to prevent overfitting on small samples
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # In-sample residuals for uncertainty estimation
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_error = np.std(residuals) if len(residuals) > 1 else np.std(y) * 0.1

        # Predict next month (Index = n_obs)
        next_month_idx = np.array([[n_obs]])
        pred_val = float(model.predict(next_month_idx)[0])
        pred_val = max(pred_val, 1000.0)  # Bound to non-negative

        # 90% prediction interval (~1.645 * std_error)
        margin = 1.645 * std_error
        lower_bound = max(pred_val - margin, 0.0)
        upper_bound = pred_val + margin

        # Trend direction
        slope = float(model.coef_[0])
        trend_direction = "Increasing" if slope > 500 else ("Decreasing" if slope < -500 else "Stable")

        # Overrun risk calculation via Z-score
        if std_error > 0:
            z_score = (target_budget - pred_val) / std_error
            # Approximate normal CDF
            from scipy.stats import norm
            prob_exceed = (1.0 - norm.cdf(z_score)) * 100
        else:
            prob_exceed = 95.0 if pred_val > target_budget else 5.0

        prob_exceed = min(max(prob_exceed, 1.0), 99.0)

        return {
            "Forecasted_Spend": round(pred_val, 2),
            "Lower_Bound_90": round(lower_bound, 2),
            "Upper_Bound_90": round(upper_bound, 2),
            "Historical_Months_Count": n_obs,
            "Model_Name": "Time-Trend Ridge Regression",
            "Monthly_Slope_INR": round(slope, 2),
            "Budget_Overrun_Risk_Pct": round(prob_exceed, 1),
            "Trend_Direction": trend_direction,
            "Target_Budget": target_budget
        }

    def forecast_category_breakdown(self) -> pd.DataFrame:
        """
        Projects next month's spending across each specific category based on
        recent category trajectory and weighted share.
        """
        # Group by category and month
        cat_monthly = self.df.pivot_table(
            index="Year_Month",
            columns="Category",
            values="Amount",
            aggfunc="sum",
            fill_value=0
        )
        
        # Exponentially weighted moving average to give higher weight to recent months
        ewm_forecast = cat_monthly.ewm(span=3).mean().iloc[-1]
        
        forecast_df = pd.DataFrame({
            "Category": ewm_forecast.index,
            "Projected_Spend": ewm_forecast.values.round(2)
        })
        
        total = forecast_df["Projected_Spend"].sum()
        forecast_df["Projected_Share_Pct"] = (forecast_df["Projected_Spend"] / total * 100).round(2)
        forecast_df = forecast_df.sort_values(by="Projected_Spend", ascending=False).reset_index(drop=True)
        return forecast_df
