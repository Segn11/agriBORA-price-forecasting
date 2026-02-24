## =========================
## 1️⃣ Import libraries
## =========================
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

## =========================
## 2️⃣ Load data
## =========================
kamis_df = pd.read_csv('/content/kamis_maize_prices.csv', parse_dates=['Date'])
agri_df = pd.read_csv('/content/agriBORA_maize_prices.csv', parse_dates=['Date'])

# Filter White Maize
kamis_df = kamis_df[kamis_df["Commodity_Classification"].str.contains("Dry_White_Maize", na=False)].copy()
agri_df = agri_df[agri_df["Commodity_Classification"].str.contains("Dry_White_Maize", na=False)].copy()

# Normalize county names
def norm_county(s):
    return s.strip() if isinstance(s, str) else s

kamis_df["county_norm"] = kamis_df["County"].apply(norm_county)
agri_df["county_norm"] = agri_df["County"].apply(norm_county)

# Focus on target counties
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
kamis_df = kamis_df[kamis_df["county_norm"].isin(target_counties)].copy()
agri_df = agri_df[agri_df["county_norm"].isin(target_counties)].copy()

# Weekly aggregation
kamis_df["week_start"] = kamis_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)
agri_df["week_start"] = agri_df["Date"].dt.to_period("W").apply(lambda p: p.start_time)

kamis_df["kamis_price"] = pd.to_numeric(kamis_df["Wholesale"], errors='coerce')
agri_df["agr_price"] = pd.to_numeric(agri_df["WholeSale"], errors='coerce')

kamis_week = kamis_df.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].mean()
agri_week = agri_df.groupby(["county_norm", "week_start"], as_index=False)["agr_price"].mean()

## =========================
## 3️⃣ Build full weekly panel
## =========================
all_panels = []
for c in target_counties:
    sub = kamis_week[kamis_week["county_norm"] == c].copy()
    if sub.empty:
        continue
    full_weeks = pd.date_range(sub["week_start"].min(), sub["week_start"].max(), freq="W-MON")
    df = pd.DataFrame({"week_start": full_weeks})
    df["county_norm"] = c
    df = df.merge(sub[["week_start", "kamis_price"]], on="week_start", how="left")
    df["kamis_price"] = df["kamis_price"].ffill().bfill()
    df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1).mean()
    all_panels.append(df)

kamis_panel = pd.concat(all_panels, ignore_index=True)
panel = kamis_panel.merge(agri_week, on=["county_norm", "week_start"], how="left")
panel = panel.sort_values(["county_norm", "week_start"]).reset_index(drop=True)

# Create lag features
for lag in [1,2,3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["kamis_smooth"].shift(lag)

panel_train = panel.dropna(subset=["lag1","lag2","lag3","agr_price"]).reset_index(drop=True)
X = panel_train[["kamis_smooth","lag1","lag2","lag3","county_norm"]]
y = panel_train["agr_price"]

## =========================
## 4️⃣ Preprocess features
## =========================
numeric_features = ["kamis_smooth","lag1","lag2","lag3"]
categorical_features = ["county_norm"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_prepared = preprocess.fit_transform(X)

## =========================
## 5️⃣ Train LightGBM directly
## =========================
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    learning_rate=0.01,
    n_estimators=100000,
    random_state=42
)

lgb_model.fit(
    X_prepared, y,
    eval_set=[(X_prepared, y)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# Check training performance
y_pred_train = lgb_model.predict(X_prepared)
print("LightGBM Train MAE:", mean_absolute_error(y, y_pred_train))
print("LightGBM Train RMSE:", np.sqrt(mean_squared_error(y, y_pred_train)))

## =========================
## 6️⃣ Recursive Forecast for all target weeks
## =========================
target_start = pd.Timestamp("2025-11-17")
target_end   = pd.Timestamp("2026-01-10")

forecast_rows = []

for c in target_counties:
    hist = panel[panel["county_norm"] == c].sort_values("week_start").copy()
    if hist.empty:
        continue

    # last 3 smoothed KAMIS prices
    last3 = hist["kamis_smooth"].tail(3).values
    if len(last3) == 1:
        lag1 = lag2 = lag3 = last3[-1]
    elif len(last3) == 2:
        lag1 = last3[-1]
        lag2 = lag3 = last3[-2]
    else:
        lag1, lag2, lag3 = last3[-1], last3[-2], last3[-3]

    current_week = hist["week_start"].max()

    while current_week < target_end:
        next_week = current_week + timedelta(days=7)

        X_h = pd.DataFrame({
            "kamis_smooth": [lag1],
            "lag1": [lag1],
            "lag2": [lag2],
            "lag3": [lag3],
            "county_norm": [c]
        })
        X_h_prepared = preprocess.transform(X_h)
        pred_h = lgb_model.predict(X_h_prepared)[0]

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        # shift lags for next iteration
        lag3 = lag2
        lag2 = lag1
        lag1 = pred_h
        current_week = next_week

forecast_df = pd.DataFrame(forecast_rows)

# Keep only weeks in the exact target range
mask = (forecast_df["week_start"] >= target_start) & (forecast_df["week_start"] <= target_end)
forecast_df = forecast_df[mask].reset_index(drop=True)

## =========================
## 7️⃣ Build final submission
## =========================
forecast_df["week"] = forecast_df["week_start"].dt.isocalendar().week
forecast_df["ID"] = forecast_df["county"] + "_Week_" + forecast_df["week"].astype(str)
forecast_df["Target_RMSE"] = forecast_df["agr_pred"]
forecast_df["Target_MAE"] = forecast_df["agr_pred"]

submission = forecast_df[["ID","Target_RMSE","Target_MAE"]].reset_index(drop=True)
submission.head()
