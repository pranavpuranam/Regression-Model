# STEP 1: IMPORT NECESSARY STUFF

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# STEP 2: READ IN RAW DATA

msft = pd.read_csv("msft_data.csv", parse_dates=["Date"], index_col="Date")

# STEP 3: DEFINE THE EVENT OCCURENCE DATE

event_date = pd.to_datetime("2022-01-18")

# STEP 4: DEFINE START AND END DATES FOR RUNNING OUR REGRESSION MODEL

est_end   = event_date - pd.Timedelta(days=6) # 6 days before the announcement went out for a margin of safety
est_start = est_end - pd.Timedelta(days=365)  # allows for missing weekends and holidays

est_data = msft.loc[est_start:est_end].copy()

# STEP 5: RUN AND OUTPUT A REGRESSION MODEL

X = sm.add_constant(est_data[[
    "Market Excess Return (Mkt-RF)",
    "Size Factor (SMB)",
    "Value Factor (HML)"
]]) # These are the key inputs into the Fama-French model

y = est_data["MSFT Excess Return"] # This is what we're trying to get out

model = sm.OLS(y, X).fit()

print(model.summary())

# STEP 6: DEFINE THE WINDOW OF THE EVENT ITSELF

k_back = 0 # How many days before the announcement are we looking at, i.e. did news leak early
k_front = 9 # How many days after the announcement are we looking at, i.e. how long did event impact returns

evt_start = event_date - pd.Timedelta(days = k_back)
evt_end = event_date + pd.Timedelta(days = k_front)

evt_data = msft.loc[evt_start:evt_end].copy()

# STEP 7: MAKE PREDICTIONS USING REGRESSION MODEL

X_evt = sm.add_constant(evt_data[[
    "Market Excess Return (Mkt-RF)",
    "Size Factor (SMB)",
    "Value Factor (HML)"
]])

evt_data["Expected"] = model.predict(X_evt)

# STEP 8: CALCULATE ABNORMAL RETURNS AND OUTPUT

evt_data["Abnormal"] = evt_data["MSFT Excess Return"] - evt_data["Expected"]
evt_data["CAR"] = evt_data["Abnormal"].cumsum()

print(evt_data[["MSFT Excess Return", "Expected", "Abnormal", "CAR"]])
print(f"\nCumulative Abnormal Return (CAR) from {evt_start.date()} to {evt_end.date()}: {evt_data['Abnormal'].sum():.4%}")

# STEP 9: CONDUCT A T TEST

X_est = sm.add_constant(est_data[[
    "Market Excess Return (Mkt-RF)",
    "Size Factor (SMB)",
    "Value Factor (HML)"
]])
est_data["Expected"] = model.predict(X_est)
est_data["Residual"] = est_data["MSFT Excess Return"] - est_data["Expected"]

sigma_AR = est_data["Residual"].std()

CAR = evt_data["Abnormal"].sum()
n = len(evt_data)

t_stat = CAR / (sigma_AR * np.sqrt(n))

print(f"\nT-statistic for CAR: {t_stat:.2f}")

df = len(est_data) - 4  # 3 betas + intercept

p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
print(f"P-value: {p_val:.4f}")

