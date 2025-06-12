# STEP 1: IMPORT NECESSARY STUFF

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

