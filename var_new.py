# #Step 1: Install libraries

# pip install pandas numpy scikit-bio statsmodels


# ---

# Step 2: Load your data

# import pandas as pd

# df = pd.DataFrame({
#     "Year": [2015, 2016, 2017, 2018, 2019],
#     "Chemical_only": [40, 42, 45, 44, 46],
#     "Organic_only": [20, 18, 17, 16, 15],
#     "Both": [30, 28, 27, 28, 27],
#     "None": [10, 12, 11, 12, 12]
# })

# df = df.set_index("Year")


# ---

# Step 3: Convert to proportions

# import numpy as np

# X = df / 100.0


# ---

# Step 4: ILR transformation (VERY IMPORTANT)

# from skbio.stats.composition import closure, ilr

# # ensure compositional closure (sums to 1)
# X_closed = closure(X.values)

# # ILR transformation
# X_ilr = ilr(X_closed)

# ilr_df = pd.DataFrame(X_ilr, index=df.index)
# print(ilr_df.head())


# ---

# Step 5: Fit VAR model

# from statsmodels.tsa.api import VAR

# model = VAR(ilr_df)

# # choose lag order
# lag_order = model.select_order(maxlags=3)
# print(lag_order.summary())

# # fit model (example lag = 1)
# var_model = model.fit(1)

# print(var_model.summary())


# ---

# Step 6: Forecast future values

# forecast_ilr = var_model.forecast(ilr_df.values, steps=3)

# forecast_ilr_df = pd.DataFrame(forecast_ilr)
# print(forecast_ilr_df)


# ---

# Step 7: Convert back to percentages

# from skbio.stats.composition import ilr_inv

# forecast_comp = ilr_inv(forecast_ilr)
# forecast_percent = forecast_comp * 100

# forecast_df = pd.DataFrame(
#     forecast_percent,
#     columns=df.columns
# )

# print(forecast_df)
