# Consumer Sentiment Analysis Project Guide

## Project Overview

This document compiles detailed guidance for a graduate capstone group project focused on using Federal Reserve Economic Data (FRED) to predict and analyze the Michigan Consumer Sentiment Survey (UMCSENT). The goal is to identify key economic indicators that drive consumer sentiment, understand how these relationships shift over time (from the early 1990s to the present, e.g., January 1990 to May 2025 or latest available), and explore trailing indicators that follow changes in sentiment.

Key objectives:
- **Predict UMCSENT**: Use quantitative FRED data points as inputs (features) to model consumer sentiment, which is inherently qualitative but anchored in economic realities.
- **Analyze Shifts Over Time**: Demonstrate how drivers of sentiment change across periods (e.g., pre-2000 tech boom, 2001-2008 pre-crisis, 2009-2019 recovery, 2020-2025 post-COVID).
- **Quantify Impacts and Lags**: Examine how percentage changes in inputs (e.g., a 1% increase in inflation) affect UMCSENT with specific lags (e.g., 30 days) and magnitudes, including non-linear effects (e.g., larger changes may have shorter lags).
- **Identify Trailing Indicators**: Determine outcomes that lag behind sentiment changes, such as retail sales or housing starts.
- **Address Challenges**: Handle the mix of quantitative drivers and qualitative "vibes" (e.g., media, uncertainty), avoid overfitting with too many features (aim for 10-15), and ensure robust analysis.

The project is built in Python using Jupyter Notebooks. Libraries include: pandas (data handling), numpy (computations), statsmodels (time-series models like VAR), scikit-learn (machine learning like Random Forest, LASSO), matplotlib/seaborn (visualizations).

Time frame: Early 1990s (e.g., January 1990) to present (latest FRED data, ~May 2025). Use monthly data to align with UMCSENT frequency.

## Data Sources and Preparation

### FRED Data Acquisition
- Access FRED via their API (use `fredapi` library in Python if available; otherwise, download CSVs from fred.stlouisfed.org).
- Pull monthly series for UMCSENT and all features/indicators.
- Handle missing data: Use linear interpolation for gaps; adjust start dates if series don't go back to 1990 (e.g., VIXCLS starts ~1990 but confirm).
- Standardization: Z-score normalize (subtract mean, divide by std dev) for modeling; compute percentage changes for impact analysis: `(value_t - value_{t-1}) / value_{t-1} * 100`.
- Alignment: Ensure all series are on the same timestamp (e.g., end-of-month).

### Top 20 Input Features (Leading Indicators/Drivers)
These are FRED series likely to influence UMCSENT, covering inflation, jobs, housing, spending, and market vibes. Prioritize based on correlation strength; start with 10-12 to avoid overfitting.

1. **CPIAUCSL**: Consumer Price Index (All Urban Consumers) – Core inflation measure; higher prices erode sentiment.
2. **UNRATE**: Unemployment Rate – Job security is a primary driver; rises correlate with sentiment drops.
3. **GASREGW**: Weekly Regular Gasoline Prices – Immediate pocketbook impact; spikes hit sentiment quickly.
4. **RSAFS**: Retail Sales and Food Services – Reflects spending behavior; proxy for economic health.
5. **DSPIC96**: Real Disposable Personal Income (Chained 2017 Dollars) – More income boosts confidence.
6. **AHETPI**: Average Hourly Earnings of Production and Nonsupervisory Employees (Total Private, Inflation-Adjusted) – Wage growth affects purchasing power.
7. **CPIUFDSL**: Consumer Price Index for Food – Grocery costs are a daily concern.
8. **CUSR0000SEHA**: Consumer Price Index for Shelter (Owners' Equivalent Rent) – Housing expenses weigh heavily.
9. **HOUST**: Housing Starts – Signals construction activity and optimism.
10. **SP500**: S&P 500 Index – Stock market performance influences wealth effects, especially for investors.
11. **PSAVERT**: Personal Savings Rate – Higher savings can indicate caution or confidence.
12. **TCMDO**: Total Consumer Debt Outstanding – Debt burdens dampen sentiment.
13. **M1SL**: M1 Money Supply – Liquidity affects spending ease.
14. **INDPRO**: Industrial Production Index – Broad economic activity indicator.
15. **CSUSHPINSA**: S&P/Case-Shiller U.S. National Home Price Index – Housing wealth impacts feelings.
16. **PCE**: Personal Consumption Expenditures – Core spending trends.
17. **UMCSENT (Lagged)**: Lagged Consumer Sentiment – Past sentiment influences current views (autoregressive component).
18. **FEDFUNDS**: Effective Federal Funds Rate – Interest rates affect borrowing and savings.
19. **CC4WSA**: Consumer Credit Outstanding (Seasonally Adjusted) – Credit availability shapes spending.
20. **VIXCLS**: CBOE Volatility Index (VIX) – Market uncertainty proxies qualitative "vibes" and can spook consumers.

**Selection Tips**: Use Random Forest feature importance or LASSO regression to rank and select top 10-15. Post-COVID, prioritize inflation-related (e.g., CPIAUCSL, GASREGW).

### Trailing Indicators (Outputs/Lagging Effects)
These follow changes in UMCSENT, often with 1-3 month lags. Model UMCSENT as predictor for these.

1. **RSAFS**: Retail Sales and Food Services – Spending increases after sentiment improves.
2. **PCE**: Personal Consumption Expenditures – Broader spending lags confidence.
3. **CC4WSA**: Consumer Credit – Borrowing rises post-sentiment uptick.
4. **HOUST**: Housing Starts – Construction ramps up with optimism (slow due to permits).
5. **DGORDER**: Manufacturers' New Orders: Durable Goods – Big-ticket orders follow sentiment.
6. **PSAVERT**: Personal Savings Rate – Drops (more spending) after sentiment rises; rises with caution.
7. **BUSINV**: Total Business Inventories – Adjusted based on demand, trailing sentiment-driven spending.

**Historical Shifts**: In 1990s, stock market (SP500) trailed booms; post-2008, housing/job recovery; post-COVID, inflation fears delayed spending/credit.

## Analysis Structure: Step-by-Step Guide

This is a comprehensive workflow for Jupyter Notebooks. Divide into notebooks: 1) Data Prep, 2) Exploratory Analysis, 3) Modeling, 4) Visualization & Interpretation.

### Step 1: Data Gathering and Cleaning
- **Code Snippet**:
  ```python
  import pandas as pd
  from fredapi import Fred  # If using API; install if needed (pip install fredapi)
  
  # Initialize FRED API (get key from fred.stlouisfed.org)
  fred = Fred(api_key='YOUR_API_KEY')
  
  # Pull data (example for UMCSENT and CPIAUCSL)
  umcsent = fred.get_series('UMCSENT', observation_start='1990-01-01')
  cpiaucsl = fred.get_series('CPIAUCSL', observation_start='1990-01-01')
  
  # Combine into DataFrame
  df = pd.DataFrame({'UMCSENT': umcsent, 'CPIAUCSL': cpiaucsl})
  
  # Handle missing: Interpolate
  df = df.interpolate(method='linear')
  
  # Percentage changes
  df_pct = df.pct_change() * 100
  df_pct = df_pct.dropna()  # Drop first row
  
  # Z-score normalization
  from scipy.stats import zscore
  df_norm = df.apply(zscore)
  ```
- Repeat for all 20+ features and trailing indicators.
- Save as CSV: `df.to_csv('fred_data.csv')`.

### Step 2: Exploratory Data Analysis (EDA)
- **Correlation Analysis**: Compute Pearson correlations between inputs and UMCSENT.
  ```python
  corr_matrix = df_norm.corr()
  print(corr_matrix['UMCSENT'].sort_values(ascending=False))  # Top correlates
  ```
- **Time-Series Plots**: Overlay UMCSENT with key inputs.
  ```python
  import matplotlib.pyplot as plt
  plt.plot(df.index, df['UMCSENT'], label='UMCSENT')
  plt.plot(df.index, df['CPIAUCSL'], label='CPIAUCSL')
  plt.legend()
  plt.show()
  ```
- **Sub-Periods**: Split data (e.g., df_1990s = df['1990':'2000']).
- **Rolling Correlations**: To see shifts.
  ```python
  rolling_corr = df['UMCSENT'].rolling(window=60).corr(df['CPIAUCSL'])  # 5-year window
  rolling_corr.plot()
  ```
- **Cross-Correlations for Lags**: Use `numpy.correlate` to find optimal lags.
  ```python
  import numpy as np
  lags = np.arange(-12, 13)  # Months
  cross_corr = [df_pct['UMCSENT'].corr(df_pct['CPIAUCSL'].shift(lag)) for lag in lags]
  max_lag = lags[np.argmax(cross_corr)]  # Lag with highest corr
  ```

### Step 3: Modeling Inputs to Predict UMCSENT
- **Linear Regression Baseline**:
  ```python
  import statsmodels.api as sm
  X = df_norm[['CPIAUCSL', 'UNRATE', 'GASREGW']]  # Example features
  y = df_norm['UMCSENT']
  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  print(model.summary())  # Coefficients show impacts
  ```
- **Vector Autoregression (VAR) for Dynamics**:
  ```python
  from statsmodels.tsa.vector_ar.var_model import VAR
  model_var = VAR(df_norm[['UMCSENT', 'CPIAUCSL', 'UNRATE']])
  results = model_var.fit(maxlags=6)  # Up to 6 months
  print(results.summary())
  
  # Impulse Response Functions (IRFs)
  irf = results.irf(12)  # 12 periods ahead
  irf.plot(orth=False)  # Shows shock effects, e.g., 1% CPI shock on UMCSENT
  ```
- **Quantifying Lags and Magnitudes**:
  - Bin changes: e.g., low_CPI = df_pct['CPIAUCSL'] < 1, high_CPI = df_pct['CPIAUCSL'] >= 2.
  - Interaction Terms: Add to OLS: `X['CPI_high_interact'] = X['CPIAUCSL'] * (df_pct['CPIAUCSL'] >= 2)`.
  - Non-Linear: Threshold regression or piecewise (custom functions).
  - For each bin, compute cross-corr to find lag, then regress for magnitude (e.g., β for 1% change = X% sentiment drop).
- **Machine Learning Enhancements**:
  - Random Forest for Feature Importance:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))
    ```
  - LASSO for Selection: `from sklearn.linear_model import Lasso; lasso = Lasso(alpha=0.1).fit(X, y)`.
- **Rolling Windows for Time Shifts**:
  ```python
  windows = pd.date_range(start='1995-01-01', end=df.index[-1], freq='5Y')  # 5-year steps
  for end in windows:
      window_df = df[:end]
      # Re-run corr or VAR here
  ```

### Step 4: Modeling Trailing Indicators
- Flip the VAR: Use UMCSENT as predictor for trailing like RSAFS.
  ```python
  model_var_trail = VAR(df_norm[['UMCSENT', 'RSAFS', 'PCE']])
  results_trail = model_var_trail.fit(maxlags=6)
  irf_trail = results_trail.irf(12)
  irf_trail.plot()  # UMCSENT shock on RSAFS
  ```
- Granger Causality: Test if UMCSENT "causes" RSAFS.
  ```python
  from statsmodels.tsa.stattools import grangercausalitytests
  grangercausalitytests(df_norm[['RSAFS', 'UMCSENT']], maxlag=6)
  ```

### Step 5: Handling Qualitative Aspects and Robustness
- Proxy Qualitatives: Use VIXCLS for uncertainty.
- Feature Engineering: Ratios like CPI-to-DSPIC96 (inflation vs. income).
- Overfitting Checks: Cross-validation in sklearn; AIC/BIC in statsmodels for lag selection.
- Structural Breaks: Chow test in statsmodels to detect shifts (e.g., 2008, 2020).
  ```python
  from statsmodels.stats.diagnostic import breaks_cusumolsresid
  # Apply to residuals
  ```

### Step 6: Visualization and Interpretation
- Heatmaps: Correlation matrices per period.
- Plots: IRFs, rolling corrs, lag-vs-magnitude (e.g., scatter of CPI % change vs. lag to UMCSENT peak).
- Tables: Summarize coefficients, e.g.,
  | Feature | Lag (Days) | Magnitude (per 1% Change) | Period |
  |---------|------------|----------------------------|--------|
  | CPIAUCSL | 30 | -0.5% in UMCSENT | Post-2020 |
- Insights: E.g., "A 1% CPI increase drops UMCSENT by 0.5% after 30 days; 2% shortens to 15 days."

### Step 7: Presentation and Next Steps
- Compile findings: Which inputs dominate per era? How do lags/magnitudes vary? Trailing effects?
- Challenges: Qualitative noise may weaken correlations; mitigate with proxies and sub-periods.
- Extensions: Forecast UMCSENT; add external data if allowed (e.g., news sentiment via APIs).

This guide serves as a core reference for code development. Update with group findings.