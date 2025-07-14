#!/usr/bin/env python3
"""
Run the enhanced consumer sentiment analysis with full indicator set
"""

import os
import sys
import time
import json
import warnings
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# FRED API
from fredapi import Fred
from dotenv import load_dotenv

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

# Create output directories
output_dirs = [
    'visualizations/time_series',
    'visualizations/correlations', 
    'visualizations/irf_plots',
    'visualizations/feature_importance',
    'visualizations/dashboard',
    'data_outputs/raw_data',
    'data_outputs/processed_data',
    'data_outputs/model_predictions',
    'data_outputs/cache',
    'results/summary_tables',
    'results/model_outputs',
    'results/performance_metrics'
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

print("=" * 80)
print("ENHANCED CONSUMER SENTIMENT ANALYSIS - FULL PIPELINE TEST")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load environment and initialize FRED
load_dotenv()
fred_api_key = os.getenv('FRED_API_KEY')
if not fred_api_key:
    raise ValueError("FRED_API_KEY not found!")
fred = Fred(api_key=fred_api_key)

# Define full indicator set
series_dict = {
    # Target variable
    'UMCSENT': 'Consumer Sentiment Index',
    
    # Core Economic Indicators
    'CPIAUCSL': 'Consumer Price Index (All Urban)',
    'UNRATE': 'Unemployment Rate',
    'GASREGW': 'Regular Gasoline Prices (Weekly)',
    'RSAFS': 'Retail Sales and Food Services',
    'DSPIC96': 'Real Disposable Personal Income',
    'AHETPI': 'Average Hourly Earnings (Production Workers)',
    'CPIUFDSL': 'Consumer Price Index for Food',
    'CUSR0000SEHA': 'Consumer Price Index for Shelter',
    'HOUST': 'Housing Starts',
    'SP500': 'S&P 500 Index',
    'PSAVERT': 'Personal Savings Rate',
    'TCMDO': 'Total Consumer Debt Outstanding',
    'M1SL': 'M1 Money Supply',
    'INDPRO': 'Industrial Production Index',
    'CSUSHPINSA': 'Case-Shiller Home Price Index',
    'PCE': 'Personal Consumption Expenditures',
    'FEDFUNDS': 'Effective Federal Funds Rate',
    'CC4WSA': 'Consumer Credit Outstanding',
    'VIXCLS': 'CBOE Volatility Index (VIX)',
    'GS10': '10-Year Treasury Yield',
    'FMNHSHPSIUS': 'Home Purchase Sentiment Index',
    'HPIPONM226S': 'FHFA House Price Index',
    'PERMIT': 'Building Permits',
    'MORTGAGE30US': '30-Year Mortgage Rate',
    'DGORDER': 'Durable Goods Orders',
    'BUSINV': 'Total Business Inventories',
    
    # Additional Leading Indicators
    'ICSA': 'Initial Jobless Claims',
    'PAYEMS': 'Total Nonfarm Payrolls',
    'CIVPART': 'Labor Force Participation Rate',
    'U6RATE': 'Underemployment Rate (U6)',
    'T10Y2Y': '10-Year/2-Year Treasury Spread',
    'BAMLH0A0HYM2': 'High Yield Bond Spread',
    'TEDRATE': 'TED Spread (Financial Stress)',
    'ANFCI': 'Chicago Fed Financial Conditions Index',
    'UMEXPINF1YR': '1-Year Expected Inflation (Michigan)',
    'MICH': 'Consumer Expectations Index (Michigan)',
    'CPILFESL': 'Core CPI (ex Food & Energy)',
    'NAPMPMI': 'ISM Manufacturing PMI',
    'NEWORDER': 'Manufacturers New Orders',
    'AWHMAN': 'Average Weekly Hours - Manufacturing',
    'DCOILWTICO': 'WTI Crude Oil Prices',
    'DEXUSEU': 'US Dollar/Euro Exchange Rate',
    'USEPUINDXD': 'Economic Policy Uncertainty Index'
}

# Cache functions
cache_dir = Path('data_outputs/cache')

def get_cache_key(code, start_date, end_date):
    key_str = f"{code}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_from_cache(code, start_date, end_date):
    cache_key = get_cache_key(code, start_date, end_date)
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None

def save_to_cache(code, start_date, end_date, data):
    cache_key = get_cache_key(code, start_date, end_date)
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass

# Fetch data with exponential backoff
def fetch_with_exponential_backoff(fred, code, start_date, end_date, max_retries=5):
    base_delay = 1
    for attempt in range(max_retries):
        try:
            series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            return series
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "Rate Limit" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"\n    Rate limit hit. Waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})...", end='')
                    time.sleep(delay)
                else:
                    raise e
            else:
                raise e
    return None

# Fetch all data
print("\n1. FETCHING DATA")
print("-" * 80)

start_date = '1990-01-01'
end_date = '2025-05-31'

data_dict = {}
fetch_errors = []
cache_hits = 0
api_calls = 0
request_delay = 0.25

# Limit to first 20 indicators for testing
test_limit = 20
indicators_to_fetch = list(series_dict.items())[:test_limit]

print(f"Fetching {len(indicators_to_fetch)} indicators (limited for testing)...")

for i, (code, description) in enumerate(indicators_to_fetch):
    print(f"[{i+1}/{len(indicators_to_fetch)}] {code}: {description[:40]}...", end=' ')
    
    # Try cache first
    cached_data = load_from_cache(code, start_date, end_date)
    
    if cached_data is not None:
        data_dict[code] = cached_data
        cache_hits += 1
        print(f"✓ CACHED ({len(cached_data)} obs)")
        continue
    
    # Fetch from API
    try:
        if api_calls > 0:
            time.sleep(request_delay)
        
        series = fetch_with_exponential_backoff(fred, code, start_date, end_date)
        api_calls += 1
        
        if series is not None and len(series) > 0:
            data_dict[code] = series
            save_to_cache(code, start_date, end_date, series)
            print(f"✓ FETCHED ({len(series)} obs)")
        else:
            print(f"✗ No data")
            fetch_errors.append((code, "No data in date range"))
            
    except Exception as e:
        error_msg = str(e)
        print(f"✗ ERROR: {error_msg[:50]}...")
        fetch_errors.append((code, error_msg))

print(f"\nData fetching complete: {len(data_dict)}/{len(indicators_to_fetch)} successful")
print(f"Cache hits: {cache_hits}, API calls: {api_calls}")

if len(data_dict) == 0:
    print("ERROR: No data fetched!")
    sys.exit(1)

# Process data
print("\n2. PROCESSING DATA")
print("-" * 80)

# Combine into DataFrame
df_raw = pd.DataFrame(data_dict)
df_raw.index = pd.to_datetime(df_raw.index)

# Resample to monthly
df_monthly = pd.DataFrame(index=pd.date_range(start=df_raw.index.min(), 
                                              end=df_raw.index.max(), 
                                              freq='ME'))

# Identify frequencies
for col in df_raw.columns:
    non_null_data = df_raw[col].dropna()
    if len(non_null_data) > 1:
        avg_days = (non_null_data.index[-1] - non_null_data.index[0]).days / (len(non_null_data) - 1)
        if avg_days < 15:  # Daily or weekly
            df_monthly[col] = df_raw[col].resample('ME').mean()
        else:  # Monthly
            df_monthly[col] = df_raw[col].resample('ME').last()

df_monthly = df_monthly.ffill().bfill()
print(f"Monthly data shape: {df_monthly.shape}")

# Enhanced feature engineering
print("\n3. FEATURE ENGINEERING")
print("-" * 80)

# Create features
df_pct_change = df_monthly.pct_change() * 100
df_pct_change.columns = [f"{col}_pct" for col in df_pct_change.columns]

df_yoy = df_monthly.pct_change(12) * 100
df_yoy.columns = [f"{col}_yoy" for col in df_yoy.columns]

df_momentum = df_monthly.rolling(window=3, min_periods=1).mean()
df_momentum.columns = [f"{col}_ma3" for col in df_momentum.columns]

# Create spreads
df_spreads = pd.DataFrame(index=df_monthly.index)
if 'GS10' in df_monthly.columns and 'FEDFUNDS' in df_monthly.columns:
    df_spreads['yield_curve_spread'] = df_monthly['GS10'] - df_monthly['FEDFUNDS']

# Combine features
df_features = pd.concat([df_monthly, df_pct_change, df_yoy, df_momentum, df_spreads], axis=1)

# Add UMCSENT lags
if 'UMCSENT' in df_monthly.columns:
    for lag in [1, 3, 6, 12]:
        df_features[f"UMCSENT_lag{lag}"] = df_monthly['UMCSENT'].shift(lag)

df_features_clean = df_features.dropna()
print(f"Feature matrix: {df_features_clean.shape}")

# Analysis
print("\n4. ANALYSIS")
print("-" * 80)

if 'UMCSENT' in df_features_clean.columns:
    # Correlation analysis
    corr_with_sentiment = df_monthly.corr()['UMCSENT'].sort_values(ascending=False)
    print("Top correlations with UMCSENT:")
    print(corr_with_sentiment.head(10))
    
    # Feature selection
    feature_cols = [col for col in df_features_clean.columns 
                    if not col.startswith('UMCSENT') or 'lag' in col]
    X = df_features_clean[feature_cols]
    y = df_features_clean['UMCSENT']
    
    # Split data
    split_date = '2020-01-01'
    X_train = X.loc[:split_date]
    X_test = X.loc[split_date:]
    y_train = y.loc[:split_date]
    y_test = y.loc[split_date:]
    
    print(f"\nTrain/Test split: {len(X_train)} train, {len(X_test)} test")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_test = rf.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nRandom Forest Performance:")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(importance.head(10))

# Save results
print("\n5. SAVING RESULTS")
print("-" * 80)

# Save data
df_monthly.to_csv('data_outputs/processed_data/monthly_data_enhanced.csv')
df_features_clean.to_csv('data_outputs/processed_data/features_enhanced.csv')

# Save summary
summary = {
    'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'indicators_requested': len(indicators_to_fetch),
    'indicators_fetched': len(data_dict),
    'cache_hits': cache_hits,
    'api_calls': api_calls,
    'monthly_data_shape': list(df_monthly.shape),
    'features_shape': list(df_features_clean.shape),
    'date_range': [str(df_monthly.index.min()), str(df_monthly.index.max())]
}

if 'test_r2' in locals():
    summary['model_performance'] = {
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse)
    }

with open('results/summary_tables/enhanced_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Results saved")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)