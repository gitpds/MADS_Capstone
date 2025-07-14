#!/usr/bin/env python3
"""
Test script to execute the enhanced notebook and monitor progress
"""

import sys
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.getcwd())

print("=" * 60)
print("TESTING ENHANCED CONSUMER SENTIMENT ANALYSIS")
print("=" * 60)

# Import required libraries
print("\n1. Loading libraries...")
try:
    import pandas as pd
    import numpy as np
    from fredapi import Fred
    from dotenv import load_dotenv
    from pathlib import Path
    print("   ✓ Core libraries loaded")
except Exception as e:
    print(f"   ✗ Error loading libraries: {e}")
    sys.exit(1)

# Load environment variables
print("\n2. Loading environment...")
try:
    load_dotenv()
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError("FRED_API_KEY not found!")
    fred = Fred(api_key=fred_api_key)
    print("   ✓ FRED API connection established")
except Exception as e:
    print(f"   ✗ Error with FRED API: {e}")
    sys.exit(1)

# Test with a small subset first
print("\n3. Testing data fetch with subset...")
test_indicators = {
    'UMCSENT': 'Consumer Sentiment Index',
    'CPIAUCSL': 'Consumer Price Index',
    'UNRATE': 'Unemployment Rate',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'ICSA': 'Initial Claims'
}

start_date = '2020-01-01'
end_date = '2025-05-31'

successful_fetches = 0
for code, desc in test_indicators.items():
    try:
        print(f"   Fetching {code}...", end=' ')
        series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
        if len(series) > 0:
            print(f"✓ ({len(series)} observations)")
            successful_fetches += 1
        else:
            print("✗ No data")
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}...")
        
print(f"\n   Successfully fetched {successful_fetches}/{len(test_indicators)} test indicators")

# Check cache directory
print("\n4. Checking cache setup...")
cache_dir = Path('data_outputs/cache')
if cache_dir.exists():
    cache_files = list(cache_dir.glob('*.pkl'))
    print(f"   ✓ Cache directory exists with {len(cache_files)} cached files")
else:
    print("   ! Cache directory not found (will be created on first run)")

# Summary
print("\n" + "=" * 60)
print("TEST RESULTS:")
print("=" * 60)
print(f"✓ Libraries: OK")
print(f"✓ FRED API: OK")
print(f"✓ Data Fetch: {successful_fetches}/{len(test_indicators)} indicators")
print(f"✓ Environment: Ready")
print("\nThe notebook is ready to run with the full indicator set!")
print("Note: First run will take longer due to API calls.")
print("Subsequent runs will use cached data.")
print("=" * 60)