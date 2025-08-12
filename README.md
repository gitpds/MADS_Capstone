# Consumer Sentiment Analysis: Using Federal Reserve Data to Predict Consumer Sentiment

**MADS Capstone Project - Rate Hike Rangers**  
Paul Stotts (pdstotts), Dave Norine (dnorine), Ali Alrubaiee (aalrubai)

## Overview
This project investigates the drivers and predictability of U.S. consumer sentiment using a wide array of economic indicators from the Federal Reserve Economic Data (FRED) API. The primary target is the University of Michigan Consumer Sentiment Index (UMCSENT), a key barometer of economic optimism or pessimism.

### Key Questions Addressed
1. **What drives consumer sentiment?** Which economic indicators have the strongest influence?
2. **How quickly do changes propagate?** What are the lag structures between economic events and sentiment shifts?
3. **Can we forecast sentiment accurately?** How well can we predict future sentiment using current data?
4. **What are the downstream effects?** How does sentiment influence actual economic behavior?

## Repository Structure
- `Final_unified_sentiment_analysis.ipynb`: Main analysis notebook (data collection, EDA, modeling, results)
- `requirements.txt`: Python dependencies
- `outputs/`: Generated data, results, and visualizations
    - `data/`: Processed datasets and engineered features
    - `results/`: Model outputs, feature importances, Granger causality, etc.
    - `visualizations/`: Plots and figures
    - `cache/`: Cached API responses for reproducibility

## Data Access Statement
- **FRED API**: 43+ economic indicators, including inflation, unemployment, stock market, housing, interest rates, and more. To access this data, request an API key here: https://fred.stlouisfed.org/docs/api/api_key.html
- **UMCSENT**: University of Michigan Consumer Sentiment Index (target variable)

## Main Steps
1. **Data Collection**: Fetches and caches monthly data for all indicators from FRED
2. **Feature Engineering**: Creates lags, percent changes, rolling stats, spreads, and interaction terms
3. **Exploratory Data Analysis**: Examines relationships and correlations between sentiment and economic indicators
4. **Lead-Lag & Causality Analysis**: Cross-correlation and Granger causality to identify leading/lagging indicators
5. **Modeling**: Forecasts sentiment using Linear Regression, Random Forest, SVR, and ensemble methods with rolling cross-validation
6. **Evaluation & Visualization**: Compares model performance and visualizes key findings

## Usage
1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
2. **Set up FRED API key**
   - Create a `.env` file with your FRED API key:
     ```
     FRED_API_KEY=your_fred_api_key_here
     ```
3. **Run the notebook**
   - Open `Final_unified_sentiment_analysis.ipynb` in Jupyter and run all cells.

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn, xgboost
- statsmodels
- fredapi
- python-dotenv
- shap (optional, for model explainability)

## Outputs
- **Data**: Processed monthly datasets and engineered features
- **Results**: Feature importances, Granger causality, model performance metrics
- **Visualizations**: Time series, correlation heatmaps, lag analysis, model comparisons

## Notable Findings
- Tangible indicators (housing, gas prices) have stronger relationships with sentiment than financial markets
- Some economic indicators lead sentiment by months or years, while others are contemporaneous or lagging
- Linear models perform best, but ensemble methods offer robust performance
- Sentiment can Granger-cause changes in some economic indicators

## Authors
- Paul Stotts (pdstotts)
- Dave Norine (dnorine)
- Ali Alrubaiee (aalrubai)
- Tools Used: Generative AI

## License
This project is for educational purposes. Please cite appropriately if using or extending this work.
