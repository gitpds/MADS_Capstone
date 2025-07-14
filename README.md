# Consumer Sentiment Analysis Using Federal Reserve Economic Data

## MADS Capstone Project - Rate Hike Rangers

### Project Overview

This project analyzes the Michigan Consumer Sentiment Index (UMCSENT) using Federal Reserve Economic Data (FRED) to understand the complex relationships between economic indicators and consumer confidence. We employ machine learning techniques to both predict sentiment from economic conditions and use sentiment as a leading indicator for future economic activity.

### Key Features

- **Comprehensive Economic Analysis**: Analyzes 27+ economic indicators from FRED
- **Time Series Modeling**: Proper cross-validation for time series data
- **Period-Specific Analysis**: Examines how relationships change across economic cycles
- **Dual Perspective**: Sentiment as both outcome and predictor
- **Robust Evaluation**: Baseline comparisons and multiple metrics

### Repository Structure

```
MADS_Capstone/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── consumer_sentiment_analysis_final.ipynb      # Main analysis notebook
├── diagnostic_analysis.ipynb                    # Model diagnostic tools
├── data_outputs/                               # Processed data files
│   ├── processed_data/                         # Monthly FRED data
│   └── cache/                                  # API cache files
├── final_outputs/                              # Final analysis results
│   ├── visualizations/                         # Charts and graphs
│   ├── data/                                   # Feature datasets
│   └── results/                                # Model outputs
└── project_requirements/                       # Project specifications
```

### Installation and Setup

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd MADS_Capstone
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up FRED API access**
   - Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
   - Create a `.env` file in the project root:
     ```
     FRED_API_KEY=your_api_key_here
     ```

### Running the Analysis

1. **Main Analysis**
   ```bash
   jupyter notebook consumer_sentiment_analysis_final.ipynb
   ```
   Run all cells in order. The notebook will:
   - Load cached FRED data (or fetch if not cached)
   - Engineer features based on economic theory
   - Train and evaluate multiple models
   - Generate visualizations and results

2. **Diagnostic Analysis** (optional)
   ```bash
   jupyter notebook diagnostic_analysis.ipynb
   ```
   Use this to investigate model performance issues.

### Data Access

All data is publicly available from the Federal Reserve Economic Data (FRED) API:
- Website: https://fred.stlouisfed.org/
- API Documentation: https://fred.stlouisfed.org/docs/api/fred/
- Terms of Use: https://fred.stlouisfed.org/legal/

The analysis uses monthly frequency data from January 1990 to May 2025 (or latest available).

### Key Results

- **Model Performance**: Achieved R² of 0.3-0.5 for sentiment prediction
- **Primary Drivers**: Inflation, unemployment, and gas prices most impact sentiment
- **Temporal Patterns**: Model performance varies across economic periods
- **Leading Indicator**: Sentiment shows modest predictive power for retail sales and consumption

### Visualizations

The analysis generates several key visualizations:
- Model performance comparison
- Feature importance analysis
- Period-specific sentiment trends
- Leading indicator relationships

All visualizations are saved to `final_outputs/visualizations/`

### Technical Requirements

- Python 3.8+
- 4GB RAM minimum
- Internet connection for initial data fetch
- Jupyter Notebook or JupyterLab

### Troubleshooting

**Issue**: FRED API rate limiting
- **Solution**: The code includes exponential backoff. If persistent, wait 1 minute between runs.

**Issue**: Missing data for certain indicators
- **Solution**: The code handles missing data through interpolation and forward filling.

**Issue**: Negative R² values
- **Solution**: This indicates model performance worse than baseline. Check feature selection and data quality.

### Contributing

This is a capstone project for the Master of Applied Data Science program. For questions or collaboration:
- Open an issue in the repository
- Contact the Rate Hike Rangers team

### License

This project is for educational purposes as part of the University of Michigan MADS program.

### Acknowledgments

- University of Michigan MADS Program
- Federal Reserve Bank of St. Louis for FRED data
- Project advisor and instructional team

---

*Last updated: December 2024*
