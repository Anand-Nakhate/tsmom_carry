# TSMOM & Carry: State-of-the-Art Quantitative Research

## Overview
This project implements a professional-grade quantitative research pipeline for Time-Series Momentum (TSMOM) and Carry strategies across a global universe of futures contracts. It features robust data engineering, advanced signal processing, volatility scaling, and sophisticated portfolio stacking methods.

## Key Features
- **Data Pipeline:** Automated ingestion of Databento `.dbn` files, cleaning, and continuous series construction.
- **Signal Generation:**
    - **TSMOM:** Multi-horizon (3m, 6m, 12m) trend following.
    - **Carry:** Sleeve-neutral carry signals based on term structure.
- **Portfolio Construction:**
    - **Volatility Scaling:** Inverse volatility weighting with caps and floors.
    - **Stacking:** Equal Weight, Risk Parity, and Dynamic Ridge Regression.
- **Analysis:**
    - **Performance Attribution:** Decomposition of returns by asset class.
    - **Regime Analysis:** Performance evaluation across Pre-COVID, COVID, and Post-COVID periods.
    - **Robustness:** Sensitivity analysis to transaction costs and parameters.

## Project Structure
```
├── data/                   # Processed data (parquet)
├── notebooks/              # Research notebooks
│   ├── 01_data_intake.ipynb
│   ├── 02_term_structure.ipynb
│   ├── 03_data_analysis.ipynb
│   ├── 04_signal_generation.ipynb
│   ├── 05_volatility_calculation.ipynb
│   ├── 06_portfolio_construction.ipynb
│   └── 07_backtest_and_robustness.ipynb
├── src/                    # Source code modules
│   ├── analysis.py         # Statistical analysis & metrics
│   ├── backtest.py         # Backtesting engine
│   ├── cleaning.py         # Data cleaning logic
│   ├── config.py           # Universe definition
│   ├── download.py         # Databento downloader
│   ├── portfolio.py        # Portfolio construction logic
│   ├── process_dataset.py  # Data processing pipeline
│   ├── signals.py          # Signal generation logic
│   └── term_structure.py   # Term structure calculations
│   └── vol.py              # Vol calculations
└── README.md
```

## Installation
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy scipy statsmodels matplotlib seaborn databento scikit-learn`.
3.  Set `DATABENTO_API_KEY` environment variable.

## Usage
Run the notebooks in order to reproduce the research:
1.  `01_data_intake.ipynb`: Download and process raw data.
2.  `02_term_structure.ipynb`: Analyze term structure and liquidity.
3.  `03_data_analysis.ipynb`: Clean returns and perform PCA.
4.  `04_signal_generation.ipynb`: Generate TSMOM and Carry signals.
5.  `06_portfolio_construction.ipynb`: Construct and stack portfolios.
6.  `07_backtest_and_robustness.ipynb`: Run backtests and analyze performance.

## License
Proprietary Research.
