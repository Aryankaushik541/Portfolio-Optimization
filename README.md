# Portfolio Optimizer Comparison

This folder contains a clean combined version of the portfolio optimizer.

It uses:

- Real Yahoo Finance adjusted close data from 2010-01-01 to the latest available market date
- Long-only portfolio weights
- Maximum single-stock allocation cap, default 40%
- Single-objective optimization for Sharpe Ratio
- RMPSO, Tabu Search, and Genetic Algorithm comparison
- Result table shows only Annual Risk and Sharpe Ratio
- Fresh Yahoo download by default
- Report and chart output in `reports/`

Run:

```powershell
python optimized_portfolio_ga\portfolio_ga_optimized.py
```

Use cached Yahoo data if available:

```powershell
python optimized_portfolio_ga\portfolio_ga_optimized.py --use-cache
```

Useful tuning:

```powershell
python optimized_portfolio_ga\portfolio_ga_optimized.py --generations 1000 --population 500 --max-weight 0.40
```
