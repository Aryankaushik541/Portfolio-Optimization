import argparse
from datetime import date
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "V", "PG", "XOM", "NVDA"]
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V":   "Visa Inc.",
    "PG":  "The Procter & Gamble Company",
    "XOM": "Exxon Mobil Corporation",
    "NVDA": "NVIDIA Corporation",
}

START_DATE     = "2010-01-01"
END_DATE       = date.today().isoformat()
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.02
SEED           = 748286077

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR   = OUTPUT_DIR / "data"
REPORT_DIR = OUTPUT_DIR / "reports"


@dataclass(frozen=True)
class GAConfig:
    population_size: int   = 50
    generations:     int   = 1000
    elite_size:      int   = 3
    tournament_size: int   = 5
    crossover_rate:  float = 0.92
    mutation_rate:   float = 0.15
    max_weight:      float = 0.40


def fetch_prices(tickers, start, end, use_cache=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"yahoo_prices_{start}_{end}.csv".replace("-", "")

    if cache_path.exists() and use_cache:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance: pip install yfinance") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)

    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(axis=1, thresh=int(0.90 * len(prices))).dropna()
    prices = prices.loc[:, [c for c in tickers if c in prices.columns]]

    if prices.shape[1] < 2:
        raise RuntimeError("Not enough valid ticker data.")

    prices.to_csv(cache_path)
    return prices


def daily_returns(prices):
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        raise RuntimeError("No usable daily returns.")
    return returns


def repair_weights(weights, max_weight=1.0, iterations=8):
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)

    if w.sum() <= 1e-15:
        return np.ones_like(w) / len(w)

    w = w / w.sum()

    if max_weight >= 1.0:
        return w

    for _ in range(iterations):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if under.any() and excess > 0:
            capacity = np.maximum(max_weight - w[under], 0.0)
            cap_sum = capacity.sum()
            if cap_sum > 1e-15:
                w[under] += excess * capacity / cap_sum
        if w.sum() > 1e-15:
            w /= w.sum()

    return w / w.sum()


def portfolio_metrics(weights, mean_daily, cov_daily, returns_matrix, rf, max_weight=1.0):
    w             = repair_weights(weights, max_weight)
    annual_return = float(w @ mean_daily * TRADING_DAYS)
    annual_risk   = float(np.sqrt(max(w @ cov_daily @ w * TRADING_DAYS, 0.0)))
    sharpe        = (annual_return - rf) / annual_risk if annual_risk > 1e-12 else -np.inf

    daily_ret  = returns_matrix @ w
    cumulative = np.cumprod(1.0 + daily_ret)
    peak       = np.maximum.accumulate(cumulative)
    max_dd     = float(np.min((cumulative - peak) / peak))

    return {"weights": w, "return": annual_return, "risk": annual_risk, "sharpe": float(sharpe), "max_drawdown": max_dd}


def score_population(population, mean_daily, cov_daily, rf):
    returns = population @ mean_daily * TRADING_DAYS
    risks   = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", population, cov_daily, population) * TRADING_DAYS, 0.0))
    return np.where(risks > 1e-12, (returns - rf) / risks, -np.inf)


def init_population(rng, n_assets, cfg):
    population = [np.ones(n_assets) / n_assets]

    for i in range(n_assets):
        w = np.zeros(n_assets)
        w[i] = 1.0
        population.append(repair_weights(w, cfg.max_weight))

    while len(population) < cfg.population_size:
        if rng.random() < 0.70:
            w = rng.dirichlet(rng.uniform(0.4, 2.5, n_assets))
        else:
            w = np.zeros(n_assets)
            picked = rng.choice(n_assets, size=int(rng.integers(2, min(5, n_assets) + 1)), replace=False)
            w[picked] = rng.random(len(picked))
        population.append(repair_weights(w, cfg.max_weight))

    return np.array(population)


def tournament_select(rng, population, scores, cfg):
    idx = rng.choice(len(population), size=cfg.tournament_size, replace=False)
    return population[idx[np.argmax(scores[idx])]].copy()


def crossover(rng, p1, p2, cfg):
    alpha = rng.uniform(-0.15, 1.15, len(p1))
    return repair_weights(alpha * p1 + (1.0 - alpha) * p2, cfg.max_weight)


def mutate(rng, weights, generation, cfg):
    w      = weights.copy()
    sigma  = 0.18 * (1.0 - generation / max(cfg.generations - 1, 1)) + 0.015
    mask   = rng.random(len(w)) < cfg.mutation_rate
    w[mask] += rng.normal(0.0, sigma, mask.sum())

    if rng.random() < 0.10:
        i, j   = rng.choice(len(w), size=2, replace=False)
        amount = rng.uniform(0.0, min(max(float(w[i]), 0.0), 0.08))
        w[i]  -= amount
        w[j]  += amount

    return repair_weights(w, cfg.max_weight)


def run_ga(mean_daily, cov_daily, returns_matrix, cfg, seed=SEED):
    rng        = np.random.default_rng(seed)
    n_assets   = len(mean_daily)
    population = init_population(rng, n_assets, cfg)
    best_w     = None
    best_score = -np.inf
    history    = []

    for gen in range(cfg.generations):
        scores = score_population(population, mean_daily, cov_daily, RISK_FREE_RATE)
        order  = np.argsort(scores)[::-1]

        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_w     = population[order[0]].copy()

        history.append(best_score)

        new_pop = [population[i].copy() for i in order[:cfg.elite_size]]

        while len(new_pop) < cfg.population_size:
            p1    = tournament_select(rng, population, scores, cfg)
            p2    = tournament_select(rng, population, scores, cfg)
            child = crossover(rng, p1, p2, cfg) if rng.random() < cfg.crossover_rate else p1.copy()
            child = mutate(rng, child, gen, cfg)
            new_pop.append(child)

        population = np.array(new_pop)

    return portfolio_metrics(best_w, mean_daily, cov_daily, returns_matrix, RISK_FREE_RATE, cfg.max_weight), history


def run_ga_multi(mean_daily, cov_daily, returns_matrix, cfg, runs, seed=SEED):
    all_metrics   = []
    all_histories = []

    for i in range(runs):
        print(f"    Run {i + 1}/{runs}...", flush=True)
        metrics, history = run_ga(mean_daily, cov_daily, returns_matrix, cfg, seed + i)
        all_metrics.append(metrics)
        all_histories.append(history)

    best_idx = int(np.argmax([m["sharpe"] for m in all_metrics]))
    avg_hist = np.mean(np.array(all_histories), axis=0)
    best_hist = np.array(all_histories[best_idx])

    return all_metrics[best_idx], avg_hist, best_hist, all_metrics


def stock_rows(tickers, returns, weights):
    ann_ret    = returns.mean(axis=0) * TRADING_DAYS
    ann_risk   = returns.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    sharpes    = (ann_ret - RISK_FREE_RATE) / ann_risk.replace(0.0, np.nan)
    rows = []
    for ticker, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True):
        rows.append([ticker, COMPANY_NAMES.get(ticker, ticker), f"{w*100:.2f}%",
                     f"{ann_ret[ticker]*100:.2f}%", f"{ann_risk[ticker]*100:.2f}%", f"{sharpes[ticker]:.4f}"])
    return rows


def save_report(tickers, prices, returns, cfg, best, avg_hist, best_hist, all_metrics, runs):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "optimized_ga_report.txt"
    chart_path  = REPORT_DIR / "optimized_ga_result.png"

    sharpes   = np.array([m["sharpe"]       for m in all_metrics])
    rets      = np.array([m["return"] * 100 for m in all_metrics])
    risks     = np.array([m["risk"] * 100   for m in all_metrics])
    best_idx  = int(np.argmax(sharpes))
    worst_idx = int(np.argmin(sharpes))
    weights   = best["weights"]

    summary = [["Optimized Portfolio", f"{best['return']*100:.2f}%", f"{best['risk']*100:.2f}%",
                 f"{best['sharpe']:.4f}", f"{best['max_drawdown']*100:.2f}%"]]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Optimized GA - Single Objective Portfolio Optimisation Report\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Date range    : {prices.index[0].date()} to {prices.index[-1].date()}\n")
        f.write(f"Trading days  : {len(returns)}\n")
        f.write(f"Seed          : {SEED}\n")
        f.write(f"Risk-free rate: {RISK_FREE_RATE*100:.2f}%\n")
        f.write(f"Runs: {runs}, Generations: {cfg.generations}, Population: {cfg.population_size}\n")
        f.write("Objective: Maximize Sharpe Ratio\n\n")
        f.write("Stock Universe:\n")
        for t in tickers:
            f.write(f"  - {COMPANY_NAMES.get(t, t)}\n")
        f.write(f"Max single-stock weight: {cfg.max_weight*100:.0f}%\n\n")
        f.write(tabulate(summary, headers=["Portfolio", "Return", "Risk", "Sharpe", "Max Drawdown"], tablefmt="github"))
        f.write("\n\n")
        f.write(tabulate(stock_rows(tickers, returns, weights),
                         headers=["Ticker", "Stock Name", "Allocation", "Stock Return", "Stock Risk", "Stock Sharpe"],
                         tablefmt="github"))
        f.write("\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(avg_hist, linewidth=2.5, label="Average Sharpe", color="#e41a1c")
    axes[0].plot(best_hist, linewidth=1.6, linestyle="--", label="Best Run", color="#2ca02c")
    axes[0].set_title(f"GA Convergence\n({runs} runs, {cfg.generations} generations)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    bars = axes[1].bar(tickers, weights * 100, color=plt.cm.tab10(np.arange(len(tickers))), edgecolor="black")
    axes[1].set_title(f"Optimal Portfolio Weights\nReturn={best['return']*100:.2f}% | Risk={best['risk']*100:.2f}% | Sharpe={best['sharpe']:.4f}",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Allocation (%)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].bar_label(bars, labels=[f"{w*100:.1f}%" if w > 0.001 else "" for w in weights], padding=3, fontsize=8)
    axes[1].set_ylim(0, max(weights * 100) * 1.2)
    axes[1].grid(True, axis="y", alpha=0.3)

    labels      = ["Best Run", "Mean Run", "Worst Run"]
    ret_vals    = [rets[best_idx],    float(np.mean(rets)),    rets[worst_idx]]
    risk_vals   = [risks[best_idx],   float(np.mean(risks)),   risks[worst_idx]]
    sharpe_vals = [sharpes[best_idx], float(np.mean(sharpes)), sharpes[worst_idx]]
    x = np.arange(3)
    w = 0.25
    b1 = axes[2].bar(x - w, ret_vals,               w, label="Return (%)",   color="#4c72b0", edgecolor="black")
    b2 = axes[2].bar(x,     risk_vals,              w, label="Risk (%)",     color="#dd8452", edgecolor="black")
    b3 = axes[2].bar(x + w, [s * 20 for s in sharpe_vals], w, label="Sharpe x20", color="#2ca02c", edgecolor="black")
    axes[2].set_title("Run Performance Comparison\n(Best, Mean, Worst)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Value")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].bar_label(b1, labels=[f"{v:.2f}%" for v in ret_vals],    padding=3, fontsize=8)
    axes[2].bar_label(b2, labels=[f"{v:.2f}%" for v in risk_vals],   padding=3, fontsize=8)
    axes[2].bar_label(b3, labels=[f"{v:.4f}"  for v in sharpe_vals], padding=3, fontsize=8)
    axes[2].set_ylim(0, max(max(ret_vals), max(risk_vals), max(sharpe_vals) * 20) * 1.25)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    axes[3].hist(sharpes, bins=min(10, max(3, runs // 2)), color="#2ca02c", alpha=0.72, edgecolor="black")
    axes[3].axvline(np.mean(sharpes), color="#e41a1c", linestyle="--", linewidth=2, label=f"Mean = {np.mean(sharpes):.4f}")
    axes[3].axvline(max(sharpes),     color="#2ca02c", linewidth=2,               label=f"Best = {max(sharpes):.4f}")
    axes[3].set_title(f"Sharpe Distribution\nacross {runs} Independent Runs", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Sharpe Ratio")
    axes[3].set_ylabel("Frequency")
    mid    = float(np.mean(sharpes))
    spread = max(float(np.ptp(sharpes)), 0.0004)
    axes[3].set_xlim(mid - spread * 1.8, mid + spread * 1.8)
    axes[3].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(
        "Optimized GA - Single Objective Portfolio Optimisation (Maximise Sharpe Ratio)\n"
        f"Data: S&P 500 Yahoo Finance {prices.index[0].date()} to {prices.index[-1].date()} | Seed: {SEED} | RF = {RISK_FREE_RATE*100:.1f}%",
        fontsize=15, fontweight="bold"
    )
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return report_path, chart_path, summary, stock_rows(tickers, returns, weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache",   action="store_true")
    parser.add_argument("--max-weight",  type=float, default=GAConfig.max_weight)
    parser.add_argument("--generations", type=int,   default=GAConfig.generations)
    parser.add_argument("--population",  type=int,   default=GAConfig.population_size)
    parser.add_argument("--runs",        type=int,   default=10)
    args = parser.parse_args()

    cfg = GAConfig(population_size=args.population, generations=args.generations, max_weight=args.max_weight)

    print("\nOptimized GA - Single Objective Portfolio Optimisation")
    print("=" * 54)
    for t in TICKERS:
        print(f"  {COMPANY_NAMES.get(t, t)}")
    print(f"\nPeriod: {START_DATE} to latest | Runs={args.runs} | Pop={cfg.population_size} | Gen={cfg.generations}")

    prices  = fetch_prices(TICKERS, START_DATE, END_DATE, use_cache=args.use_cache)
    returns = daily_returns(prices)

    tickers        = list(returns.columns)
    returns_matrix = returns.to_numpy()
    mean_daily     = returns_matrix.mean(axis=0)
    cov_daily      = np.cov(returns_matrix.T)

    print(f"Data: {len(tickers)} stocks, {len(returns)} trading days ({returns.index[0].date()} to {returns.index[-1].date()})")

    print(f"\nRunning GA ({args.runs} runs)...")
    best, avg_hist, best_hist, all_metrics = run_ga_multi(mean_daily, cov_daily, returns_matrix, cfg, max(args.runs, 1))

    report_path, chart_path, summary, weight_rows = save_report(
        tickers, prices, returns, cfg, best, avg_hist, best_hist, all_metrics, max(args.runs, 1)
    )

    print("\nOptimized Portfolio Summary")
    print(tabulate(summary, headers=["Portfolio", "Return", "Risk", "Sharpe", "Max Drawdown"], tablefmt="grid"))

    print("\nOptimal Portfolio Stock Breakdown")
    print(tabulate(weight_rows, headers=["Ticker", "Stock Name", "Allocation", "Stock Return", "Stock Risk", "Stock Sharpe"], tablefmt="grid"))

    print(f"\nReport : {report_path}")
    print(f"Chart  : {chart_path}")
    print(f"Sharpe : {best['sharpe']:.4f}")


if __name__ == "__main__":
    main()