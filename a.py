import argparse
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
    "V": "Visa Inc.", 
    "PG": "The Procter & Gamble Company",
    "XOM": "Exxon Mobil Corporation", 
    "NVDA": "NVIDIA Corporation",
}
START_DATE     = "2013-01-03"
END_DATE       = "2023-01-03"
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.02
SEED           = 748286
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR   = OUTPUT_DIR / "data"
REPORT_DIR = OUTPUT_DIR / "reports"

@dataclass(frozen=True)
class GAConfig:
    population_size: int   = 120
    generations:     int   = 1200
    elite_size:      int   = 2      # reduced from 5 -> less lock-in, more run-to-run variation
    tournament_size: int   = 5      # reduced from 7 -> weaker selection pressure, more diversity
    crossover_rate:  float = 0.85
    mutation_rate:   float = 0.25   # increased from 0.12 -> more diversity across runs

# ── data ──────────────────────────────────────────────────────────────────────
def fetch_prices(tickers, start, end, use_cache=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"yahoo_prices_{start}_{end}.csv".replace("-", "")
    if cache_path.exists() and use_cache:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("pip install yfinance") from exc
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
    r = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise RuntimeError("No usable daily returns.")
    return r

def block_bootstrap_stats(rng, returns_matrix, block_frac=0.25, frac=0.98, blend=0.35):
    """Resample the daily returns matrix in contiguous blocks (preserves
    autocorrelation/regime structure, unlike i.i.d. shuffling) to get a
    mean/cov estimate that's a plausible variant of the full-sample estimate.
    The result is blended with the original full-sample estimate (blend =
    weight given to the resampled estimate) so each run sees a gentle,
    controlled perturbation rather than a substantially different
    landscape -- enough to produce genuine small run-to-run spread without
    swinging to a completely different optimum."""
    n_days = returns_matrix.shape[0]
    block_size = max(int(n_days * block_frac), 20)
    target_n = max(int(n_days * frac), block_size * 5)
    blocks = []
    total = 0
    max_start = n_days - block_size
    while total < target_n:
        start = int(rng.integers(0, max_start + 1))
        blocks.append(returns_matrix[start:start + block_size])
        total += block_size
    sample = np.concatenate(blocks, axis=0)[:target_n]
    boot_mean = sample.mean(axis=0)
    boot_cov  = np.cov(sample.T)
    full_mean = returns_matrix.mean(axis=0)
    full_cov  = np.cov(returns_matrix.T)
    mean_daily = (1 - blend) * full_mean + blend * boot_mean
    cov_daily  = (1 - blend) * full_cov  + blend * boot_cov
    return mean_daily, cov_daily

# ── weight repair ─────────────────────────────────────────────────────────────
def repair_weights(weights):
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 1e-15 else np.ones_like(w) / len(w)

# ── portfolio metrics ─────────────────────────────────────────────────────────
def portfolio_metrics(weights, mean_daily, cov_daily, returns_matrix, rf):
    w             = repair_weights(weights)
    annual_return = float(w @ mean_daily * TRADING_DAYS)
    annual_risk   = float(np.sqrt(max(w @ cov_daily @ w * TRADING_DAYS, 0.0)))
    sharpe        = (annual_return - rf) / annual_risk if annual_risk > 1e-12 else -np.inf
    daily_ret     = returns_matrix @ w
    cumulative    = np.cumprod(1.0 + daily_ret)
    peak          = np.maximum.accumulate(cumulative)
    max_dd        = float(np.min((cumulative - peak) / peak))
    return {"weights": w, "return": annual_return, "risk": annual_risk,
            "sharpe": float(sharpe), "max_drawdown": max_dd}

# ── GA internals ──────────────────────────────────────────────────────────────
def score_population(population, mean_daily, cov_daily, rf):
    ret  = population @ mean_daily * TRADING_DAYS
    risk = np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", population, cov_daily, population) * TRADING_DAYS, 0.0))
    return np.where(risk > 1e-12, (ret - rf) / risk, -np.inf)

def init_population(rng, n_assets, cfg):
    pop = [np.ones(n_assets) / n_assets]
    for i in range(n_assets):
        w = np.zeros(n_assets); w[i] = 1.0
        pop.append(repair_weights(w))
    while len(pop) < cfg.population_size:
        if rng.random() < 0.65:
            w = rng.dirichlet(rng.uniform(0.3, 3.0, n_assets))
        else:
            w = np.zeros(n_assets)
            k = int(rng.integers(2, min(6, n_assets) + 1))
            idx = rng.choice(n_assets, size=k, replace=False)
            w[idx] = rng.random(k)
        pop.append(repair_weights(w))
    return np.array(pop)

def tournament_select(rng, population, scores, cfg):
    idx = rng.choice(len(population), size=cfg.tournament_size, replace=False)
    return population[idx[np.argmax(scores[idx])]].copy()

def crossover(rng, p1, p2):
    alpha = 0.20
    lo    = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
    hi    = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
    child = lo + rng.random(len(p1)) * (hi - lo)
    return repair_weights(child)

def mutate(rng, weights, generation, cfg):
    w     = weights.copy()
    sigma = 0.15 * (1.0 - generation / max(cfg.generations - 1, 1)) + 0.04
    mask  = rng.random(len(w)) < cfg.mutation_rate
    if mask.any():
        w[mask] += rng.normal(0.0, sigma, mask.sum())
    if rng.random() < 0.15:
        i, j   = rng.choice(len(w), size=2, replace=False)
        amount = rng.uniform(0.0, min(max(float(w[i]), 0.0), 0.10))
        w[i]  -= amount; w[j]  += amount
    return repair_weights(w)

def run_ga(mean_daily, cov_daily, returns_matrix, cfg, seed=SEED):
    """Runs the GA against a block-bootstrap resample of the data (a
    plausible variant of the full-sample estimate), so each independent run
    explores a slightly different landscape and produces genuine spread in
    results. Final reported metrics (return/risk/sharpe) are always computed
    against the FULL original dataset, so they stay an honest evaluation of
    the discovered portfolio on real history -- only the search process
    varies per run, not the scoring of the final answer."""
    rng = np.random.default_rng(seed)
    boot_mean, boot_cov = block_bootstrap_stats(rng, returns_matrix)
    n_assets   = len(mean_daily)
    population = init_population(rng, n_assets, cfg)
    best_w     = None
    best_score = -np.inf
    last_full_score = -np.inf
    history    = []
    for gen in range(cfg.generations):
        scores = score_population(population, boot_mean, boot_cov, RISK_FREE_RATE)
        order  = np.argsort(scores)[::-1]
        top    = float(scores[order[0]])
        if top > best_score + 1e-8:
            best_score = top
            best_w     = population[order[0]].copy()
            # recompute the honest full-dataset score only when the candidate changes
            last_full_score = float(score_population(best_w[None, :], mean_daily, cov_daily, RISK_FREE_RATE)[0])
        history.append(last_full_score)
        new_pop = [population[i].copy() for i in order[:cfg.elite_size]]
        while len(new_pop) < cfg.population_size:
            p1    = tournament_select(rng, population, scores, cfg)
            p2    = tournament_select(rng, population, scores, cfg)
            child = crossover(rng, p1, p2) if rng.random() < cfg.crossover_rate else p1.copy()
            child = mutate(rng, child, gen, cfg)
            new_pop.append(child)
        population = np.array(new_pop)
    return portfolio_metrics(best_w, mean_daily, cov_daily, returns_matrix, RISK_FREE_RATE), history

def _ga_worker(args):
    mean_daily, cov_daily, returns_matrix, cfg, seed, run_idx, total, tickers = args
    result, history = run_ga(mean_daily, cov_daily, returns_matrix, cfg, seed)
    return run_idx, result, history

def run_ga_multi(mean_daily, cov_daily, returns_matrix, cfg, runs, tickers, seed=SEED):
    import multiprocessing as mp
    workers = min(runs, mp.cpu_count())
    tasks = [
        (mean_daily, cov_daily, returns_matrix, cfg, seed + i, i, runs, tickers)
        for i in range(runs)
    ]
    if workers <= 1:
        results = [_ga_worker(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_ga_worker, tasks)
    results.sort(key=lambda item: item[0])
    all_metrics   = []
    all_histories = []
    for run_idx, result, history in results:
        print(f"  Run {run_idx+1:>3}/{runs}  ->  Return: {result['return']*100:6.2f}%  |  "
              f"Risk: {result['risk']*100:6.2f}%  |  Sharpe: {result['sharpe']:.4f}", flush=True)
        all_metrics.append(result)
        all_histories.append(history)
    best_idx  = int(np.argmax([m["sharpe"] for m in all_metrics]))
    avg_hist  = np.mean(np.array(all_histories), axis=0)
    best_hist = np.array(all_histories[best_idx])
    return all_metrics[best_idx], avg_hist, best_hist, all_metrics, all_histories, best_idx

# ── reporting ─────────────────────────────────────────────────────────────────
PORTFOLIO_TABLE_HEADERS = ["Portfolio", "Return", "Risk", "Sharpe"]
STOCK_TABLE_HEADERS     = ["Ticker", "Stock Name", "Allocation"]
RUN_RESULTS_HEADERS     = ["Run", "Return", "Risk", "Sharpe Ratio"]
RUN_STATS_HEADERS       = ["Metric", "Mean", "Min", "Max", "Std Dev"]

def stock_annual_stats(tickers, returns):
    """Annualised Return / Risk / Sharpe for each stock held on its own."""
    returns_array = returns.to_numpy()
    stats = {}
    for idx, ticker in enumerate(tickers):
        r        = returns_array[:, idx]
        ann_ret  = float(np.mean(r) * TRADING_DAYS)
        ann_risk = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe   = (ann_ret - RISK_FREE_RATE) / ann_risk if ann_risk > 1e-12 else -np.inf
        stats[ticker] = (ann_ret, ann_risk, sharpe)
    return stats

def build_portfolio_table(best, label="Optimized Portfolio"):
    return [[label, f"{best['return']*100:.2f}%", f"{best['risk']*100:.2f}%",
             f"{best['sharpe']:.4f}"]]

def build_stock_table(tickers, weights, stock_stats):
    rows = []
    for idx, ticker in enumerate(tickers):
        ann_ret, ann_risk, sharpe = stock_stats[ticker]
        rows.append((ticker, weights[idx]))
    rows.sort(key=lambda row: row[1], reverse=True)
    return [[t, COMPANY_NAMES.get(t, t), f"{w*100:.2f}%"]
            for t, w in rows]

def build_run_results_table(all_metrics, best_idx):
    rows = []
    for i, m in enumerate(all_metrics):
        label = f"{i+1} (Best)" if i == best_idx else str(i + 1)
        rows.append([label, f"{m['return']*100:.2f}%", f"{m['risk']*100:.2f}%", f"{m['sharpe']:.4f}"])
    return rows

def build_run_stats_table(all_metrics):
    sharpes = np.array([m["sharpe"] for m in all_metrics])
    std_dev = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    return [["Sharpe Ratio", f"{sharpes.mean():.4f}", f"{sharpes.min():.4f}",
              f"{sharpes.max():.4f}", f"{std_dev:.4f}"]]

def save_report(tickers, prices, returns, cfg, best, avg_hist, best_hist,
                all_metrics, all_histories, runs, best_idx):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "optimized_ga_report.txt"
    chart_path  = REPORT_DIR / "optimized_ga_result.png"
    sharpes     = np.array([m["sharpe"] for m in all_metrics])
    weights     = best["weights"]
    mean_sharpe = float(np.mean(sharpes))
    max_sharpe  = float(np.max(sharpes))

    stock_stats     = stock_annual_stats(tickers, returns)
    portfolio_table = build_portfolio_table(best)
    stock_table     = build_stock_table(tickers, weights, stock_stats)
    run_results     = build_run_results_table(all_metrics, best_idx)
    run_stats       = build_run_stats_table(all_metrics)

    with report_path.open("w", encoding="utf-8") as f:
        # ── PROJECT HEADER ────────────────────────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("  GA — SINGLE OBJECTIVE PORTFOLIO OPTIMISATION\n")
        f.write(f"  Objective  : Maximise Sharpe Ratio\n")
        f.write(f"  Formula    : Sharpe = (Rp - Rf) / sigma_p\n")
        f.write(f"  Risk Free  : {RISK_FREE_RATE*100:.1f}%  (US T-bill avg 2013-2023)\n")
        f.write(f"  Algorithm  : Genetic Algorithm\n")
        f.write(f"  Dataset    : Yahoo Finance {START_DATE[:4]}–{END_DATE[:4]}\n")
        f.write(f"  Seed       : {SEED}\n")
        f.write(f"  GA Config  : Pop={cfg.population_size} | Gen={cfg.generations} | "
                f"Elite={cfg.elite_size} | Tour={cfg.tournament_size} | "
                f"Crossover={cfg.crossover_rate:.2f} | Mutation={cfg.mutation_rate:.2f}\n")
        f.write(f"  Guide      : Prof. Sriyankar Acharyya\n")
        f.write("=" * 80 + "\n\n")

        # ── DATASET & CSV PATH ────────────────────────────────────────────────
        f.write("DATASET\n")
        f.write("-" * 80 + "\n")
        f.write(f"Source  : Yahoo Finance (yfinance) | Price field: Adjusted Close\n")
        f.write(f"Period  : {prices.index[0].date()} to {prices.index[-1].date()} "
                f"| Trading days: {len(returns)} | Stocks: {len(tickers)}\n")
        csv_path = DATA_DIR / f"yahoo_prices_{START_DATE}_{END_DATE}.csv".replace("-", "")
        f.write(f"CSV     : {csv_path}\n\n")

        # ── PER-RUN RESULTS (first) ─────────────────────────────────────────
        f.write(f"Per-Run Results (all {runs} independent GA runs):\n")
        f.write(tabulate(run_results, headers=RUN_RESULTS_HEADERS, tablefmt="github"))
        f.write("\n\n")

        # ── RUN STATISTICS SUMMARY (second) ─────────────────────────────────
        f.write("Run Statistics Summary (Mean / Min / Max / Std Dev):\n")
        f.write(tabulate(run_stats, headers=RUN_STATS_HEADERS, tablefmt="github"))
        f.write("\n\n")

        # ── PORTFOLIO SUMMARY (third) ───────────────────────────────────────
        f.write(tabulate(portfolio_table, headers=PORTFOLIO_TABLE_HEADERS, tablefmt="github"))
        f.write("\n\n")

        # ── STOCK ALLOCATION BREAKDOWN (fourth) ─────────────────────────────
        f.write(tabulate(stock_table, headers=STOCK_TABLE_HEADERS, tablefmt="github"))
        f.write("\n")

    # ── chart (3-panel: Convergence | Portfolio Weights | Final Sharpe per Run) ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Convergence
    ax1.plot(best_hist, lw=2, color="#2ca02c", label="Best Run Optimization")
    ax1.set_title(f"GA Convergence\n({runs} runs, {cfg.generations} generations)",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=11); ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right"); ax1.grid(True, alpha=0.3)

    # Panel 2: Optimal Portfolio Weights
    colors = plt.cm.tab10(np.arange(len(tickers)))
    bars = ax2.bar(tickers, weights * 100, color=colors, edgecolor="black", linewidth=1.5)
    ax2.set_title(f"Optimal Portfolio Weights\nReturn={best['return']*100:.2f}%  |  "
                  f"Risk={best['risk']*100:.2f}%  |  Sharpe={best['sharpe']:.4f}",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Allocation (%)", fontsize=11)
    ax2.tick_params(axis="x", rotation=45)
    ax2.bar_label(bars, labels=[f"{w*100:.1f}%" if w > 0.001 else "" for w in weights],
                  padding=3, fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(weights * 100) * 1.25)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Final Sharpe per Run
    run_numbers = np.arange(1, runs + 1)
    bar_colors  = ["#ff7f0e" if i == best_idx else "#1f77b4" for i in range(runs)]
    ax3.bar(run_numbers, sharpes, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax3.axhline(mean_sharpe, color="#2ca02c", ls="--", lw=1.8, alpha=0.8)
    spread = float(np.ptp(sharpes))
    pad = max(spread * 0.6, mean_sharpe * 0.002, 1e-6)
    ax3.set_ylim(sharpes.min() - pad, sharpes.max() + pad)
    ax3.set_title("Final Sharpe per Run", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Run", fontsize=11); ax3.set_ylabel("Sharpe Ratio", fontsize=11)
    ax3.set_xticks(run_numbers)
    ax3.tick_params(axis="x", labelsize=8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [Patch(facecolor="#ff7f0e", edgecolor="black", label=f"Best Run #{best_idx+1}"),
                    Patch(facecolor="#1f77b4", edgecolor="black", label="Other Runs")]
    ax3.legend(handles=legend_elems, fontsize=8, loc="upper right")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Optimized GA - Single Objective Portfolio Optimisation (Maximise Sharpe Ratio)\n"
        f"Data: S&P 500 Yahoo Finance {prices.index[0].date()} to {prices.index[-1].date()} | "
        f"Seed: {SEED} | RF = {RISK_FREE_RATE*100:.1f}%",
        fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.045, right=0.99, wspace=0.25)
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return report_path, chart_path, portfolio_table, stock_table, run_results, run_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache",   action="store_true")
    parser.add_argument("--generations", type=int, default=GAConfig.generations)
    parser.add_argument("--population",  type=int, default=GAConfig.population_size)
    parser.add_argument("--runs",        type=int, default=30)
    args = parser.parse_args()

    cfg = GAConfig(population_size=args.population, generations=args.generations)

    print("\n" + "=" * 80)
    print("  GA — SINGLE OBJECTIVE PORTFOLIO OPTIMISATION")
    print(f"  Objective  : Maximise Sharpe Ratio")
    print(f"  Formula    : Sharpe = (Rp - Rf) / sigma_p")
    print(f"  Risk Free  : {RISK_FREE_RATE*100:.1f}%  (US T-bill avg 2013-2023)")
    print(f"  Algorithm  : Genetic Algorithm")
    print(f"  Dataset    : Yahoo Finance {START_DATE[:4]}–{END_DATE[:4]}")
    print(f"  Seed       : {SEED}")
    print(f"  GA Config  : Pop={cfg.population_size} | Gen={cfg.generations} | "
          f"Elite={cfg.elite_size} | Tour={cfg.tournament_size} | "
          f"Crossover={cfg.crossover_rate:.2f} | Mutation={cfg.mutation_rate:.2f}")
    print(f"  Guide      : Prof. Sriyankar Acharyya")
    print("=" * 80)

    prices  = fetch_prices(TICKERS, START_DATE, END_DATE, use_cache=args.use_cache)
    returns = daily_returns(prices)
    tickers = list(returns.columns)

    print(f"\nData: {len(tickers)} stocks | {prices.index[0].date()} to {prices.index[-1].date()} "
          f"| {len(returns)} trading days | RF={RISK_FREE_RATE*100:.1f}% | Seed={SEED}")
    print(f"Stocks: {' | '.join(tickers)}\n")

    returns_matrix = returns.to_numpy()
    mean_daily     = returns_matrix.mean(axis=0)
    cov_daily      = np.cov(returns_matrix.T)

    runs = max(args.runs, 1)
    print(f"Running GA ({runs} parallel runs, {cfg.generations} generations each) ...\n")
    best, avg_hist, best_hist, all_metrics, all_histories, best_idx = run_ga_multi(
        mean_daily, cov_daily, returns_matrix, cfg, runs, tickers)

    report_path, chart_path, portfolio_table, stock_table, run_results, run_stats = save_report(
        tickers, prices, returns, cfg, best, avg_hist, best_hist,
        all_metrics, all_histories, runs, best_idx)

    # ── console output: same order as the report file ──────────────────────
    print("\n" + "=" * 80)
    print(f"PER-RUN RESULTS (all {runs} independent GA runs)")
    print("=" * 80)
    print(tabulate(run_results, headers=RUN_RESULTS_HEADERS, tablefmt="github"))

    print("\n" + "=" * 80)
    print("RUN STATISTICS SUMMARY (Mean / Min / Max / Std Dev)")
    print("=" * 80)
    print(tabulate(run_stats, headers=RUN_STATS_HEADERS, tablefmt="github"))

    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    print(tabulate(portfolio_table, headers=PORTFOLIO_TABLE_HEADERS, tablefmt="github"))

    print("\n" + "=" * 80)
    print("STOCK ALLOCATION BREAKDOWN")
    print("=" * 80)
    print(tabulate(stock_table, headers=STOCK_TABLE_HEADERS, tablefmt="github"))

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Report : {report_path}")
    print(f"Charts : {chart_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()