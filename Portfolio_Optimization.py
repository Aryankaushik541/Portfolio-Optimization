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
    elite_size:      int   = 5
    tournament_size: int   = 7
    crossover_rate:  float = 0.90
    mutation_rate:   float = 0.12


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
    sigma = 0.15 * (1.0 - generation / max(cfg.generations - 1, 1)) + 0.01
    mask  = rng.random(len(w)) < cfg.mutation_rate
    if mask.any():
        w[mask] += rng.normal(0.0, sigma, mask.sum())
    if rng.random() < 0.15:
        i, j   = rng.choice(len(w), size=2, replace=False)
        amount = rng.uniform(0.0, min(max(float(w[i]), 0.0), 0.10))
        w[i]  -= amount; w[j]  += amount
    return repair_weights(w)


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
        top    = float(scores[order[0]])
        if top > best_score + 1e-8:
            best_score = top
            best_w     = population[order[0]].copy()
        history.append(best_score)
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

    weights = result["weights"]

    # portfolio-level metrics for this run
    pdr        = returns_matrix @ weights
    mean_r     = float(np.mean(pdr) * 100)
    min_r      = float(np.percentile(pdr, 5) * 100)
    max_r      = float(np.percentile(pdr, 95) * 100)
    std_r      = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)
    ann_ret    = float(np.mean(pdr) * TRADING_DAYS)
    ann_std    = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe     = (ann_ret - RISK_FREE_RATE) / ann_std if ann_std > 1e-12 else -np.inf

    # ── har run mein table: portfolio metrics + har stock ka allocation ──
    print(f"\n  {'─'*74}", flush=True)
    print(f"  Run {run_idx+1:>3}/{total}", flush=True)
    print(f"  {'─'*74}", flush=True)

    pm_table = [[f"{mean_r:.2f}%", f"{min_r:.2f}%", f"{max_r:.2f}%",
                 f"{std_r:.2f}%", f"{sharpe:.4f}"]]
    print(tabulate(pm_table,
                   headers=["Mean", "Min", "Max", "Std Dev", "Sharpe Ratio"],
                   tablefmt="simple"), flush=True)

    print(f"\n  Stock Allocations:", flush=True)
    stock_table = [[i+1, t, f"{weights[i]*100:.2f}%"] for i, t in enumerate(tickers)]
    print(tabulate(stock_table, headers=["#", "Ticker", "Allocation"], tablefmt="simple"), flush=True)

    return result, history


def format_run_summary(result, tickers, returns_matrix, run_idx, total):
    weights = result["weights"]
    pdr        = returns_matrix @ weights
    mean_r     = float(np.mean(pdr) * 100)
    min_r      = float(np.percentile(pdr, 5) * 100)
    max_r      = float(np.percentile(pdr, 95) * 100)
    std_r      = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)
    ann_ret    = float(np.mean(pdr) * TRADING_DAYS)
    ann_std    = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe     = (ann_ret - RISK_FREE_RATE) / ann_std if ann_std > 1e-12 else -np.inf

    pm_table = [[f"{mean_r:.2f}%", f"{min_r:.2f}%", f"{max_r:.2f}%",
                 f"{std_r:.2f}%", f"{sharpe:.4f}"]]
    stock_table = [[i+1, t, f"{weights[i]*100:.2f}%"] for i, t in enumerate(tickers)]

    return "\n".join([
        "",
        f"  {'-'*74}",
        f"  Run {run_idx+1:>3}/{total}",
        f"  {'-'*74}",
        tabulate(pm_table,
                 headers=["Mean", "Min", "Max", "Std Dev", "Sharpe Ratio"],
                 tablefmt="simple"),
        "",
        "  Stock Allocations:",
        tabulate(stock_table, headers=["#", "Ticker", "Allocation"], tablefmt="simple"),
    ])


def run_ga_multi(mean_daily, cov_daily, returns_matrix, cfg, runs, tickers, seed=SEED):
    import multiprocessing as mp

    workers = runs
    tasks = [
        (mean_daily, cov_daily, returns_matrix, cfg, seed + i, i, runs, tickers)
        for i in range(runs)
    ]

    if workers == 1:
        results = [_ga_worker(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_ga_worker, tasks)

    results.sort(key=lambda item: item[0])

    all_metrics   = []
    all_histories = []

    for run_idx, result, history in results:
        print(format_run_summary(result, tickers, returns_matrix, run_idx, runs), flush=True)
        all_metrics.append(result)
        all_histories.append(history)

    best_idx  = int(np.argmax([m["sharpe"] for m in all_metrics]))
    avg_hist  = np.mean(np.array(all_histories), axis=0)
    best_hist = np.array(all_histories[best_idx])
    return all_metrics[best_idx], avg_hist, best_hist, all_metrics, all_histories


# ── portfolio-level metrics ───────────────────────────────────────────────────
def get_portfolio_level_metrics(weights, returns_matrix, rf=RISK_FREE_RATE):
    pdr          = returns_matrix @ weights
    mean_return  = float(np.mean(pdr) * 100)
    min_return   = float(np.percentile(pdr, 5) * 100)
    max_return   = float(np.percentile(pdr, 95) * 100)
    std_return   = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)
    ann_ret      = float(np.mean(pdr) * TRADING_DAYS)
    ann_std      = float(pdr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_ratio = (ann_ret - rf) / ann_std if ann_std > 1e-12 else -np.inf
    return mean_return, min_return, max_return, std_return, float(sharpe_ratio)


# ── reporting ─────────────────────────────────────────────────────────────────
def stock_rows(tickers, returns, weights):
    returns_array = returns.to_numpy()
    rows = []
    for idx, ticker in enumerate(tickers):
        r      = returns_array[:, idx]
        sharpe = (np.mean(r) * TRADING_DAYS - RISK_FREE_RATE) / (np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append([ticker, weights[idx], np.mean(r)*100,
                     np.percentile(r,5)*100, np.percentile(r,95)*100,
                     np.std(r,ddof=1)*np.sqrt(TRADING_DAYS)*100, sharpe])
    rows.sort(key=lambda x: x[1], reverse=True)
    return [[i+1, t, COMPANY_NAMES.get(t,t), f"{a*100:.2f}%",
             f"{m:.2f}%", f"{mn:.2f}%", f"{mx:.2f}%", f"{s:.2f}%", f"{sh:.4f}"]
            for i,(t,a,m,mn,mx,s,sh) in enumerate(rows)]


def save_report(tickers, prices, returns, cfg, best, avg_hist, best_hist,
                all_metrics, all_histories, runs):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "optimized_ga_report.txt"
    chart_path  = REPORT_DIR / "optimized_ga_result.png"

    sharpes     = np.array([m["sharpe"] for m in all_metrics])
    weights     = best["weights"]
    mean_sharpe = float(np.mean(sharpes))
    max_sharpe  = float(np.max(sharpes))

    returns_matrix = returns.to_numpy()
    mean_ret, min_ret, max_ret, std_ret, portfolio_sharpe = get_portfolio_level_metrics(
        weights, returns_matrix)

    summary_headers = ["Mean", "Min", "Max", "Std Dev", "Sharpe Ratio"]
    summary = [[f"{mean_ret:.2f}%", f"{min_ret:.2f}%", f"{max_ret:.2f}%",
                f"{std_ret:.2f}%", f"{portfolio_sharpe:.4f}"]]

    w_rows = stock_rows(tickers, returns, weights)

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

        # ── BEST RUN RESULTS ──────────────────────────────────────────────────
        f.write("BEST RUN — PORTFOLIO METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(tabulate(summary, headers=summary_headers, tablefmt="github"))
        f.write("\n\n")
        f.write("BEST RUN — STOCK BREAKDOWN (all 10 stocks)\n")
        f.write("-" * 80 + "\n")
        f.write(tabulate(w_rows,
                         headers=["Serial No.","Ticker","Stock Name","Allocation","Mean","Min","Max","Std Dev","Sharpe"],
                         tablefmt="github"))
        f.write("\n")

    # ── charts (unchanged) ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ((ax1, ax2), (ax3, ax4)) = axes

    ax1.plot(avg_hist,  lw=2.5, label="Average Sharpe", color="#e41a1c", alpha=0.9)
    ax1.plot(best_hist, lw=1.8, ls="--", label="Best Run", color="#2ca02c", alpha=0.85)
    hist_arr = np.array(all_histories)
    if hist_arr.shape[0] > 1:
        ax1.fill_between(range(len(avg_hist)),
                         np.percentile(hist_arr, 25, axis=0),
                         np.percentile(hist_arr, 75, axis=0),
                         alpha=0.15, color="#e41a1c")
    ax1.set_title(f"GA Convergence\n({runs} runs, {cfg.generations} generations)",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=11); ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right"); ax1.grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.arange(len(tickers)))
    bars = ax2.bar(tickers, weights * 100, color=colors, edgecolor="black", linewidth=1.5)
    ax2.set_title(f"Optimal Portfolio Weights\nReturn={best['return']*100:.2f}%  |  "
                  f"Risk={best['risk']*100:.2f}%  |  Sharpe={best['sharpe']:.4f}",
                  fontsize=13, fontweight="bold")
    ax2.set_ylabel("Allocation (%)", fontsize=11)
    ax2.tick_params(axis="x", rotation=45)
    ax2.bar_label(bars, labels=[f"{w*100:.1f}%" if w > 0.001 else "" for w in weights],
                  padding=3, fontsize=9)
    ax2.set_ylim(0, max(weights * 100) * 1.25)
    ax2.grid(True, axis="y", alpha=0.3)

    best_idx_run  = int(np.argmax(sharpes))
    worst_idx_run = int(np.argmin(sharpes))
    run_labels  = ["Best Run", "Mean Run", "Worst Run"]
    run_returns = [all_metrics[best_idx_run]["return"]*100,
                   float(np.mean([m["return"] for m in all_metrics]))*100,
                   all_metrics[worst_idx_run]["return"]*100]
    run_risks   = [all_metrics[best_idx_run]["risk"]*100,
                   float(np.mean([m["risk"] for m in all_metrics]))*100,
                   all_metrics[worst_idx_run]["risk"]*100]
    run_sharpes_x20     = [sharpes[best_idx_run]*20, mean_sharpe*20, sharpes[worst_idx_run]*20]
    run_sharpes_actual  = [sharpes[best_idx_run], mean_sharpe, sharpes[worst_idx_run]]
    x_pos = np.arange(3); bw = 0.25
    ax3.bar(x_pos-bw, run_returns,      bw, label="Return (%)",  color="#4c72b0", edgecolor="black")
    ax3.bar(x_pos,    run_risks,         bw, label="Risk (%)",    color="#dd8452", edgecolor="black")
    ax3.bar(x_pos+bw, run_sharpes_x20,  bw, label="Sharpe x20",  color="#55a868", edgecolor="black")
    for i,(r,k,s) in enumerate(zip(run_returns, run_risks, run_sharpes_actual)):
        ax3.text(i-bw, r+0.5, f"{r:.2f}%", ha="center", fontsize=8)
        ax3.text(i,    k+0.5, f"{k:.2f}%", ha="center", fontsize=8)
        ax3.text(i+bw, run_sharpes_x20[i]+0.5, f"{s:.4f}", ha="center", fontsize=8)
    ax3.set_title("Run Performance Comparison\n(Best, Mean, Worst)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Value", fontsize=11)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(run_labels)
    ax3.legend(fontsize=9); ax3.grid(True, axis="y", alpha=0.3)

    spread = float(np.ptp(sharpes))
    if spread < 1e-6:
        ax4.bar([0], [len(sharpes)], width=0.5, color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=1.5)
        ax4.set_xticks([0]); ax4.set_xticklabels([f"{mean_sharpe:.4f}"])
        ax4.set_title(f"Sharpe Distribution\n({runs} runs, all converged to {mean_sharpe:.4f})",
                      fontsize=12, fontweight="bold")
        ax4.set_ylim(0, len(sharpes) * 1.2)
    else:
        bins = min(15, max(3, runs // 2))
        ax4.hist(sharpes, bins=bins, color="#2ca02c", alpha=0.72, edgecolor="black", linewidth=1.5)
        ax4.axvline(mean_sharpe, color="#e41a1c", ls="--", lw=2.5, label=f"Mean = {mean_sharpe:.4f}")
        ax4.axvline(max_sharpe,  color="#2ca02c", lw=2.5,           label=f"Best = {max_sharpe:.4f}")
        ax4.set_title(f"Sharpe Distribution\nacross {runs} Independent Runs",
                      fontsize=12, fontweight="bold")
        ax4.set_xlim(mean_sharpe - spread*1.8, mean_sharpe + spread*1.8)
        ax4.legend(fontsize=9)
    ax4.set_xlabel("Sharpe Ratio", fontsize=10); ax4.set_ylabel("Frequency", fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Optimized GA - Single Objective Portfolio Optimisation (Maximise Sharpe Ratio)\n"
        f"Data: S&P 500 Yahoo Finance {prices.index[0].date()} to {prices.index[-1].date()} | "
        f"Seed: {SEED} | RF = {RISK_FREE_RATE*100:.1f}%",
        fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return report_path, chart_path, summary, summary_headers, w_rows, \
           mean_ret, min_ret, max_ret, std_ret, portfolio_sharpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache",   action="store_true")
    parser.add_argument("--generations", type=int, default=GAConfig.generations)
    parser.add_argument("--population",  type=int, default=GAConfig.population_size)
    parser.add_argument("--runs",        type=int, default=30)
    args = parser.parse_args()

    cfg = GAConfig(population_size=args.population, generations=args.generations)

    print("\n" + "=" * 80)
    print("  OPTIMIZED GA — PORTFOLIO OPTIMISATION")
    print("  Objective: Maximise Sharpe Ratio")
    print("=" * 80)

    prices  = fetch_prices(TICKERS, START_DATE, END_DATE, use_cache=args.use_cache)
    returns = daily_returns(prices)
    tickers = list(returns.columns)

    # ── import data: ek line mein ─────────────────────────────────────────────
    print(f"\nData: {len(tickers)} stocks | {prices.index[0].date()} to {prices.index[-1].date()} "
          f"| {len(returns)} trading days | RF={RISK_FREE_RATE*100:.1f}% | Seed={SEED}")
    print(f"Stocks: {' | '.join(tickers)}\n")

    returns_matrix = returns.to_numpy()
    mean_daily     = returns_matrix.mean(axis=0)
    cov_daily      = np.cov(returns_matrix.T)

    print(f"Running GA ({args.runs} parallel runs, serial output) ...\n")
    best, avg_hist, best_hist, all_metrics, all_histories = run_ga_multi(
        mean_daily, cov_daily, returns_matrix, cfg, max(args.runs, 1), tickers)

    sharpes     = np.array([m["sharpe"] for m in all_metrics])
    mean_sharpe = float(np.mean(sharpes))

    report_path, chart_path, summary, summary_headers, w_rows, \
        mean_ret, min_ret, max_ret, std_ret, portfolio_sharpe = save_report(
            tickers, prices, returns, cfg, best, avg_hist, best_hist,
            all_metrics, all_histories, max(args.runs, 1))

    print("\n" + "=" * 80)
    print("BEST RUN — PORTFOLIO SUMMARY")
    print("=" * 80)
    print(tabulate([[f"{mean_ret:.2f}%", f"{min_ret:.2f}%", f"{max_ret:.2f}%",
                     f"{std_ret:.2f}%", f"{portfolio_sharpe:.4f}"]],
                   headers=summary_headers, tablefmt="grid"))

    print("\n" + "=" * 80)
    print("BEST RUN — STOCK BREAKDOWN")
    print("=" * 80)
    print(tabulate(w_rows,
                   headers=["Serial No.","Ticker","Stock Name","Allocation","Mean","Min","Max","Std Dev","Sharpe Ratio"],
                   tablefmt="grid"))

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Report : {report_path}")
    print(f"Charts : {chart_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
