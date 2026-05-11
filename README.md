# DQN Stock Trading

Deep Q-Network family applied to single- and multi-asset trading with realistic
frictions, proper out-of-sample evaluation, and a Streamlit dashboard.

## Quickstart

```bash
source .venv/bin/activate
python -m src.train --ticker AAPL --agent rainbow --episodes 200
```

Open the showcase notebook:

```bash
jupyter lab notebooks/showcase.ipynb   # kernel: Python (clauder RL/ML)
```

Streamlit:

```bash
streamlit run app/dashboard.py
```

TensorBoard:

```bash
tensorboard --logdir runs
```

Hyperparameter search:

```bash
python scripts/tune.py --trials 30
```

Multi-seed sweep:

```bash
python scripts/run_seeds.py --agents dqn double_dueling rainbow --seeds 0 1 2 --episodes 200
```

## Layout

| Path | Purpose |
|------|---------|
| `src/data.py` | yfinance fetch + cache + chronological split + walk-forward |
| `src/features.py` | RSI, MACD, Bollinger, MAs, momentum, rolling-z normalization |
| `src/env.py` | `SingleAssetEnv`, `MultiAssetEnv` with costs, slippage, lookback, reward variants |
| `src/metrics.py` | CAGR, Sharpe, Sortino, max drawdown, Calmar, differential Sharpe |
| `src/networks.py` | MLP, Dueling, NoisyLinear, Distributional (C51) |
| `src/replay.py` | Uniform, n-step, prioritized (sum-tree) buffers |
| `src/agents.py` | DQN, Double, Dueling, PER, Noisy, Rainbow (`make_agent` factory) |
| `src/baselines.py` | Buy & hold, SMA crossover, RSI mean-revert, random |
| `src/plots.py` | Equity bands, drawdown, rolling Sharpe, monthly heatmap, action heatmap |
| `src/train.py` | Train loop, evaluation, TensorBoard, checkpoints, CLI |
| `scripts/run_seeds.py` | Multi-seed sweep with summary aggregation |
| `scripts/tune.py` | Optuna hyperparameter search (TPE + MedianPruner) |
| `app/dashboard.py` | Streamlit dashboard |
| `notebooks/showcase.ipynb` | End-to-end walkthrough |

## Reward functions

- `log`, `log(equity_t / equity_{t-1})`. Scale-invariant; default.
- `delta`, `equity_t - equity_{t-1}`. Original notebook style.
- `dsr`, Differential Sharpe Ratio (Moody & Saffell 1998). Risk-adjusted online.

## Frictions

- `cost_bps`, commission per traded notional.
- `slippage_bps`, adverse fill on every trade.
- Without these, RL strategies look profitable but evaporate live.
