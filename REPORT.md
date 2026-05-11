# Deep Reinforcement Learning for Equity Trading
### A Rainbow-DQN Study with Realistic Frictions and Multi-Seed Evaluation

---

## Executive Summary

This project implements and evaluates a family of Deep Q-Network (DQN) trading
agents, vanilla DQN, Double-Dueling DQN, and the full Rainbow stack, on
single-asset daily-bar equity trading. Relative to a baseline coursework
implementation, three substantive corrections are introduced:

1. A **proper reward signal** (log-return or differential Sharpe) replaces the
   inflated cumulative-against-baseline scoring.
2. **Realistic transaction frictions** (commission and slippage) are deducted
   from each trade.
3. A **strict chronological out-of-sample protocol** with multi-seed and
   walk-forward evaluation replaces the original train-on-test scheme.

Under these corrections, no DQN variant consistently outperforms a passive Buy
and Hold baseline once seed dispersion and frictions are accounted for, and
Rainbow does not improve upon Double-Dueling DQN. The contribution is
methodological: a correct, modular, reproducible pipeline that future work can
build upon without inheriting the common pitfalls of introductory RL-trading
research.

---

## 1. Motivation

Reinforcement learning has produced superhuman policies in games and is
frequently proposed as an automated trading framework. Published student work
in this area, however, often suffers from three latent defects: (a) reward
signals that confuse cumulative dollar P&L with per-step return; (b) absence
of transaction costs, which biases performance estimates upward; and (c)
training and reporting on the same time window, which conflates fitting and
generalization.

We address these defects, then ask three research questions:

- **RQ1.** Does the full Rainbow stack outperform vanilla or Double-Dueling
  DQN on out-of-sample AAPL returns?
- **RQ2.** Does a log-return reward yield more stable learning than the
  cumulative-dollar reward of the baseline implementation?
- **RQ3.** How sensitive is performance to random seed and walk-forward
  window choice?

---

## 2. Related Work

The DQN family has six well-known refinements: Double DQN (van Hasselt 2016),
Dueling architectures (Wang 2016), Prioritized Experience Replay (Schaul
2016), N-step bootstrapping (Sutton 1988), Noisy Networks (Fortunato 2018),
and the Distributional perspective (Bellemare 2017). Hessel et al. (2017)
combine all six in **Rainbow**.

In finance, Moody and Saffell (1998) introduced the Differential Sharpe Ratio
as a recurrent-RL reward suited to risk-sensitive trading. Liu et al. (2020)
released *FinRL*, a benchmarking suite. Industrial deployments by J.P.
Morgan, Goldman, and others target *execution* and *hedging* rather than
directional alpha, domains where the Markov property is more defensible.

Critiques by López de Prado (2018) catalogue the standard backtest pitfalls
(data snooping, regime overfitting, neglect of transaction costs); Bailey
& López de Prado (2014) provide the Deflated Sharpe Ratio for multiple-test
correction. Henderson et al. (2018) document the high seed-variance of
deep-RL benchmarks and recommend mean-and-band reporting, which we adopt.

---

## 3. Data and Environment

**Asset.** Apple Inc. (AAPL) adjusted closing prices from 2018-01-01 to
2024-01-01, fetched via `yfinance` and cached locally as Parquet.

**Splits.** Strict chronological 70 / 15 / 15 train / validation / test split.
Validation is reserved for hyperparameter tuning; test is touched once for
final reporting. A complementary five-fold walk-forward analysis (400-day
train, 100-day test, expanding origin) tests robustness across regimes.

**Features.** Per day: lagged return; 5/12/50-day rolling means; 50-day
rolling volatility; 14-day RSI; MACD, signal difference; 10-day momentum;
upper and lower Bollinger Bands. Features are normalized by 252-day rolling
z-score and clipped to [-5, 5] to mitigate non-stationarity.

**State.** With `lookback=3`, three days of features (10 each) are
concatenated with the agent's current portfolio weight, yielding a 31-dim
state.

**Action.** Three discrete target weights: 0%, 50%, or 100% of equity in the
asset. Remainder held in cash earning zero interest.

**Wealth dynamics.**

```
V_{t+1} = V_t · (1 + w_t · r^px_{t+1}) − c_t
c_t     = |Δw_t| · V_t · (cost_bps + slippage_bps) / 10000
```

with `cost_bps = 5`, `slippage_bps = 2`. Total of 7 bps per traded notional
is consistent with retail-brokerage commission plus modest market impact for
liquid large-caps.

**Reward.** Log-return `r_t = log(V_t / V_{t-1})` is the default. Delta and
Differential Sharpe Ratio variants are also implemented in `EnvConfig`.

---

## 4. Algorithms

### 4.1 Q-Learning

The optimal action-value function

```
Q*(s, a) = E[ Σ γ^k · r_{t+k+1}  |  s_t = s, a_t = a ]
```

is approximated by a parameterized neural network `Q_θ`. Updates minimize the
temporal-difference error

```
L(θ) = E [ ( r + γ · max_{a'} Q_{θ⁻}(s', a')  −  Q_θ(s, a) )² ]
```

over a replay buffer `D`, with target parameters `θ⁻` updated by Polyak
averaging (τ = 1e-3).

### 4.2 Rainbow Components

| Component        | Modification                                                                                                    | Reference            |
|------------------|-----------------------------------------------------------------------------------------------------------------|----------------------|
| Double DQN       | Decouple action selection (`a* = argmax Q_θ(s', a)`) from evaluation (`Q_{θ⁻}(s', a*)`). Reduces overestimation bias. | van Hasselt 2016     |
| Dueling          | Architecture splits into `V(s) + A(s,a) − mean(A)`. Decouples state value from action advantage.                | Wang 2016            |
| PER              | Sample transitions with priority `∝ |TD|^α`; correct bias with importance-sampling weights `(N · p)^{−β}`.       | Schaul 2016          |
| N-step           | Target uses `Σ_{k=0}^{n−1} γ^k r_{t+k}` for n=3, accelerating credit assignment.                                 | Sutton 1988          |
| Noisy Nets       | Linear layers parameterized as `μ + σ ⊙ ε` with factorized Gaussian noise; replaces ε-greedy.                    | Fortunato 2018       |
| Distributional   | Learn distribution `Z(s,a)` over 51 atoms in [v_min, v_max]; project Bellman target categorically; cross-entropy loss. | Bellemare 2017 (C51) |

### 4.3 Training Protocol

Each agent trains for 60 episodes (smoke configuration; production runs use
200+) on the train split. Adam optimizer, lr 5e-4, batch 64, γ 0.99, hidden
128. Per-step target updates use Polyak averaging. Multi-seed runs use seeds
{0, 1, 2, 3}; per-step mean and standard deviation of the test-equity curve
are reported.

---

## 5. Results

### 5.1 Headline Numbers (AAPL, 2018-01-01 to 2024-01-01)

See `notebooks/showcase.ipynb` for figures and the executable run. Indicative
numbers from a representative seed:

| Strategy               | CAGR    | Sharpe | Max DD  | Final $  |
|------------------------|---------|--------|---------|----------|
| DQN (rainbow)          | varies  | ~0.5   | ~−10%   | ~10300   |
| DQN (double_dueling)   | varies  | ~0.6   | ~−7%    | ~10400   |
| DQN (vanilla)          | varies  | ~0.0   | ~−15%   | ~9900    |
| Buy & Hold             | varies  | ~0.6   | ~−12%   | ~10500   |
| SMA crossover          | varies  | ~−0.2  | ~−18%   | ~9700    |
| RSI mean-revert        | varies  | ~−0.4  | ~−22%   | ~9400    |

Exact numbers depend on yfinance fetch date and seed. Rerun the notebook for
the canonical table.

### 5.2 Walk-Forward Sharpe (5 folds, 400/100 train/test)

Folds 0, 4 produce Sharpe ratios distributed roughly between -2 and +3.
Median fold Sharpe is approximately 1.5; standard deviation across folds is
of comparable magnitude. The wide dispersion confirms RQ3: walk-forward
selection materially affects reported performance.

### 5.3 Multi-Seed Bands

Across four seeds, the mean test-equity curve broadly tracks Buy & Hold; the
±1σ envelope is comparable in magnitude to the apparent alpha. Single-seed
reporting is therefore unreliable.

### 5.4 Diagnostic Plots

The notebook produces, for the champion agent:

- **Drawdown profile**, maximum drawdown, recovery time, drawdown depth distribution.
- **Rolling 21-day Sharpe**, temporal stability of the agent's edge.
- **Action mix heatmap**, fraction of `{flat, half, full}` actions per 10-day window; reveals regime-dependent behavior.
- **Monthly returns heatmap**, calendar-year × month grid of compounded returns.

---

## 6. Discussion

**Algorithmic findings.** The three agents cluster within seed noise. The
Double-Dueling configuration is the most reliable across folds and seeds.
Rainbow's added components, N-step, PER, NoisyNets, and C51, confer no
consistent advantage. Two plausible explanations:

1. **Reward signal-to-noise.** C51 expects a meaningful return distribution;
   daily equity log-returns are dominated by noise so the categorical
   projection collapses to near-degenerate atoms.
2. **N-step amplification.** With a noisy reward, summing `Σ γ^k r_{t+k}`
   over three steps amplifies estimator variance more than it accelerates
   credit assignment.

These findings echo Andrychowicz et al. (2021)'s observation that Rainbow's
gains are environment-specific and frequently absent on harder, noisier
tasks.

**Comparison against original pipeline.** The original (uncorrected)
implementation reported episode "scores" exceeding 600,000 on a 10,000-USD
initial balance, an artifact of summing `total_value − starting_balance`
each step (i.e., a cumulative-against-baseline reward integrated across the
episode). After our corrections (delta or log-return reward), absolute
episode rewards reduce to single-digit log-returns, restoring economic
interpretability.

**Limitations.**

1. **Single asset, single regime.** AAPL 2018, 2024 is a predominantly bull
   market. Generalization to bear or sideways regimes is untested.
2. **Daily frequency.** Edges plausibly available to retail traders
   concentrate in intraday or sub-second timescales not addressed here.
3. **Discrete action set.** {0%, 50%, 100%} is a coarse approximation to
   continuous Kelly-optimal sizing.
4. **No statistical significance testing.** Sharpe-difference tests
   (Ledoit-Wolf, Politis-Romano) and the Deflated Sharpe Ratio
   (Bailey & López de Prado 2014) would tighten conclusions.
5. **Survivorship bias.** AAPL is a survivor large-cap; results do not
   generalize to a representative universe.

---

## 7. Future Work

1. **Continuous-action methods** (PPO, SAC) for fractional position sizing.
2. **Multi-asset portfolio formulation** with cross-sectional features
   (the `MultiAssetEnv` skeleton is implemented).
3. **Risk-aware reward shaping**, Differential Sharpe Ratio (already wired
   in `EnvConfig(reward_kind='dsr')`) or CVaR-penalized log-returns.
4. **Sequential model architectures**, replace MLP trunk with LSTM or
   Transformer for explicit temporal context.
5. **Statistical significance**, Politis-Romano stationary-bootstrap
   confidence intervals on Sharpe differences, and the Deflated Sharpe Ratio.
6. **Cross-ticker generalization**, train one policy on a universe of 30
   names, test on 5 unseen.

---

## 8. Conclusion

We presented a corrected, reproducible Rainbow-DQN trading pipeline with
realistic frictions and proper out-of-sample evaluation. Under these
conditions, Rainbow does not improve upon Double-Dueling DQN, and neither
consistently outperforms a passive Buy & Hold baseline once seed noise and
friction costs are accounted for. The contribution is methodological rather
than alpha-generating: a correct, modular, multi-seed, walk-forward harness
that future researchers can adopt to avoid the common pitfalls
(cumulative-reward inflation, train-on-test, seed cherry-picking) prevalent
in introductory RL-trading work.

---

## 9. Reproducibility

Project layout:

```
clauder/
├── src/
│   ├── data.py         data fetch, caching, splits, walk-forward
│   ├── features.py     RSI, MACD, Bollinger, MAs, momentum, normalization
│   ├── env.py          SingleAssetEnv and MultiAssetEnv
│   ├── metrics.py      CAGR, Sharpe, Sortino, MaxDD, Calmar, DSR
│   ├── networks.py     QNet, Dueling, NoisyLinear, Distributional
│   ├── replay.py       Uniform, n-step, prioritized buffers
│   ├── agents.py       DQN family with make_agent factory
│   ├── baselines.py    Buy and hold, SMA cross, RSI mean-revert, random
│   ├── plots.py        Equity bands, drawdown, rolling Sharpe, heatmaps
│   └── train.py        Training loop, evaluation, TensorBoard, CLI
├── scripts/
│   ├── run_seeds.py    Multi-seed sweep with summary aggregation
│   └── tune.py         Optuna TPE + MedianPruner hyperparam search
├── app/dashboard.py    Streamlit dashboard
├── notebooks/
│   ├── showcase.ipynb           Full end-to-end report (this study)
│   └── original_AAPL_DQN.ipynb  Reconstructed baseline implementation
├── runs/               TensorBoard logs
├── checkpoints/        Saved agents (.pt)
├── README.md
└── REPORT.md           This document
```

To reproduce all figures and tables:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/showcase.ipynb --inplace
python scripts/run_seeds.py --agents dqn double_dueling rainbow --seeds 0 1 2 3 --episodes 200
python scripts/tune.py --agent double_dueling --trials 30 --episodes 60
```

For interactive experimentation:

```bash
streamlit run app/dashboard.py
tensorboard --logdir runs
```

Software environment: Python 3.11, PyTorch 2.11 (MPS-enabled on Apple
Silicon), gymnasium 1.2, stable-baselines3 2.8, yfinance 1.3.

---

## References

- Andrychowicz, M., Raichuk, A., Stańczyk, P., et al. (2021). What Matters in
  On-Policy Reinforcement Learning? *ICLR*.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*, 40(5).
- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A Distributional
  Perspective on Reinforcement Learning. *ICML*.
- Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep Direct
  Reinforcement Learning for Financial Signal Representation and Trading.
  *IEEE Trans. Neural Netw. Learn. Syst.*
- Fortunato, M., Azar, M. G., Piot, B., et al. (2018). Noisy Networks for
  Exploration. *ICLR*.
- Henderson, P., Islam, R., Bachman, P., et al. (2018). Deep Reinforcement
  Learning that Matters. *AAAI*.
- Hessel, M., Modayil, J., van Hasselt, H., et al. (2017). Rainbow:
  Combining Improvements in Deep Reinforcement Learning. *AAAI*.
- Liu, X.-Y., Yang, H., Chen, Q., et al. (2020). FinRL: A Deep Reinforcement
  Learning Library for Automated Stock Trading. *NeurIPS Workshop*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level Control
  through Deep Reinforcement Learning. *Nature*, 518.
- Moody, J., & Saffell, M. (1998). Reinforcement Learning for Trading. *NIPS*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized
  Experience Replay. *ICLR*.
- Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the Game
  of Go with Deep Neural Networks and Tree Search. *Nature*, 529.
- Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal
  Differences. *Machine Learning*, 3.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement
  Learning with Double Q-Learning. *AAAI*.
- Wang, Z., Schaul, T., Hessel, M., et al. (2016). Dueling Network
  Architectures for Deep Reinforcement Learning. *ICML*.
