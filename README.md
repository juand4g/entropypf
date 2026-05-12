# entropypf

Fast entropy-based period finder for unevenly sampled astronomical time series.

Implements the minimum Shannon entropy method introduced by Cincotta, Méndez & Núñez (1995, ApJ 449, 231) for detecting periodicity in variable-star light curves and other time series with irregular sampling.

## Installation

```bash
pip install entropypf
```

## Quick start

```python
import numpy as np
from entropypf import find_best_period, get_entropies

# t: observation times (days), u: magnitudes
periods, entropies = find_best_period(t, u, p0=0.1, p1=10.0, p_num=5000)
print(f"Best period: {periods[0]:.4f} days  (entropy={entropies[0]:.4f})")
```

To inspect the full entropy periodogram:

```python
trial_periods = np.linspace(0.1, 10.0, 5000)
S = get_entropies(t, u, trial_periods, L=7, K=7)
best = trial_periods[np.argmin(S)]
```

## How it works

For each trial period `p`, the observations are folded into a phase–magnitude diagram on the unit square and divided into an `L × K` grid. The Shannon entropy of the resulting occupation-probability matrix is computed:

```
S = -∑ μᵢ ln(μᵢ)
```

When `p` equals the true period the light curve is ordered and entropy is low; otherwise the diagram is disordered and entropy is high. The true period corresponds to the deepest minimum of the entropy periodogram.

## API

| Function | Description |
|---|---|
| `find_best_period(t, u, p0, p1, p_num, ...)` | Full pipeline: builds period grid, removes aliases, returns top candidates |
| `get_entropies(t, u, p, L, K)` | Vectorized entropy at each trial period |
| `get_test_periods(p0, p1, n, ...)` | Non-uniform period grid (dense at short periods, sparse at long) |
| `get_phases(t, u, period)` | Fold times into phases in [0, 1) |
| `get_entropy(Mu)` | Shannon entropy of an occupation-probability matrix |

See each function's docstring for full parameter details.

## Reference

Cincotta, P. M., Méndez, M., & Núñez, J. A. (1995). *Astronomical Time Series Analysis. I. A Search for Periodicity Using Information Entropy.* ApJ, 449, 231. [ADS](https://ui.adsabs.harvard.edu/abs/1995ApJ...449..231C)

## License

MIT
