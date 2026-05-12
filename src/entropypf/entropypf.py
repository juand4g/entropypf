import numpy as np
from scipy.signal import find_peaks


def get_test_periods(period_i, period_f, n, flat=False, threshold=30.0, fine_weight=2.0):
    """
    Build a grid of trial periods to search over.

    By default (`flat=False`) the grid is non-uniform: the region below
    `threshold` days is sampled with a dense, evenly-spaced linspace, while
    the region above it uses a power-law-distributed sparse sampling that
    concentrates points near the threshold and thins out toward `period_f`.
    When the search range straddles the threshold both sub-grids are combined,
    with `fine_weight` controlling how many extra points are allocated to the
    fine region relative to its fractional width.

    Setting `flat=True` returns a plain linspace regardless of `threshold`.

    Parameters
    ----------
    period_i : float
        Lower bound of the period search range (days).
    period_f : float
        Upper bound of the period search range (days).
    n : int
        Total number of trial periods to generate.
    flat : bool, optional
        If True, return a uniform linspace from `period_i` to `period_f`.
        Default is False.
    threshold : float, optional
        Boundary (days) between the fine and sparse sampling regions.
        Default is 30.0.
    fine_weight : float, optional
        Multiplier applied to the fine region's fractional width when
        computing the fine/sparse split. Higher values allocate more points
        to the fine region. Default is 2.0.

    Returns
    -------
    periods : ndarray of shape (n,)
        Sorted array of trial periods in days.
    """
    if flat == False:
        THRESHOLD   = threshold
        FINE_WEIGHT = fine_weight

        if period_f <= THRESHOLD:
            return np.linspace(period_i, period_f, n)

        if period_i >= THRESHOLD:
            u = np.sort(np.random.uniform(0, 1, n))
            u = u ** 1.8  # power-law: denser near threshold, sparser at long periods
            return period_i + u * (period_f - period_i)

        # Mixed range: part fine, part sparse
        fine_fraction     = (THRESHOLD - period_i) / (period_f - period_i)
        weighted_fraction = (fine_fraction * FINE_WEIGHT) / (
            fine_fraction * FINE_WEIGHT + (1 - fine_fraction)
        )

        n_fine   = int(round(n * weighted_fraction))
        n_sparse = n - n_fine

        fine_periods = np.linspace(period_i, THRESHOLD, n_fine,
                                   endpoint=(n_sparse == 0)) if n_fine > 0 else np.array([])

        if n_sparse > 0:
            u     = np.sort(np.random.uniform(0, 1, n_sparse))
            u     = u ** 1.8
            noise = np.random.uniform(-0.5 / n_sparse, 0.5 / n_sparse, n_sparse)
            u     = np.sort(np.clip(u + noise, 0, 1))
            sparse_periods = THRESHOLD + u * (period_f - THRESHOLD)
        else:
            sparse_periods = np.array([])

        return np.concatenate([fine_periods, sparse_periods])
    else:
        return np.linspace(period_i, period_f, n)


def get_phases(t_data, u_data, trial_period):
    """
    Compute folded phases for a given trial period.

    The phase reference epoch t0 is chosen as the observation time whose
    magnitude is closest to the 2nd percentile of `u_data` (near peak
    brightness for standard astronomical magnitudes). Phases are returned
    in [0, 1).

    Parameters
    ----------
    t_data : array-like of shape (N,)
        Observation times (days).
    u_data : array-like of shape (N,)
        Observed magnitudes.
    trial_period : float
        Period to fold on (days).

    Returns
    -------
    phases : ndarray of shape (N,)
        Folded phases in [0, 1).
    """
    threshold = np.percentile(u_data, 2)
    idx       = np.argmin(np.abs(u_data - threshold))
    t0        = t_data[idx]
    return ((t_data - t0) / trial_period) % 1


def get_probabilities_unitcube(phases, u_data, L, K):
    """
    Estimate the 2-D occupation probability in phase-magnitude space.

    Both axes are mapped to [0, 1]: phase is already in that range, and
    magnitude is min-max normalized. The unit square is divided into an
    L × K grid and the fraction of data points in each cell is returned.

    Parameters
    ----------
    phases : array-like of shape (N,)
        Folded phases in [0, 1).
    u_data : array-like of shape (N,)
        Observed magnitudes (any scale; normalized internally).
    L : int
        Number of bins along the phase axis.
    K : int
        Number of bins along the magnitude axis.

    Returns
    -------
    Mu : ndarray of shape (L, K)
        Occupation probabilities. Entries sum to 1 (or less if any point
        falls on the boundary; clipping ensures no out-of-bounds access).
    """
    phi = np.array(phases)
    u   = np.array(u_data, dtype=float)

    u_min, u_max = u.min(), u.max()
    if u_max > u_min:
        u_norm = (u - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(u)

    N_total = len(phi)
    H, _, _ = np.histogram2d(phi, u_norm,
                              bins=[L, K],
                              range=[[0, 1], [0, 1]])
    Mu = H / N_total
    return Mu


def get_entropy(Mu):
    """
    Compute the Shannon entropy of an occupation-probability matrix.

    Uses the convention 0·log(0) = 0, so empty cells contribute nothing.

    Parameters
    ----------
    Mu : ndarray
        Occupation probabilities (e.g. returned by `get_probabilities_unitcube`).

    Returns
    -------
    entropy : float
        H = -Σ p·ln(p), summed over all cells with p > 0.
    """
    Mu_nz = Mu[Mu > 0]
    return (-Mu_nz * np.log(Mu_nz)).sum()


def get_entropies(t, u, p, L=7, K=7):
    """
    Compute the normalized Shannon entropy of the phase-magnitude diagram
    for an array of trial periods.

    This is the bare entropy periodogram: no period grid is built internally,
    no aliases are removed, and no peaks are selected. It is useful when you
    want to supply your own period array, inspect the full entropy curve, or
    locate minima with your own criteria.

    Parameters
    ----------
    t : array-like of shape (N,)
        Observation times (days).
    u : array-like of shape (N,)
        Observed magnitudes.
    p : array-like of shape (P,)
        Trial periods (days) at which to evaluate the entropy. All values
        must be strictly positive.
    L : int, optional
        Number of phase bins (horizontal axis of the grid). Default is 7.
    K : int, optional
        Number of magnitude bins (vertical axis of the grid). Default is 7.

    Returns
    -------
    entropies : ndarray of shape (P,)
        Normalized Shannon entropy at each trial period, in [0, 1]. Lower
        values indicate a more structured (better-phased) light curve. Values
        are normalized by log(L*K) so that a perfectly uniform distribution
        gives 1 and a single occupied cell gives 0.

    Examples
    --------
    >>> import numpy as np
    >>> import entropypf as epf
    >>> periods = np.linspace(0.5, 2.0, 5000)
    >>> entropies = epf.get_entropies(t, u, periods, L=7, K=7)
    >>> best = periods[np.argmin(entropies)]
    """
    t_data  = np.asarray(t, dtype=float)
    u_data  = np.asarray(u, dtype=float)
    periods = np.asarray(p, dtype=float)
    N = len(t_data)
    P = len(periods)

    # Phase reference: observation nearest to the 2nd-percentile magnitude
    threshold = np.percentile(u_data, 2)
    t0        = t_data[np.argmin(np.abs(u_data - threshold))]

    # Min-max normalize magnitudes to [0, 1]
    u_min, u_max = u_data.min(), u_data.max()
    if u_max > u_min:
        u_norm = (u_data - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros(N)

    # Phase matrix for all trial periods simultaneously: shape (N, P)
    all_phases = ((t_data[:, None] - t0) / periods[None, :]) % 1.0

    # Integer bin indices
    phase_bins = np.floor(all_phases * L).astype(np.int32).clip(0, L - 1)  # (N, P)
    u_bins     = np.floor(u_norm * K).astype(np.int32).clip(0, K - 1)     # (N,)

    # Flatten all 2-D histograms into a single bincount call.
    # Each period occupies its own L*K-wide band in the flat index space.
    combined        = phase_bins * K + u_bins[:, None]                     # (N, P)
    offsets         = np.arange(P, dtype=np.int64) * (L * K)              # (P,)
    combined_offset = combined.T.astype(np.int64) + offsets[:, None]      # (P, N)

    counts = np.bincount(combined_offset.ravel(), minlength=P * L * K)
    Mu_all = counts.reshape(P, L * K) / N                                  # (P, L*K)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_Mu = np.where(Mu_all > 0, np.log(Mu_all), 0.0)
    return -(Mu_all * log_Mu).sum(axis=1) / np.log(L * K)                 # (P,)


def find_best_period(t, u, p0, p1, p_num, L=7, K=7,
                     alias_eps=0.01,
                     n_candidates=3,
                     peak_prominence=0.01,
                     peak_distance=30,
                     aliases=(1,)):
    """
    Find the best period(s) of a light curve using the minimum-entropy method.

    Builds a non-uniform trial period grid via `get_test_periods`, removes
    known alias periods, then computes the normalized Shannon entropy of the
    phase-magnitude diagram for every trial period in a single vectorized
    pass. Local entropy minima (corresponding to well-phased light curves)
    are located with `scipy.signal.find_peaks` and returned sorted by
    ascending entropy.

    Alias removal
    -------------
    For each base period ``b`` in ``aliases``, trial periods within
    ``alias_eps`` of any multiple of ``b/2`` or ``b/3`` are discarded.
    With the default ``aliases=(1,)`` this removes harmonics of the sidereal
    day (0.5 d, 1.0 d, 1.5 d, … and 1/3 d, 2/3 d, …), which are the most
    common observing-cadence aliases. Adding further bases extends the mask
    to additional periodicities:

    * ``aliases=(1, 30)``   — also masks monthly aliases (15 d, 30 d, 10 d, …)
    * ``aliases=(1, 365)``  — also masks yearly aliases (~182.5 d, 365 d, …)

    Parameters
    ----------
    t : array-like of shape (N,)
        Observation times (days).
    u : array-like of shape (N,)
        Observed magnitudes.
    p0 : float
        Lower bound of the period search range (days).
    p1 : float
        Upper bound of the period search range (days).
    p_num : int
        Number of trial periods to sample from [p0, p1].
    L : int, optional
        Phase-axis bins for the entropy grid. Default is 7.
    K : int, optional
        Magnitude-axis bins for the entropy grid. Default is 7.
    alias_eps : float, optional
        Half-width (days) of the exclusion window around each alias center.
        Default is 0.01.
    n_candidates : int, optional
        Number of top candidates to return. Default is 3.
    peak_prominence : float, optional
        Minimum prominence (in normalized entropy units) required for a local
        minimum to be considered a candidate. Default is 0.01.
    peak_distance : int, optional
        Minimum number of grid steps between consecutive candidates.
        Default is 30.
    aliases : sequence of float, optional
        Base periods (days) around which alias harmonics are removed.
        For each base ``b`` the excluded centers are all multiples of ``b/2``
        and ``b/3`` that fall within [p0, p1]. Default is ``(1,)``, which
        masks only the standard daily aliases.

    Returns
    -------
    candidate_periods : ndarray of shape (n_candidates,)
        Best-fit periods in days, sorted by ascending entropy (best first).
    candidate_entropies : ndarray of shape (n_candidates,)
        Normalized Shannon entropy at each candidate period.
    """
    testing_periods = get_test_periods(p0, p1, p_num)

    # For each base b, mask multiples of b/2 and b/3 (half- and third-period
    # harmonics). With aliases=(1,) this reproduces the original behavior:
    # multiples of 0.5 d and 1/3 d are excluded.
    alias_parts = []
    for base in aliases:
        half_step  = base / 2.0
        third_step = base / 3.0
        alias_parts.append(np.arange(1, int(p1 / half_step)  + 2) * half_step)
        alias_parts.append(np.arange(1, int(p1 / third_step) + 2) * third_step)

    alias_centers = np.unique(np.concatenate(alias_parts))
    alias_centers = alias_centers[(alias_centers >= p0) & (alias_centers <= p1)]

    if len(alias_centers) > 0:
        aliasing_mask = (
            np.abs(testing_periods[:, None] - alias_centers[None, :]) < alias_eps
        ).any(axis=1)
    else:
        aliasing_mask = np.zeros(len(testing_periods), dtype=bool)

    testing_periods = testing_periods[~aliasing_mask]

    entropies = get_entropies(t, u, testing_periods, L, K)

    # Locate local minima by finding peaks in the negated entropy curve
    peaks, _ = find_peaks(
        -entropies,
        prominence=peak_prominence,
        distance=peak_distance
    )

    if len(peaks) == 0:
        peaks = [np.argmin(entropies)]

    candidate_periods   = testing_periods[peaks]
    candidate_entropies = entropies[peaks]

    order = np.argsort(candidate_entropies)
    candidate_periods   = candidate_periods[order][:n_candidates]
    candidate_entropies = candidate_entropies[order][:n_candidates]

    return candidate_periods, candidate_entropies
