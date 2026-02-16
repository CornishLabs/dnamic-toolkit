"""
Jeffreys binomial credible intervals — accuracy & speed benchmark.

Implementations:
  - ours_scipy: SciPy betaincinv (vectorized, compiled)
  - ours_mpmath: pure-Python mpmath.betaincinv fallback
  - statsmodels.proportion_confint(method="jeffreys")
  - astropy.stats.binom_conf_interval(interval="jeffreys")

Change LEVEL below to 0.68, 0.90, 0.95, 0.99, etc.
"""

import numpy as np
import time
import sys

# -------- settings --------
LEVEL = 0.95
ALPHA = 1 - LEVEL
SEED = 42
SIZES = [1_000, 10_000, 100_000]  # vector sizes for timing
RNG = np.random.default_rng(SEED)


# -------- implementations --------
def jeffreys_ci_ours_scipy(y, n, alpha=ALPHA):
    """Vectorized Jeffreys CI via SciPy's compiled inverse incomplete beta."""
    from scipy.special import betaincinv
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    a = y + 0.5
    b = n - y + 0.5
    lo = betaincinv(a, b, alpha / 2.0)
    hi = betaincinv(a, b, 1.0 - alpha / 2.0)
    return lo, hi


def jeffreys_ci_ours_mpmath(y, n, alpha=ALPHA):
    """Pure-Python fallback using mpmath.betaincinv; fine for small vectors."""
    import mpmath as mp
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    a = y + 0.5
    b = n - y + 0.5
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    betainv = np.vectorize(lambda aa, bb, qq: float(mp.betaincinv(aa, bb, 0, qq)))
    lo = betainv(a, b, lo_q)
    hi = betainv(a, b, hi_q)
    return lo, hi


def jeffreys_ci_statsmodels(y, n, alpha=ALPHA):
    """statsmodels wrapper (calls SciPy under the hood)."""
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(count=y, nobs=n, alpha=alpha, method="jeffreys")
    return np.asarray(lo), np.asarray(hi)


def jeffreys_ci_astropy(y, n, alpha=ALPHA):
    """Astropy implementation."""
    from astropy.stats import binom_conf_interval
    arr = binom_conf_interval(k=y, n=n, confidence_level=1 - alpha, interval="jeffreys")
    lo = np.asarray(arr[:, 0])
    hi = np.asarray(arr[:, 1])
    return lo, hi


# -------- helpers --------
def lib_available(fn) -> bool:
    try:
        fn(np.array([1]), np.array([10]), alpha=ALPHA)
        return True
    except Exception:
        return False


def time_call(fn, y, n, alpha, reps=5):
    """Median runtime over 'reps' calls; includes a warmup."""
    fn(y, n, alpha)  # warmup
    dt = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(y, n, alpha)
        dt.append(time.perf_counter() - t0)
    return float(np.median(dt))


def validate_against_scipy(N=10_000, alpha=ALPHA):
    print("\n=== Accuracy check vs SciPy reference ===")
    try:
        import scipy  # noqa: F401
    except Exception:
        print("SciPy not installed — cannot run accuracy comparison. Exiting.")
        sys.exit(1)

    n = RNG.integers(1, 2000, size=N)
    y = RNG.integers(0, n + 1, size=N)

    lo_ref, hi_ref = jeffreys_ci_ours_scipy(y, n, alpha=alpha)

    # our mpmath (sample only — slow)
    try:
        lo_m, hi_m = jeffreys_ci_ours_mpmath(y[:50], n[:50], alpha=alpha)
        print(f"[ours_mpmath] max|Δ| vs ours_scipy (50 elems): "
              f"lo={np.max(np.abs(lo_m - lo_ref[:50])):.2e}, "
              f"hi={np.max(np.abs(hi_m - hi_ref[:50])):.2e}")
    except Exception:
        print("[ours_mpmath] skipped (mpmath not installed).")

    # statsmodels
    if lib_available(jeffreys_ci_statsmodels):
        lo_sm, hi_sm = jeffreys_ci_statsmodels(y, n, alpha=alpha)
        print(f"[statsmodels] max|Δ| vs ours_scipy: "
              f"lo={np.max(np.abs(lo_sm - lo_ref)):.2e}, "
              f"hi={np.max(np.abs(hi_sm - hi_ref)):.2e}")
    else:
        print("[statsmodels] not available; skipping accuracy check.")

    # astropy
    if lib_available(jeffreys_ci_astropy):
        lo_as, hi_as = jeffreys_ci_astropy(y, n, alpha=alpha)
        print(f"[astropy]     max|Δ| vs ours_scipy: "
              f"lo={np.max(np.abs(lo_as - lo_ref)):.2e}, "
              f"hi={np.max(np.abs(hi_as - hi_ref)):.2e}")
    else:
        print("[astropy] not available; skipping accuracy check.")

    # Edge cases (y=0 / y=n)
    edge_n = np.array([1, 2, 5, 10, 100, 1000])
    edge_y0 = np.zeros_like(edge_n)
    edge_yn = edge_n.copy()
    lo0, hi0 = jeffreys_ci_ours_scipy(edge_y0, edge_n, alpha=alpha)
    lon, hin = jeffreys_ci_ours_scipy(edge_yn, edge_n, alpha=alpha)
    print("\nEdge cases (y=0 and y=n), Jeffreys CI (level=%.3f):" % LEVEL)
    print("  y=0 -> lo:", np.round(lo0, 6), " hi:", np.round(hi0, 6))
    print("  y=n -> lo:", np.round(lon, 6), " hi:", np.round(hin, 6))


def benchmark(alpha=ALPHA):
    print("\n=== Benchmarks (median over 5 runs) ===")
    for N in SIZES:
        n = RNG.integers(1, 2000, size=N)
        y = RNG.integers(0, n + 1, size=N)

        # ours_scipy (reference)
        try:
            t_ours = time_call(jeffreys_ci_ours_scipy, y, n, alpha)
            print(f"N={N:6d}  ours (SciPy betaincinv):        {1e3 * t_ours:8.3f} ms")
        except Exception as e:
            print(f"N={N:6d}  ours (SciPy): error -> {e}")

        # statsmodels
        if lib_available(jeffreys_ci_statsmodels):
            t_sm = time_call(jeffreys_ci_statsmodels, y, n, alpha)
            print(f"N={N:6d}  statsmodels.proportion_confint: {1e3 * t_sm:8.3f} ms")
        else:
            print(f"N={N:6d}  statsmodels: not available")

        # astropy
        if lib_available(jeffreys_ci_astropy):
            t_as = time_call(jeffreys_ci_astropy, y, n, alpha)
            print(f"N={N:6d}  astropy.binom_conf_interval:    {1e3 * t_as:8.3f} ms")
        else:
            print(f"N={N:6d}  astropy: not available")

        # mpmath (only sensible to time at small N)
        if N <= 1_000:
            try:
                t_mp = time_call(jeffreys_ci_ours_mpmath, y, n, alpha, reps=3)
                print(f"N={N:6d}  ours (mpmath fallback):        {1e3 * t_mp:8.3f} ms")
            except Exception:
                print(f"N={N:6d}  ours (mpmath): not available")


def main():
    print("Jeffreys CI benchmark | LEVEL=%.3f  (alpha=%.3f)" % (LEVEL, ALPHA))
    # Accuracy checks (requires SciPy)
    try:
        import scipy  # noqa: F401
        validate_against_scipy(N=10_000, alpha=ALPHA)
    except Exception:
        print("\nSciPy is not installed; skipping accuracy checks.")
    # Benchmarks
    benchmark(alpha=ALPHA)


if __name__ == "__main__":
    main()
