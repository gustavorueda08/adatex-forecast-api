"""
Safety stock and reorder point calculator.

Uses the standard statistical formula:

  Safety Stock  = z × √(L × σ_d²  +  μ_d² × σ_L²)

  Where:
    z    = service-level z-score (e.g. 1.645 for 95 %)
    L    = mean lead time in periods
    σ_d  = standard deviation of demand per period
    μ_d  = mean demand per period
    σ_L  = standard deviation of lead time in periods (0 if fixed)

  Reorder Point = μ_d × L + Safety Stock

Both demand and lead time are expressed in the **same** time unit
(typically weeks when using weekly demand series).
"""

import numpy as np

# z-scores for common service levels
_Z_TABLE: dict[float, float] = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
}


def _z_score(service_level: float) -> float:
    """Return the z-score closest to the requested service level."""
    # Exact match first
    if service_level in _Z_TABLE:
        return _Z_TABLE[service_level]
    # Nearest key
    closest = min(_Z_TABLE.keys(), key=lambda k: abs(k - service_level))
    return _Z_TABLE[closest]


def calculate(
    demand_history: list[float],
    lead_time_periods: float = 4.0,
    service_level: float = 0.95,
    lead_time_std_periods: float = 0.0,
) -> dict:
    """
    Calculate safety stock, reorder point, and demand statistics.

    Parameters
    ----------
    demand_history : list[float]
        Observed demand quantities per period (e.g. weekly units sold).
        At least 4 observations are recommended.
    lead_time_periods : float
        Mean supplier lead time expressed in the same unit as demand_history.
        Default 4 weeks (~30 days).
    service_level : float
        Desired in-stock probability.  One of: 0.80, 0.85, 0.90, 0.95, 0.98, 0.99.
    lead_time_std_periods : float
        Standard deviation of lead time (0 = fixed lead time).

    Returns
    -------
    dict
        safety_stock, reorder_point, avg_demand, demand_std,
        coefficient_of_variation, service_level, lead_time_periods
    """
    if not demand_history or len(demand_history) < 2:
        return {
            "safety_stock": 0.0,
            "reorder_point": 0.0,
            "avg_demand_per_period": 0.0,
            "demand_std": 0.0,
            "coefficient_of_variation": 0.0,
            "service_level": service_level,
            "lead_time_periods": lead_time_periods,
        }

    arr = np.array(demand_history, dtype=float)
    mu_d = float(arr.mean())
    sigma_d = float(arr.std(ddof=1))
    cv = sigma_d / mu_d if mu_d > 0 else 0.0

    z = _z_score(service_level)

    # Full formula (handles both fixed and variable lead time)
    variance_combined = (
        lead_time_periods * sigma_d**2
        + mu_d**2 * lead_time_std_periods**2
    )
    ss = z * float(np.sqrt(max(0.0, variance_combined)))
    rop = mu_d * lead_time_periods + ss

    return {
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "avg_demand_per_period": round(mu_d, 2),
        "demand_std": round(sigma_d, 2),
        "coefficient_of_variation": round(cv, 3),
        "service_level": service_level,
        "lead_time_periods": lead_time_periods,
    }
