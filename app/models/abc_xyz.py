"""
ABC-XYZ product classification.

  ABC  — based on cumulative revenue contribution
         A: top 80%   (high value)
         B: next 15%  (medium value)
         C: remaining  (low value)

  XYZ  — based on demand variability (Coefficient of Variation of monthly demand)
         X: CV < 0.5   (stable, predictable)
         Y: 0.5 ≤ CV < 1.0  (moderate variability)
         Z: CV ≥ 1.0   (sporadic / unpredictable)

The combined 3×3 matrix drives replenishment strategy recommendations.
"""

import numpy as np
import pandas as pd


# Action / replenishment strategy per category (in Spanish to match the app)
_STRATEGIES: dict[str, str] = {
    "AX": "Alto valor, demanda estable — stock alto, reposición automática",
    "AY": "Alto valor, variabilidad media — buffer moderado, revisión quincenal",
    "AZ": "Alto valor, demanda esporádica — análisis manual, stock de seguridad alto",
    "BX": "Valor medio, demanda estable — reposición periódica estándar",
    "BY": "Valor medio, variabilidad media — revisión mensual",
    "BZ": "Valor medio, demanda esporádica — pedido bajo demanda, mínimos bajos",
    "CX": "Bajo valor, demanda estable — stock mínimo, reposición simple",
    "CY": "Bajo valor, variabilidad media — stock bajo",
    "CZ": "Bajo valor, demanda esporádica — evaluar descontinuación",
}


def _abc(df: pd.DataFrame) -> dict[int, str]:
    """
    Parameters
    ----------
    df : DataFrame with columns [product_id, total_revenue]
    """
    df = df.sort_values("total_revenue", ascending=False).copy()
    grand_total = df["total_revenue"].sum()

    if grand_total == 0:
        return {pid: "C" for pid in df["product_id"]}

    df["cum_pct"] = df["total_revenue"].cumsum() / grand_total * 100

    def _label(cum_pct: float) -> str:
        if cum_pct <= 80:
            return "A"
        if cum_pct <= 95:
            return "B"
        return "C"

    return dict(zip(df["product_id"], df["cum_pct"].map(_label)))


def _xyz(product_monthly: list[dict]) -> dict[int, str]:
    """
    Parameters
    ----------
    product_monthly : list of {product_id, monthly_quantities: list[float]}
    """
    result: dict[int, str] = {}
    for item in product_monthly:
        pid = item["product_id"]
        qtys = item.get("monthly_quantities", [])

        if not qtys or len(qtys) < 3:
            # Too few months → cannot assess variability → treat as sporadic
            result[pid] = "Z"
            continue

        arr = np.array(qtys, dtype=float)
        mu = arr.mean()
        sigma = arr.std(ddof=1)
        cv = sigma / mu if mu > 0 else float("inf")

        if cv < 0.5:
            result[pid] = "X"
        elif cv < 1.0:
            result[pid] = "Y"
        else:
            result[pid] = "Z"

    return result


def classify(
    revenue_rows: list[dict],
    monthly_demand_rows: list[dict],
) -> list[dict]:
    """
    Build the full ABC-XYZ matrix.

    Parameters
    ----------
    revenue_rows : list of {product_id, total_revenue}
    monthly_demand_rows : list of {product_id, monthly_quantities: list[float]}

    Returns
    -------
    list of {product_id, abc, xyz, category, strategy}
    """
    if not revenue_rows:
        return []

    df_rev = pd.DataFrame(revenue_rows)
    abc_map = _abc(df_rev)
    xyz_map = _xyz(monthly_demand_rows)

    all_ids = set(abc_map) | set(xyz_map)
    classifications = []
    for pid in all_ids:
        a = abc_map.get(pid, "C")
        x = xyz_map.get(pid, "Z")
        cat = f"{a}{x}"
        classifications.append(
            {
                "product_id": pid,
                "abc": a,
                "xyz": x,
                "category": cat,
                "strategy": _STRATEGIES.get(cat, ""),
            }
        )

    # Sort: A first, then X first within each A/B/C group
    abc_order = {"A": 0, "B": 1, "C": 2}
    xyz_order = {"X": 0, "Y": 1, "Z": 2}
    classifications.sort(key=lambda r: (abc_order[r["abc"]], xyz_order[r["xyz"]]))
    return classifications
