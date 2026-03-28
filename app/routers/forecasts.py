"""
Forecasts router.

Endpoints
---------
GET  /forecasts/purchase-suggestions
    The main endpoint. Returns all products with forecast, safety stock,
    ABC-XYZ category, current stock, and replenishment status.
    Results are cached for CACHE_TTL_SECONDS to avoid re-running Prophet on
    every request (Prophet fitting takes ~0.5-2 s per product).

GET  /forecasts/products/{product_id}
    Detailed forecast for a single product (weekly series + confidence bands).

GET  /forecasts/customers/{customer_id}
    All product forecasts for a specific customer's purchase history.

GET  /forecasts/abc-xyz
    Full ABC-XYZ classification matrix for all products.

POST /forecasts/refresh-cache
    Invalidate the in-memory cache so the next GET recomputes everything.
"""

import asyncio
import logging
import time
from collections import defaultdict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.data.strapi_client import StrapiClient
from app.models import abc_xyz as abc_xyz_module
from app.models import safety_stock as ss_module
from app.models.forecaster import DemandForecaster
from app.schemas.forecast import AbcXyzResult, ProductForecast, PurchaseSuggestion

logger = logging.getLogger(__name__)
router = APIRouter()

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(key: str = Security(_api_key_header)) -> None:
    if settings.forecast_api_key and key != settings.forecast_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


# ------------------------------------------------------------------
# Simple in-memory cache
# ------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}  # key → (timestamp, data)


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < settings.cache_ttl_seconds:
        return entry[1]
    return None


def _cache_set(key: str, data: object) -> None:
    _cache[key] = (time.time(), data)


# ------------------------------------------------------------------
# Shared data-loading helper
# ------------------------------------------------------------------

async def _load_order_history(months_back: int = 24) -> list[dict]:
    client = StrapiClient()
    return await client.get_order_history(months_back=months_back)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/purchase-suggestions", dependencies=[Depends(_require_api_key)])
async def purchase_suggestions(
    horizon_days: int = 90,
    lead_time_days: int | None = None,
    service_level: float | None = None,
    months_back: int = 24,
) -> list[PurchaseSuggestion]:
    """
    Main replenishment dashboard.

    For each product that has been sold in the last `months_back` months:
    - Runs Prophet (or exponential smoothing fallback) to forecast the next
      `horizon_days` of demand with 95% confidence intervals.
    - Calculates safety stock and reorder point using demand variability.
    - Classifies the product in the ABC-XYZ matrix.
    - Compares forecast to current inventory and assigns a status:
        * ``deficit``    — forecast > stock (urgent)
        * ``order_soon`` — stock covers < 150% of forecast (caution)
        * ``sufficient`` — stock ≥ 150% of forecast (ok)

    Results are cached for `CACHE_TTL_SECONDS` seconds.
    """
    lt = lead_time_days if lead_time_days is not None else settings.default_lead_time_days
    sl = service_level if service_level is not None else settings.default_service_level
    cache_key = f"purchase_suggestions:{horizon_days}:{lt}:{sl}:{months_back}"

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("Returning cached purchase suggestions")
        return cached

    logger.info("Computing purchase suggestions (horizon=%dd, lead_time=%dd)", horizon_days, lt)

    client = StrapiClient()
    order_lines, stock_map, products = await asyncio.gather(
        client.get_order_history(months_back=months_back),
        client.get_current_stock(),
        client.get_products(),
    )

    product_meta: dict[int, dict] = {p["id"]: p for p in products}
    forecaster = DemandForecaster()

    # Build per-product structures needed for safety stock + ABC-XYZ
    product_ids = {l["product_id"] for l in order_lines}
    weekly_demand: dict[int, list[float]] = defaultdict(list)
    monthly_demand: dict[int, list[float]] = defaultdict(list)
    total_revenue: dict[int, float] = defaultdict(float)
    customer_count: dict[int, int] = defaultdict(int)

    for pid in product_ids:
        lines = [l for l in order_lines if l["product_id"] == pid]
        df = pd.DataFrame(lines)[["order_date", "qty", "revenue", "customer_id"]].copy()
        df["order_date"] = pd.to_datetime(df["order_date"])

        # Weekly demand (for safety stock)
        wkly = df.set_index("order_date").resample("W-MON")["qty"].sum()
        weekly_demand[pid] = wkly.fillna(0).tolist()

        # Monthly demand (for XYZ coefficient of variation)
        mnthly = df.set_index("order_date").resample("ME")["qty"].sum()
        monthly_demand[pid] = mnthly.fillna(0).tolist()

        total_revenue[pid] = float(df["revenue"].sum())
        customer_count[pid] = int(df["customer_id"].nunique())

    # ABC classification (by revenue)
    revenue_rows = [{"product_id": pid, "total_revenue": total_revenue[pid]} for pid in product_ids]

    # XYZ classification (by demand variability)
    variability_rows = [
        {"product_id": pid, "monthly_quantities": monthly_demand[pid]} for pid in product_ids
    ]
    abc_xyz_list = abc_xyz_module.classify(revenue_rows, variability_rows)
    abc_xyz_map: dict[int, dict] = {r["product_id"]: r for r in abc_xyz_list}

    # Lead time in weeks (safety stock formula uses same unit as demand series)
    lt_weeks = lt / 7.0

    suggestions: list[PurchaseSuggestion] = []

    for pid in product_ids:
        meta = product_meta.get(pid, {"name": "", "code": "", "category": ""})
        forecast = forecaster.forecast_product(order_lines, pid, horizon_days=horizon_days)
        ss_info = ss_module.calculate(
            weekly_demand[pid],
            lead_time_periods=lt_weeks,
            service_level=sl,
        )
        az = abc_xyz_map.get(pid, {"abc": "C", "xyz": "Z", "category": "CZ", "strategy": ""})

        current_stock = round(stock_map.get(pid, 0.0), 2)
        total_fc = forecast["total_forecast_qty"]
        deficit = round(total_fc - current_stock, 2)

        if deficit <= 0:
            status_val = "sufficient"
        elif current_stock >= total_fc * 0.5:
            status_val = "order_soon"
        else:
            status_val = "deficit"

        suggestions.append(
            PurchaseSuggestion(
                product_id=pid,
                product_name=meta.get("name", ""),
                product_code=meta.get("code", ""),
                product_category=meta.get("category", ""),
                product_unit=meta.get("unit", ""),
                current_stock=current_stock,
                total_forecast_qty=total_fc,
                deficit=deficit,
                customer_count=customer_count.get(pid, 0),
                safety_stock_info=ss_info,
                abc_xyz=az,
                forecast_method=forecast["method"],
                forecast_confidence=forecast["confidence"],
                status=status_val,
                forecast_periods=forecast["forecast_periods"],
            )
        )

    # Sort: deficit first, then order_soon, then by revenue descending
    _status_order = {"deficit": 0, "order_soon": 1, "sufficient": 2}
    suggestions.sort(
        key=lambda s: (
            _status_order[s.status],
            -total_revenue.get(s.product_id, 0.0),
        )
    )

    _cache_set(cache_key, suggestions)
    return suggestions


@router.get("/products/{product_id}", dependencies=[Depends(_require_api_key)])
async def forecast_product(
    product_id: int,
    horizon_days: int = 90,
    months_back: int = 24,
) -> ProductForecast:
    """
    Detailed weekly forecast for a single product with 95% confidence intervals.
    """
    client = StrapiClient()
    order_lines, products = await asyncio.gather(
        client.get_order_history(months_back=months_back),
        client.get_products(),
    )

    product_meta = {p["id"]: p for p in products}
    meta = product_meta.get(product_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    forecaster = DemandForecaster()
    result = forecaster.forecast_product(order_lines, product_id, horizon_days=horizon_days)

    return ProductForecast(
        product_id=product_id,
        product_name=meta.get("name", ""),
        product_code=meta.get("code", ""),
        product_category=meta.get("category", ""),
        **result,
    )


@router.get("/customers/{customer_id}", dependencies=[Depends(_require_api_key)])
async def forecast_customer(
    customer_id: int,
    horizon_days: int = 90,
    months_back: int = 24,
) -> list[ProductForecast]:
    """
    All product forecasts based on a specific customer's purchase history.
    Useful to show on the customer detail page.
    """
    client = StrapiClient()
    order_lines, products = await asyncio.gather(
        client.get_order_history(months_back=months_back),
        client.get_products(),
    )

    customer_lines = [l for l in order_lines if l["customer_id"] == customer_id]
    if not customer_lines:
        return []

    product_meta = {p["id"]: p for p in products}
    product_ids = {l["product_id"] for l in customer_lines}
    forecaster = DemandForecaster()

    forecasts = []
    for pid in product_ids:
        meta = product_meta.get(pid, {"name": "", "code": "", "category": ""})
        result = forecaster.forecast_product(customer_lines, pid, horizon_days=horizon_days)
        forecasts.append(
            ProductForecast(
                product_name=meta.get("name", ""),
                product_code=meta.get("code", ""),
                product_category=meta.get("category", ""),
                **result,
            )
        )

    forecasts.sort(key=lambda f: -f.total_forecast_qty)
    return forecasts


@router.get("/abc-xyz", dependencies=[Depends(_require_api_key)])
async def abc_xyz_analysis(months_back: int = 24) -> list[AbcXyzResult]:
    """
    Full ABC-XYZ classification matrix.

    Combines revenue contribution (ABC) with demand variability (XYZ)
    to produce a 3×3 matrix of 9 product categories with replenishment strategies.
    """
    cache_key = f"abc_xyz:{months_back}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    order_lines = await _load_order_history(months_back=months_back)
    product_ids = {l["product_id"] for l in order_lines}

    client = StrapiClient()
    products = await client.get_products()
    product_meta = {p["id"]: p for p in products}

    revenue_rows = []
    variability_rows = []

    for pid in product_ids:
        lines = [l for l in order_lines if l["product_id"] == pid]
        df = pd.DataFrame(lines)[["order_date", "qty", "revenue"]].copy()
        df["order_date"] = pd.to_datetime(df["order_date"])

        monthly_qty = df.set_index("order_date").resample("ME")["qty"].sum().fillna(0).tolist()
        total_rev = float(df["revenue"].sum())

        revenue_rows.append({"product_id": pid, "total_revenue": total_rev})
        variability_rows.append({"product_id": pid, "monthly_quantities": monthly_qty})

    classifications = abc_xyz_module.classify(revenue_rows, variability_rows)
    rev_map = {r["product_id"]: r["total_revenue"] for r in revenue_rows}

    result = [
        AbcXyzResult(
            product_id=c["product_id"],
            product_name=product_meta.get(c["product_id"], {}).get("name", ""),
            product_code=product_meta.get(c["product_id"], {}).get("code", ""),
            total_revenue=round(rev_map.get(c["product_id"], 0.0), 2),
            abc=c["abc"],
            xyz=c["xyz"],
            category=c["category"],
            strategy=c["strategy"],
        )
        for c in classifications
    ]

    _cache_set(cache_key, result)
    return result


@router.post("/refresh-cache", dependencies=[Depends(_require_api_key)])
async def refresh_cache():
    """Invalidate the in-memory cache so the next request recomputes forecasts."""
    _cache.clear()
    return {"cleared": True}
