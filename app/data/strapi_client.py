"""
Strapi REST API client.

Fetches order history, products, and current inventory from the Strapi backend.
Handles pagination automatically — Strapi 5 caps pageSize at 200 by default.
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

PAGE_SIZE = 100


class StrapiClient:
    def __init__(self) -> None:
        self._base = settings.strapi_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.strapi_api_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_all(self, path: str, base_params: dict) -> list[dict]:
        """Paginate through all Strapi results and return a flat list of records."""
        results: list[dict] = []
        page = 1

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                params = {
                    **base_params,
                    "pagination[pageSize]": PAGE_SIZE,
                    "pagination[page]": page,
                }
                resp = await client.get(
                    f"{self._base}/api/{path}",
                    headers=self._headers,
                    params=params,
                )
                if not resp.is_success:
                    logger.error(
                        "Strapi %s %s → %d: %s",
                        path, dict(params), resp.status_code, resp.text[:500],
                    )
                resp.raise_for_status()
                body = resp.json()

                data = body.get("data", [])
                results.extend(data)

                meta = body.get("meta", {}).get("pagination", {})
                if page >= meta.get("pageCount", 1):
                    break
                page += 1

        return results

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_order_history(self, months_back: int = 24) -> list[dict]:
        """
        Fetch completed sale / partial-invoice orders for the last N months.

        Returns a flat list of line-level dicts:
          {
            order_id, order_date, customer_id, customer_name,
            product_id, product_name, product_code, product_category,
            qty, price, revenue
          }
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=months_back * 30)).strftime(
            "%Y-%m-%dT00:00:00.000Z"
        )

        # Strapi 5: flat top-level filters are implicitly ANDed; avoid $and/$or
        # wrappers which conflict with sibling params in REST query strings.
        raw_orders = await self._get_all(
            "orders",
            {
                "filters[type][$in][0]": "sale",
                "filters[type][$in][1]": "partial-invoice",
                "filters[state][$in][0]": "completed",
                "filters[state][$in][1]": "confirmed",
                "filters[state][$in][2]": "processing",
                "filters[createdAt][$gte]": cutoff,
                # Strapi 5: use explicit fields instead of wildcard '*' for named relations
                "populate[orderProducts][populate][product][fields][0]": "id",
                "populate[orderProducts][populate][product][fields][1]": "name",
                "populate[orderProducts][populate][product][fields][2]": "code",
                "populate[orderProducts][populate][product][fields][3]": "category",
                "populate[customer][fields][0]": "id",
                "populate[customer][fields][1]": "name",
            },
        )

        lines: list[dict] = []
        for order in raw_orders:
            attrs = order if "id" in order else order  # Strapi 5 returns flat objects
            order_id = attrs.get("id")
            customer = attrs.get("customer") or {}
            customer_id = customer.get("id")
            customer_name = customer.get("name", "")

            raw_date = (
                attrs.get("createdDate")
                or attrs.get("completedDate")
                or attrs.get("createdAt")
            )
            try:
                order_date = datetime.fromisoformat(
                    raw_date.replace("Z", "+00:00")
                ).date()
            except Exception:
                continue

            for op in attrs.get("orderProducts") or []:
                product = op.get("product") or {}
                product_id = product.get("id")
                if not product_id:
                    continue

                qty = float(op.get("confirmedQuantity") or op.get("requestedQuantity") or 0)
                price = float(op.get("price") or 0)

                lines.append(
                    {
                        "order_id": order_id,
                        "order_date": order_date,
                        "customer_id": customer_id,
                        "customer_name": customer_name,
                        "product_id": product_id,
                        "product_name": product.get("name", ""),
                        "product_code": product.get("code", ""),
                        "product_category": product.get("category", ""),
                        "qty": qty,
                        "price": price,
                        "revenue": qty * price,
                    }
                )

        logger.info(
            "Fetched %d order lines from Strapi (last %d months)",
            len(lines),
            months_back,
        )
        return lines

    async def get_products(self) -> list[dict]:
        """
        Return all active, purchasable line products.

        Excluded from forecasting:
        - type=service (no physical stock)
        - type=cutItem (derived from a parent product, not ordered independently)
        - isLineProduct=False (one-off imports, not replenished regularly)

        Note: isLineProduct defaults to True in the schema. Existing products
        without the field set (null) are treated as line products (safe default).
        """
        raw = await self._get_all(
            "products",
            {
                "filters[isActive][$eq]": "true",
                # Exclude services and cut items at the query level
                "filters[type][$notIn][0]": "service",
                "filters[type][$notIn][1]": "cutItem",
            },
        )
        return [
            {
                "id": p.get("id"),
                "name": p.get("name", ""),
                "code": p.get("code", ""),
                "unit": p.get("unit", ""),
                "category": p.get("category", ""),
                "type": p.get("type", ""),
            }
            for p in raw
            # isLineProduct=None (old records without the field) is treated as True.
            # Only products explicitly set to False are excluded.
            if p.get("isLineProduct") is not False
        ]

    async def get_incoming_purchase_stock(self) -> dict[int, dict]:
        """
        Return stock quantities that are on order but not yet physically in the warehouse.

        Only considers ``purchase`` orders in ``confirmed`` or ``processing`` state —
        these represent goods being manufactured or in ocean transit.  Items for such
        orders are created by ``PurchaseInStrategy`` only when the order reaches
        ``completed``, so they are NOT yet reflected in current_stock.

        Nationalizations and transfers are excluded: their items already exist in
        the system (free-trade-zone warehouse) and are therefore already counted
        in get_current_stock().

        Returns
        -------
        dict[int, dict]
            product_id → {
                "qty": float,              # total incoming units across all open POs
                "earliest_arrival": str | None,  # ISO date of the soonest estimatedCompletedDate
            }
        """
        raw = await self._get_all(
            "orders",
            {
                "filters[type][$eq]": "purchase",
                "filters[state][$in][0]": "confirmed",
                "filters[state][$in][1]": "processing",
                "populate[orderProducts][populate][product][fields][0]": "id",
            },
        )

        incoming: dict[int, dict] = {}
        for order in raw:
            estimated_date = order.get("estimatedCompletedDate")  # YYYY-MM-DD string or None
            for op in order.get("orderProducts") or []:
                product = op.get("product") or {}
                pid = product.get("id")
                if not pid:
                    continue
                qty = float(
                    op.get("confirmedQuantity") or op.get("requestedQuantity") or 0
                )
                if qty <= 0:
                    continue

                if pid not in incoming:
                    incoming[pid] = {"qty": 0.0, "earliest_arrival": None}

                incoming[pid]["qty"] = round(incoming[pid]["qty"] + qty, 4)

                # Track the earliest estimated arrival across all open POs for this product
                if estimated_date:
                    prev = incoming[pid]["earliest_arrival"]
                    if prev is None or estimated_date < prev:
                        incoming[pid]["earliest_arrival"] = estimated_date

        logger.info("Fetched incoming purchase stock for %d products", len(incoming))
        return incoming

    async def get_current_stock(self) -> dict[int, float]:
        """
        Return current available + reserved stock per product_id.
        """
        # Strapi 5: mixing sparse fields with populate causes a 400.
        # Fetch all item fields (small payload) and pick what we need in Python.
        raw = await self._get_all(
            "items",
            {
                "filters[state][$in][0]": "available",
                "filters[state][$in][1]": "reserved",
                "populate[product][fields][0]": "id",
            },
        )
        stock: dict[int, float] = {}
        for item in raw:
            product = item.get("product") or {}
            pid = product.get("id")
            if pid is None:
                continue
            stock[pid] = stock.get(pid, 0.0) + float(item.get("currentQuantity") or 0)
        return stock
