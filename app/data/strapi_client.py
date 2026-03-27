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
                "filters[state][$in][1]": "processing",
                "filters[createdAt][$gte]": cutoff,
                "populate[orderProducts][populate][product]": "*",
                "populate[customer]": "*",
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

                qty = float(op.get("requestedQuantity") or 0)
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
        """Return all active products."""
        raw = await self._get_all(
            "products",
            {"filters[isActive][$eq]": "true"},
        )
        return [
            {
                "id": p.get("id"),
                "name": p.get("name", ""),
                "code": p.get("code", ""),
                "unit": p.get("unit", ""),
                "category": p.get("category", ""),
            }
            for p in raw
        ]

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
