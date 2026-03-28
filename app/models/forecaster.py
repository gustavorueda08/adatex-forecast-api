"""
Demand forecaster.

Primary model: Prophet (Meta) — handles trend, multiple seasonality, and
changepoints automatically. Returns weekly forecasts with 95% confidence intervals.

Fallback: Exponential smoothing — used when a product has fewer than
MIN_PROPHET_POINTS non-zero weeks of history (sparse data).
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prophet requires at least this many non-zero observations to fit reliably.
# 26 weeks ≈ 6 months — enough for one seasonal cycle to be visible.
MIN_PROPHET_POINTS = 26


class DemandForecaster:
    def forecast_product(
        self,
        order_lines: list[dict],
        product_id: int,
        horizon_days: int = 90,
    ) -> dict:
        """
        Forecast weekly demand for a single product over the next `horizon_days`.

        Parameters
        ----------
        order_lines : list[dict]
            Full order history (all products). Each dict must contain at minimum:
            ``product_id`` (int), ``order_date`` (date), ``qty`` (float).
        product_id : int
            The product to forecast.
        horizon_days : int
            How many days ahead to forecast (rounded up to full weeks).

        Returns
        -------
        dict with keys:
            product_id, method, forecast_periods (list), total_forecast_qty,
            data_points, confidence
        """
        product_lines = [l for l in order_lines if l["product_id"] == product_id]
        if not product_lines:
            return self._empty_result(product_id)

        series = self._build_weekly_series(product_lines)
        n_weeks = max(1, -(-horizon_days // 7))  # ceiling division
        nonzero = (series["y"] > 0).sum()

        if nonzero >= MIN_PROPHET_POINTS:
            return self._prophet_forecast(series, product_id, n_weeks)
        return self._exp_smoothing_forecast(series, product_id, n_weeks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_weekly_series(self, lines: list[dict]) -> pd.DataFrame:
        """Aggregate order lines to a weekly (Monday-anchored) demand series."""
        df = pd.DataFrame(lines)[["order_date", "qty"]].copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        weekly = (
            df.set_index("order_date")
            .resample("W-MON")["qty"]
            .sum()
            .reset_index()
            .rename(columns={"order_date": "ds", "qty": "y"})
        )
        weekly["y"] = weekly["y"].fillna(0.0).clip(lower=0.0)

        # Cap extreme outliers at median + 4 IQR to prevent single spikes
        # from skewing the model (e.g. bulk one-time orders).
        nonzero = weekly.loc[weekly["y"] > 0, "y"]
        if len(nonzero) >= 4:
            q1, q3 = nonzero.quantile(0.25), nonzero.quantile(0.75)
            upper_cap = q3 + 4 * (q3 - q1)
            weekly["y"] = weekly["y"].clip(upper=upper_cap)

        return weekly

    def _prophet_forecast(
        self, series: pd.DataFrame, product_id: int, n_weeks: int
    ) -> dict:
        try:
            from prophet import Prophet  # lazy import — Prophet takes ~1 s to import
        except ImportError:
            logger.error("prophet package not installed. Run: pip install prophet")
            return self._exp_smoothing_forecast(series, product_id, n_weeks)

        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                # Multiplicative mode handles products whose variance grows with demand
                seasonality_mode="multiplicative",
                interval_width=0.95,
                # Conservative changepoint prior — avoids over-fitting on 1-2 years of data
                changepoint_prior_scale=0.05,
                # Reduce noise from outlier weeks
                mcmc_samples=0,  # use MAP estimation (faster)
            )
            # Monthly seasonality (period = 30.5 days, 5 Fourier terms)
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            m.fit(series)
            future = m.make_future_dataframe(periods=n_weeks, freq="W")
            forecast = m.predict(future)

            future_fc = forecast[forecast["ds"] > series["ds"].max()].copy()
            future_fc[["yhat", "yhat_lower", "yhat_upper"]] = future_fc[
                ["yhat", "yhat_lower", "yhat_upper"]
            ].clip(lower=0.0)

            periods = [
                {
                    "week_start": row["ds"].strftime("%Y-%m-%d"),
                    "forecast_qty": round(float(row["yhat"]), 2),
                    "lower_95": round(float(row["yhat_lower"]), 2),
                    "upper_95": round(float(row["yhat_upper"]), 2),
                }
                for _, row in future_fc.iterrows()
            ]

            total = round(float(future_fc["yhat"].sum()), 2)
            n_data = len(series)
            confidence = "high" if n_data >= 52 else ("medium" if n_data >= 26 else "low")

            return {
                "product_id": product_id,
                "method": "prophet",
                "forecast_periods": periods,
                "total_forecast_qty": total,
                "data_points": n_data,
                "confidence": confidence,
            }

        except Exception as exc:
            logger.warning(
                "Prophet failed for product %s: %s — using fallback", product_id, exc
            )
            return self._exp_smoothing_forecast(series, product_id, n_weeks)

    def _exp_smoothing_forecast(
        self, series: pd.DataFrame, product_id: int, n_weeks: int
    ) -> dict:
        """
        Holt's double exponential smoothing (level + trend) fallback.

        Captures a linear trend that simple SES misses for growing/declining
        products. α=0.3 weights recent levels; β=0.1 weights trend slowly to
        avoid over-reacting to noise in sparse data.
        Confidence bands are ±1.96σ of the one-step-ahead residuals.
        """
        y = series["y"].values.astype(float)
        alpha = 0.3
        beta = 0.1

        # Initialise level and trend
        level = y[0]
        trend = (y[1] - y[0]) if len(y) > 1 else 0.0

        fitted = np.empty_like(y)
        fitted[0] = level
        for i in range(1, len(y)):
            prev_level = level
            level = alpha * y[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            fitted[i] = level + trend

        residuals = y - fitted
        std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else abs(level) * 0.3

        last_ds = series["ds"].max()
        future_dates = pd.date_range(
            start=last_ds + pd.offsets.Week(1), periods=n_weeks, freq="W-MON"
        )

        periods = []
        for step, d in enumerate(future_dates, start=1):
            forecast_val = max(0.0, level + step * trend)
            # Uncertainty grows with horizon: widen bands proportional to √step
            band = 1.96 * std * (step ** 0.5)
            periods.append(
                {
                    "week_start": d.strftime("%Y-%m-%d"),
                    "forecast_qty": round(forecast_val, 2),
                    "lower_95": round(max(0.0, forecast_val - band), 2),
                    "upper_95": round(forecast_val + band, 2),
                }
            )

        total = round(sum(p["forecast_qty"] for p in periods), 2)

        return {
            "product_id": product_id,
            "method": "holt_smoothing",
            "forecast_periods": periods,
            "total_forecast_qty": total,
            "data_points": len(series),
            "confidence": "low",
        }

    def _empty_result(self, product_id: int) -> dict:
        return {
            "product_id": product_id,
            "method": "no_data",
            "forecast_periods": [],
            "total_forecast_qty": 0.0,
            "data_points": 0,
            "confidence": "low",
        }
