from typing import Literal
from pydantic import BaseModel


class WeekForecast(BaseModel):
    week_start: str
    forecast_qty: float
    lower_95: float
    upper_95: float


class ProductForecast(BaseModel):
    product_id: int
    product_name: str
    product_code: str
    product_category: str
    method: Literal["prophet", "exponential_smoothing", "no_data"]
    forecast_periods: list[WeekForecast]
    total_forecast_qty: float
    data_points: int
    confidence: Literal["high", "medium", "low"]


class SafetyStockInfo(BaseModel):
    safety_stock: float
    reorder_point: float
    avg_demand_per_period: float
    demand_std: float
    coefficient_of_variation: float
    service_level: float
    lead_time_periods: float


class AbcXyzInfo(BaseModel):
    abc: Literal["A", "B", "C"]
    xyz: Literal["X", "Y", "Z"]
    category: str
    strategy: str


class PurchaseSuggestion(BaseModel):
    product_id: int
    product_name: str
    product_code: str
    product_category: str
    product_unit: str
    current_stock: float
    total_forecast_qty: float
    deficit: float
    customer_count: int
    safety_stock_info: SafetyStockInfo
    abc_xyz: AbcXyzInfo
    forecast_method: Literal["prophet", "exponential_smoothing", "no_data"]
    forecast_confidence: Literal["high", "medium", "low"]
    status: Literal["sufficient", "order_soon", "deficit"]
    forecast_periods: list[WeekForecast]


class AbcXyzResult(BaseModel):
    product_id: int
    product_name: str
    product_code: str
    total_revenue: float
    abc: Literal["A", "B", "C"]
    xyz: Literal["X", "Y", "Z"]
    category: str
    strategy: str
