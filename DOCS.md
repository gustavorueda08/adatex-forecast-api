# adatex-forecast-api — Documentación Técnica

Microservicio Python de predicción de demanda avanzada para la plataforma Adatex Warehouse. Expone una API REST consumida por el frontend Next.js y aplica modelos estadísticos y de machine learning sobre el historial de ventas leído desde Strapi.

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Instalación y Configuración](#3-instalación-y-configuración)
4. [Variables de Entorno](#4-variables-de-entorno)
5. [Flujo de Datos](#5-flujo-de-datos)
6. [Módulos y Modelos](#6-módulos-y-modelos)
   - [StrapiClient](#61-strapiclient)
   - [DemandForecaster](#62-demandforecaster)
   - [Safety Stock](#63-safety-stock)
   - [ABC-XYZ](#64-abc-xyz)
7. [Endpoints de la API](#7-endpoints-de-la-api)
8. [Esquemas de Respuesta](#8-esquemas-de-respuesta)
9. [Caché en Memoria](#9-caché-en-memoria)
10. [Autenticación](#10-autenticación)
11. [Integración con el Frontend](#11-integración-con-el-frontend)

---

## 1. Visión General

`adatex-forecast-api` es un servicio independiente (puerto **8000** por defecto) que:

- Lee el historial de órdenes directamente desde Strapi 5 via REST.
- Aplica el modelo **Prophet** (Meta/Facebook) para generar predicciones semanales de demanda con intervalos de confianza al 95%.
- Calcula **stock de seguridad** y **punto de reorden** usando la fórmula estadística estándar.
- Clasifica productos en la matriz **ABC-XYZ** combinando valor de revenue con variabilidad de demanda.
- Expone todo como una API REST con autenticación por API key.

El frontend Next.js **no llama directamente** a este servicio. Las llamadas pasan por una ruta proxy en Next.js (`/api/forecast/[...path]`) que agrega la API key y verifica la sesión del usuario.

```
Browser → Next.js (/api/forecast/*) → adatex-forecast-api:8000 → Strapi:1337
```

---

## 2. Estructura del Proyecto

```
adatex-forecast-api/
├── app/
│   ├── main.py               # Aplicación FastAPI, CORS, registro de routers
│   ├── config.py             # Configuración via pydantic-settings (.env)
│   ├── data/
│   │   └── strapi_client.py  # Cliente HTTP para Strapi REST API
│   ├── models/
│   │   ├── forecaster.py     # DemandForecaster (Prophet + fallback)
│   │   ├── safety_stock.py   # Cálculo de stock de seguridad y ROP
│   │   └── abc_xyz.py        # Clasificación ABC-XYZ
│   ├── routers/
│   │   ├── forecasts.py      # Todos los endpoints de predicción
│   │   └── health.py         # GET /health
│   └── schemas/
│       └── forecast.py       # Modelos Pydantic de respuesta
├── requirements.txt
├── .env                      # Variables de entorno (no subir a git)
├── .gitignore
└── DOCS.md                   # Este archivo
```

---

## 3. Instalación y Configuración

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# Instalar dependencias
pip install -r requirements.txt

# Prophet requiere toolchain C++ en algunos sistemas
# Si falla: conda install -c conda-forge prophet

# Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores reales

# Iniciar el servidor (modo desarrollo con hot-reload)
python -m app.main
# → http://0.0.0.0:8000
```

---

## 4. Variables de Entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `STRAPI_URL` | `http://localhost:1337` | URL base del backend Strapi |
| `STRAPI_API_TOKEN` | _(requerido)_ | Token de API de Strapi (tipo "Full access" o con permisos de lectura sobre orders, products, items) |
| `FORECAST_API_KEY` | _(vacío = sin auth)_ | API key que el proxy Next.js debe enviar en el header `X-API-Key` |
| `FORECAST_HORIZON_DAYS` | `90` | Horizonte de predicción en días (se redondea a semanas completas) |
| `DEFAULT_LEAD_TIME_DAYS` | `30` | Tiempo de reabastecimiento por defecto en días |
| `DEFAULT_SERVICE_LEVEL` | `0.95` | Nivel de servicio por defecto para stock de seguridad (0.80–0.99) |
| `CACHE_TTL_SECONDS` | `3600` | Tiempo de vida del caché en memoria (1 hora por defecto) |
| `PORT` | `8000` | Puerto en que escucha el servidor |

---

## 5. Flujo de Datos

### Flujo completo de una petición a `/forecasts/purchase-suggestions`

```
1. Next.js proxy verifica cookie de sesión del usuario
2. Next.js reenvía GET /forecasts/purchase-suggestions con header X-API-Key
3. FastAPI valida la API key (_require_api_key)
4. Se consulta el caché en memoria → si hay hit, se devuelve directamente

5. StrapiClient hace 3 llamadas paralelas a Strapi (asyncio.gather):
   a. GET /api/orders?filters[type][$in][0]=sale&...  → historial de órdenes
   b. GET /api/items?filters[state][$in][0]=available&...  → stock actual
   c. GET /api/products?filters[isActive][$eq]=true  → catálogo de productos

6. Por cada producto con historial de ventas:
   a. DemandForecaster construye serie semanal (W-MON) con pandas resample
   b. Si hay ≥ 8 semanas con demanda no-cero → Prophet fit + predict
      Si no → Exponential Smoothing fallback
   c. safety_stock.calculate() con la serie semanal y el lead time
   d. abc_xyz.classify() con revenue total y demanda mensual

7. Se construye lista de PurchaseSuggestion con status:
   - deficit      → forecast > stock actual
   - order_soon   → stock cubre < 150% del forecast
   - sufficient   → stock ≥ 150% del forecast

8. Resultado ordenado y guardado en caché
9. Respuesta JSON al proxy Next.js → al browser
```

### Paginación con Strapi

Strapi 5 limita los resultados a `maxLimit: 100` por página. `StrapiClient._get_all()` pagina automáticamente hasta obtener todos los registros usando `meta.pagination.pageCount`.

---

## 6. Módulos y Modelos

### 6.1 StrapiClient

**Archivo:** `app/data/strapi_client.py`

Maneja toda la comunicación con Strapi. Autentica con `Authorization: Bearer <STRAPI_API_TOKEN>`.

| Método | Descripción |
|---|---|
| `_get_all(path, params)` | Pagina automáticamente hasta traer todos los registros. Loguea el body de error si Strapi responde con ≥ 400. |
| `get_order_history(months_back=24)` | Órdenes de tipo `sale` o `partial-invoice` en estado `completed` o `processing` de los últimos N meses. Devuelve lista plana de líneas de producto. |
| `get_products()` | Todos los productos con `isActive=true`. |
| `get_current_stock()` | Stock actual (ítems en estado `available` o `reserved`) agrupado por `product_id`. |

**Estructura de una línea de orden devuelta por `get_order_history`:**

```python
{
    "order_id": 123,
    "order_date": date(2024, 11, 15),
    "customer_id": 45,
    "customer_name": "Empresa XYZ",
    "product_id": 7,
    "product_name": "Tela Algodón",
    "product_code": "TC-001",
    "product_category": "metros",
    "qty": 150.0,
    "price": 12500.0,
    "revenue": 1875000.0,
}
```

---

### 6.2 DemandForecaster

**Archivo:** `app/models/forecaster.py`

Genera predicciones semanales de demanda para un producto. Elige automáticamente entre Prophet y exponential smoothing según la densidad de datos históricos.

#### Método principal

```python
forecaster.forecast_product(order_lines, product_id, horizon_days=90)
```

- `order_lines`: lista completa de líneas (todos los productos); el método filtra internamente por `product_id`.
- `horizon_days`: días a predecir, redondeado hacia arriba a semanas completas.

#### Decisión de modelo

| Condición | Modelo |
|---|---|
| ≥ 8 semanas con demanda > 0 | **Prophet** |
| < 8 semanas con demanda > 0 | **Exponential Smoothing** |

#### Prophet (modelo primario)

Configuración:

| Parámetro | Valor | Razón |
|---|---|---|
| `yearly_seasonality` | `True` | Captura ciclos anuales (temporadas) |
| `weekly_seasonality` | `False` | Innecesario con datos semanales |
| `daily_seasonality` | `False` | Innecesario con datos semanales |
| `seasonality_mode` | `multiplicative` | La varianza crece con la demanda (típico en productos de moda/textil) |
| `interval_width` | `0.95` | Intervalos de confianza al 95% |
| `changepoint_prior_scale` | `0.05` | Prior conservador: evita overfitting en 1-2 años de datos |
| `mcmc_samples` | `0` | Estimación MAP (mucho más rápida que MCMC) |

Estacionalidad mensual personalizada: `period=30.5 días`, `fourier_order=5`.

Los valores negativos predichos se recortan a `0.0` con `.clip(lower=0)`.

**Nivel de confianza del forecast:**

| Semanas de historia | Confianza |
|---|---|
| ≥ 52 | `high` |
| ≥ 26 | `medium` |
| < 26 | `low` |

#### Exponential Smoothing (fallback)

- Alpha `α = 0.3` (peso moderado hacia observaciones recientes).
- El nivel proyectado es constante para todo el horizonte (`level` = último valor suavizado).
- Bandas de confianza: `±1.96σ` de los residuos históricos.
- Siempre devuelve `confidence = "low"`.

#### Estructura de retorno

```python
{
    "product_id": 7,
    "method": "prophet",          # "prophet" | "exponential_smoothing" | "no_data"
    "forecast_periods": [
        {
            "week_start": "2025-01-06",
            "forecast_qty": 143.5,
            "lower_95": 98.2,
            "upper_95": 188.8,
        },
        # ... N semanas
    ],
    "total_forecast_qty": 1850.0, # suma de forecast_qty
    "data_points": 48,            # semanas en el histórico
    "confidence": "medium",
}
```

---

### 6.3 Safety Stock

**Archivo:** `app/models/safety_stock.py`

Calcula el stock de seguridad y punto de reorden usando la fórmula estadística completa.

#### Fórmula

```
Safety Stock  = z × √(L × σ_d²  +  μ_d² × σ_L²)
Reorder Point = μ_d × L + Safety Stock

Donde:
  z    = z-score del nivel de servicio
  L    = lead time medio en periodos
  σ_d  = desviación estándar de la demanda por periodo
  μ_d  = demanda media por periodo
  σ_L  = desviación estándar del lead time (0 si es fijo)
```

La unidad temporal es **semanas** (misma unidad que la serie de demanda).

#### Z-scores disponibles

| Nivel de Servicio | Z-score |
|---|---|
| 80% | 0.842 |
| 85% | 1.036 |
| 90% | 1.282 |
| **95%** | **1.645** ← por defecto |
| 98% | 2.054 |
| 99% | 2.326 |

#### Retorno de `calculate()`

```python
{
    "safety_stock": 245.8,
    "reorder_point": 890.3,
    "avg_demand_per_period": 130.2,   # μ_d (semanas)
    "demand_std": 48.5,               # σ_d
    "coefficient_of_variation": 0.37, # σ_d / μ_d
    "service_level": 0.95,
    "lead_time_periods": 4.28,        # lead time en semanas
}
```

---

### 6.4 ABC-XYZ

**Archivo:** `app/models/abc_xyz.py`

Clasifica productos en una matriz 3×3 combinando valor de revenue (ABC) con variabilidad de demanda (XYZ).

#### Clasificación ABC (por revenue acumulado)

| Categoría | Criterio | Descripción |
|---|---|---|
| **A** | Primeros 80% del revenue | Alto valor — prioridad máxima |
| **B** | Entre 80% y 95% | Valor medio |
| **C** | Restante 5% | Bajo valor |

#### Clasificación XYZ (por Coeficiente de Variación mensual)

| Categoría | CV | Descripción |
|---|---|---|
| **X** | CV < 0.5 | Demanda estable y predecible |
| **Y** | 0.5 ≤ CV < 1.0 | Variabilidad media |
| **Z** | CV ≥ 1.0 | Demanda esporádica / impredecible |

> Si un producto tiene menos de 3 meses de historia se clasifica automáticamente como **Z**.

#### Estrategias de reposición (matriz 3×3)

| | X (estable) | Y (variable) | Z (esporádico) |
|---|---|---|---|
| **A** | Stock alto, reposición automática | Buffer moderado, revisión quincenal | Análisis manual, stock de seguridad alto |
| **B** | Reposición periódica estándar | Revisión mensual | Pedido bajo demanda, mínimos bajos |
| **C** | Stock mínimo, reposición simple | Stock bajo | Evaluar descontinuación |

---

## 7. Endpoints de la API

Todos los endpoints requieren el header `X-API-Key: <FORECAST_API_KEY>` (excepto si `FORECAST_API_KEY` está vacío en `.env`).

### `GET /health`

Verificación de disponibilidad. No requiere autenticación.

**Respuesta:**
```json
{ "status": "ok", "service": "adatex-forecast-api" }
```

---

### `GET /forecasts/purchase-suggestions`

Panel principal de reabastecimiento. Devuelve todos los productos con historial de ventas, su forecast, stock de seguridad, clasificación ABC-XYZ y estado de reposición.

**Parámetros de query:**

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `horizon_days` | `90` | Días a predecir |
| `lead_time_days` | `DEFAULT_LEAD_TIME_DAYS` | Lead time del proveedor |
| `service_level` | `DEFAULT_SERVICE_LEVEL` | Nivel de servicio para SS |
| `months_back` | `24` | Meses de historial a usar |

**Los resultados se cachean** durante `CACHE_TTL_SECONDS`.

**Ordenamiento de respuesta:** `deficit` → `order_soon` → `sufficient`, y dentro de cada grupo por revenue descendente.

**Respuesta:** lista de `PurchaseSuggestion` (ver sección 8).

---

### `GET /forecasts/products/{product_id}`

Forecast detallado semanal para un producto específico con bandas de confianza al 95%.

**Parámetros de query:**

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `horizon_days` | `90` | Días a predecir |
| `months_back` | `24` | Meses de historial |

**Respuesta:** `ProductForecast` (ver sección 8).

---

### `GET /forecasts/customers/{customer_id}`

Todos los forecasts de productos basados en el historial de compras de un cliente específico. Útil para la página de detalle de cliente.

**Parámetros de query:**

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `horizon_days` | `90` | Días a predecir |
| `months_back` | `24` | Meses de historial |

El forecast se calcula usando **solo las líneas de ese cliente**, no el historial global del producto.

**Respuesta:** lista de `ProductForecast`, ordenada por `total_forecast_qty` descendente.

---

### `GET /forecasts/abc-xyz`

Clasificación ABC-XYZ completa de todos los productos con historial de ventas.

**Parámetros de query:**

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `months_back` | `24` | Meses de historial |

**Los resultados se cachean** durante `CACHE_TTL_SECONDS`.

**Respuesta:** lista de `AbcXyzResult` (ver sección 8).

---

### `POST /forecasts/refresh-cache`

Invalida el caché en memoria para que la próxima petición recalcule todo.

**Respuesta:**
```json
{ "cleared": true }
```

---

## 8. Esquemas de Respuesta

### `WeekForecast`

```json
{
    "week_start": "2025-03-03",
    "forecast_qty": 143.5,
    "lower_95": 98.2,
    "upper_95": 188.8
}
```

### `ProductForecast`

```json
{
    "product_id": 7,
    "product_name": "Tela Algodón",
    "product_code": "TC-001",
    "product_category": "metros",
    "method": "prophet",
    "forecast_periods": [ /* lista de WeekForecast */ ],
    "total_forecast_qty": 1850.0,
    "data_points": 48,
    "confidence": "medium"
}
```

`method`: `"prophet"` | `"exponential_smoothing"` | `"no_data"`
`confidence`: `"high"` | `"medium"` | `"low"`

### `PurchaseSuggestion`

```json
{
    "product_id": 7,
    "product_name": "Tela Algodón",
    "product_code": "TC-001",
    "product_category": "metros",
    "product_unit": "m",
    "current_stock": 320.0,
    "total_forecast_qty": 1850.0,
    "deficit": 1530.0,
    "customer_count": 12,
    "safety_stock_info": {
        "safety_stock": 245.8,
        "reorder_point": 890.3,
        "avg_demand_per_period": 130.2,
        "demand_std": 48.5,
        "coefficient_of_variation": 0.37,
        "service_level": 0.95,
        "lead_time_periods": 4.28
    },
    "abc_xyz": {
        "abc": "A",
        "xyz": "Y",
        "category": "AY",
        "strategy": "Alto valor, variabilidad media — buffer moderado, revisión quincenal"
    },
    "forecast_method": "prophet",
    "forecast_confidence": "medium",
    "status": "deficit",
    "forecast_periods": [ /* lista de WeekForecast */ ]
}
```

`status`: `"deficit"` | `"order_soon"` | `"sufficient"`

**Lógica de status:**

| Condición | Status |
|---|---|
| `forecast ≤ stock` | `sufficient` |
| `stock ≥ 50% del forecast` | `order_soon` |
| `stock < 50% del forecast` | `deficit` |

### `AbcXyzResult`

```json
{
    "product_id": 7,
    "product_name": "Tela Algodón",
    "product_code": "TC-001",
    "total_revenue": 48500000.0,
    "abc": "A",
    "xyz": "Y",
    "category": "AY",
    "strategy": "Alto valor, variabilidad media — buffer moderado, revisión quincenal"
}
```

---

## 9. Caché en Memoria

El servicio usa un caché en memoria simple (dict de Python) para evitar recalcular Prophet en cada petición, ya que el ajuste del modelo tarda entre 0.5 y 2 segundos por producto.

```python
_cache: dict[str, tuple[float, object]] = {}  # key → (timestamp, data)
```

**Entradas en caché:**
- `purchase_suggestions:{horizon}:{lead_time}:{service_level}:{months_back}`
- `abc_xyz:{months_back}`

**TTL:** configurable via `CACHE_TTL_SECONDS` (default: 3600 segundos = 1 hora).

**Invalidar manualmente:** `POST /forecasts/refresh-cache`

> El caché se pierde al reiniciar el proceso. Para producción, considerar Redis u otro caché persistente si el tiempo de arranque (recalcular todo) es inaceptable.

---

## 10. Autenticación

### Hacia Strapi (saliente)

El cliente HTTP incluye `Authorization: Bearer <STRAPI_API_TOKEN>` en cada petición. El token debe tener permisos de lectura sobre:
- `api::order.order` (find)
- `api::product.product` (find)
- `api::item.item` (find)

### Desde Next.js (entrante)

El header `X-API-Key` debe coincidir con `FORECAST_API_KEY`. Si `FORECAST_API_KEY` está vacío en `.env`, la validación se omite (útil en desarrollo local).

---

## 11. Integración con el Frontend

### Proxy Next.js

**Archivo:** `adatex-warehouse/src/app/api/forecast/[...path]/route.js`

Catch-all route que:
1. Verifica la cookie de sesión del usuario (`adatex_session`).
2. Si el usuario no está autenticado → `401`.
3. Reenvía la petición a `FORECAST_API_URL/forecasts/{path}` con el header `X-API-Key`.
4. Si el servicio Python no está disponible → `503` (el frontend lo maneja mostrando un aviso).

### Llamadas desde el browser

```
GET /api/forecast/purchase-suggestions
GET /api/forecast/products/{id}
GET /api/forecast/customers/{id}
GET /api/forecast/abc-xyz
POST /api/forecast/refresh-cache
```

### Componentes del frontend que consumen esta API

| Componente | Endpoint |
|---|---|
| `src/app/(auth)/(protected)/purchase-suggestions/page.js` | `GET /purchase-suggestions` |
| `src/components/forecast/CustomerForecastSection.jsx` | `GET /customers/{id}` |
