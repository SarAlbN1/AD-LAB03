"""Rutas y nombres de las capas del ETL, en un solo lugar."""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent

ETL_DIR = RAIZ / "outputs" / "etl"
BRONZE = ETL_DIR / "bronze"
SILVER = ETL_DIR / "silver"
GOLD = ETL_DIR / "gold"

MANIFIESTO = ETL_DIR / "manifest.csv"
ETL_LOG = ETL_DIR / "etl_log.csv"
SQLITE = ETL_DIR / "lab03.sqlite"

SILVER_LINE_ITEMS = SILVER / "line_items.parquet"
SILVER_FACTURAS = SILVER / "facturas.parquet"

GOLD_DIM_CALENDARIO = GOLD / "dim_calendario.parquet"
GOLD_DIM_PRODUCTO = GOLD / "dim_producto.parquet"
GOLD_FACT_LINE_ITEMS = GOLD / "fact_line_items.parquet"
GOLD_FACT_FACTURAS = GOLD / "fact_facturas.parquet"
GOLD_ONEHOT = GOLD / "onehot.parquet"

TABLAS = RAIZ / "outputs" / "tables"
CALIDAD = TABLAS / "00_calidad_datos.csv"


def preparar() -> None:
    """Crea la estructura de carpetas del ETL."""
    for d in (BRONZE, SILVER, GOLD, TABLAS):
        d.mkdir(parents=True, exist_ok=True)
