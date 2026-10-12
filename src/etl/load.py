"""Capa gold: modelo analitico.

Dos dimensiones (`dim_calendario`, `dim_producto`) y dos tablas de hechos
(`fact_line_items`, `fact_facturas`), mas la matriz one-hot que consume mlxtend.

El aporte de esta capa es que las columnas de calendario -- `dia_semana`,
`es_fin_semana`, `tipo_dia`, `semana_iso`, `panel` -- llegan **a los dos niveles de
hechos desde una sola definicion**. Antes `dia_semana` se derivaba en `parse.py` en
ingles y se traducia en `eda.py`: dos lugares para una misma regla.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.etl import paths

#: Columnas de la dimension de calendario que se propagan a los hechos.
COLUMNAS_CALENDARIO = [
    "dia_semana", "dia_num", "es_fin_semana", "tipo_dia", "semana_iso",
    "panel", "es_atipico", "n_dias_observados",
]


def _unir_calendario(hechos: pd.DataFrame, dim_cal: pd.DataFrame) -> pd.DataFrame:
    return hechos.merge(
        dim_cal[["fecha"] + COLUMNAS_CALENDARIO], on="fecha", how="left", validate="m:1"
    )


def construir_gold(items: pd.DataFrame, facturas: pd.DataFrame,
                   dim_cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une los hechos con el calendario."""
    return _unir_calendario(items, dim_cal), _unir_calendario(facturas, dim_cal)


def construir_onehot(fact_items: pd.DataFrame) -> pd.DataFrame:
    """Matriz booleana facturas x productos, entrada unica de apriori y fp-growth.

    Un producto repetido en dos renglones de la misma factura cuenta una sola vez: las
    reglas de asociacion trabajan sobre presencia, no sobre cantidad.
    """
    from mlxtend.preprocessing import TransactionEncoder

    transacciones = [
        list(v) for v in fact_items.groupby("factura")["producto"].unique()
    ]
    te = TransactionEncoder()
    matriz = te.fit(transacciones).transform(transacciones, sparse=False)
    return pd.DataFrame(matriz, columns=te.columns_, dtype=bool)


def guardar_gold(dim_cal: pd.DataFrame, dim_prod: pd.DataFrame,
                 fact_items: pd.DataFrame, fact_fact: pd.DataFrame,
                 onehot: pd.DataFrame | None = None, sqlite: bool = False) -> None:
    paths.GOLD.mkdir(parents=True, exist_ok=True)
    dim_cal.to_parquet(paths.GOLD_DIM_CALENDARIO, index=False)
    dim_prod.to_parquet(paths.GOLD_DIM_PRODUCTO, index=False)
    fact_items.to_parquet(paths.GOLD_FACT_LINE_ITEMS, index=False)
    fact_fact.to_parquet(paths.GOLD_FACT_FACTURAS, index=False)
    print(f"[load] gold -> dim_calendario({len(dim_cal)}), dim_producto({len(dim_prod):,}), "
          f"fact_line_items({len(fact_items):,}), fact_facturas({len(fact_fact):,})")

    if onehot is not None:
        onehot.to_parquet(paths.GOLD_ONEHOT, index=False)
        print(f"[load] one-hot -> {onehot.shape[0]:,} x {onehot.shape[1]:,}")

    if sqlite:
        volcar_sqlite(dim_cal, dim_prod, fact_items, fact_fact)


def volcar_sqlite(dim_cal: pd.DataFrame, dim_prod: pd.DataFrame,
                  fact_items: pd.DataFrame, fact_fact: pd.DataFrame) -> None:
    """Las tablas gold en SQLite, para poder revisarlas con SQL. Opcional."""
    with sqlite3.connect(paths.SQLITE) as con:
        for nombre, df in (
            ("dim_calendario", dim_cal),
            ("dim_producto", dim_prod),
            ("fact_line_items", fact_items),
            ("fact_facturas", fact_fact),
        ):
            copia = df.copy()
            for col in copia.columns:
                if str(copia[col].dtype) in ("category", "object"):
                    copia[col] = copia[col].astype(str)
            copia.to_sql(nombre, con, if_exists="replace", index=False)
    print(f"[load] sqlite -> {paths.SQLITE.relative_to(paths.RAIZ)}")


def leer_gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(fact_line_items, fact_facturas)`` con el dia de la semana ya resuelto."""
    from src.etl.calendario import DIAS_ES

    items = pd.read_parquet(paths.GOLD_FACT_LINE_ITEMS)
    facturas = pd.read_parquet(paths.GOLD_FACT_FACTURAS)
    for df in (items, facturas):
        df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=DIAS_ES,
                                          ordered=True)
    return items, facturas


def leer_dim_calendario() -> pd.DataFrame:
    from src.etl.calendario import DIAS_ES

    dim = pd.read_parquet(paths.GOLD_DIM_CALENDARIO)
    dim["dia_semana"] = pd.Categorical(dim["dia_semana"], categories=DIAS_ES, ordered=True)
    return dim
