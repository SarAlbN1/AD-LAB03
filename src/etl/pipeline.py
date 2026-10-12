"""Orquestacion del ETL: extract -> transform -> load, con validacion en cada capa.

Idempotente. El cache se decide por el sha256 de los .DBF de origen: si ninguno
cambio y las salidas existen, la segunda corrida no vuelve a procesar.

Uso::

    python -m src.etl.pipeline              # corre lo que falte
    python -m src.etl.pipeline --force      # ignora el cache y reprocesa todo
    python -m src.etl.pipeline --sqlite     # ademas vuelca las gold a SQLite
    python -m src.etl.pipeline --solo-extract
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from src.etl import calendario, extract, load, paths, transform, validate


def _registrar_log(filas: list[dict]) -> None:
    nuevo = pd.DataFrame(filas)
    if paths.ETL_LOG.exists():
        previo = pd.read_csv(paths.ETL_LOG)
        nuevo = pd.concat([previo, nuevo], ignore_index=True)
    paths.ETL_DIR.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(paths.ETL_LOG, index=False)


def _todo_en_cache(manifiesto: pd.DataFrame) -> bool:
    """True si nada cambio en el origen y todas las salidas existen."""
    if not (manifiesto["estado"] == "cache").all():
        return False
    salidas = (
        paths.SILVER_LINE_ITEMS, paths.SILVER_FACTURAS, paths.GOLD_DIM_CALENDARIO,
        paths.GOLD_DIM_PRODUCTO, paths.GOLD_FACT_LINE_ITEMS, paths.GOLD_FACT_FACTURAS,
        paths.GOLD_ONEHOT,
    )
    return all(p.exists() for p in salidas)


def correr(forzar: bool = False, solo_extract: bool = False,
           sqlite: bool = False) -> dict:
    from src.parse import archivos_dbf

    paths.preparar()
    log: list[dict] = []
    inicio_total = time.perf_counter()
    arranque = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def etapa(nombre: str, t0: float, detalle: str = "") -> None:
        log.append({"corrida": arranque, "etapa": nombre,
                    "segundos": round(time.perf_counter() - t0, 2), "detalle": detalle})
        print(f"[pipeline] {nombre} en {time.perf_counter() - t0:.1f}s {detalle}")

    rep = validate.Reporte()

    # ------------------------------------------------------------------- extract
    t0 = time.perf_counter()
    rutas = archivos_dbf()
    manifiesto = extract.extraer_todos(rutas, forzar=forzar)
    etapa("extract", t0, f"{len(manifiesto)} archivos")

    if solo_extract:
        rep.guardar()
        _registrar_log(log)
        return {"manifiesto": manifiesto}

    if not forzar and _todo_en_cache(manifiesto):
        print("[pipeline] nada cambio en el origen y las salidas existen: sin trabajo")
        _registrar_log(log + [{"corrida": arranque, "etapa": "cache",
                               "segundos": 0, "detalle": "sin cambios"}])
        items, facturas = load.leer_gold()
        return {"manifiesto": manifiesto, "items": items, "facturas": facturas,
                "cache": True}

    t0 = time.perf_counter()
    bronze = extract.leer_bronze()
    validate.validar_bronze(manifiesto, bronze, rep)
    validate.columnas_constantes(bronze, rep)
    etapa("validar bronze", t0, f"{len(bronze):,} filas")

    # ----------------------------------------------------------------- transform
    t0 = time.perf_counter()
    dim_producto = transform.construir_catalogo(bronze)
    items, facturas = transform.transformar(bronze, dim_producto)
    transform.guardar_silver(items, facturas)
    etapa("transform", t0, f"{len(items):,} renglones / {len(facturas):,} facturas")

    t0 = time.perf_counter()
    validate.validar_silver(items, facturas, rep)
    etapa("validar silver", t0)

    # ---------------------------------------------------------------------- load
    t0 = time.perf_counter()
    dim_cal = calendario.desde_archivos(rutas)
    fact_items, fact_facturas = load.construir_gold(items, facturas, dim_cal)
    onehot = load.construir_onehot(fact_items)
    load.guardar_gold(dim_cal, dim_producto, fact_items, fact_facturas, onehot,
                      sqlite=sqlite)
    etapa("load", t0, f"one-hot {onehot.shape[0]:,}x{onehot.shape[1]:,}")

    t0 = time.perf_counter()
    validate.validar_gold(dim_cal, fact_items, fact_facturas, rep)
    etapa("validar gold", t0)

    rep.guardar()
    log.append({"corrida": arranque, "etapa": "TOTAL",
                "segundos": round(time.perf_counter() - inicio_total, 2),
                "detalle": f"{len(fact_items):,} renglones"})
    _registrar_log(log)

    print(f"\n[pipeline] completo en {time.perf_counter() - inicio_total:.1f}s")
    print(rep.a_dataframe().to_string(index=False))
    print("\n[pipeline] cobertura del calendario:")
    print(calendario.resumen(dim_cal).to_string(index=False))

    return {
        "manifiesto": manifiesto,
        "dim_calendario": dim_cal,
        "dim_producto": dim_producto,
        "items": fact_items,
        "facturas": fact_facturas,
        "onehot": onehot,
        "calidad": rep.a_dataframe(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL del laboratorio 03")
    parser.add_argument("--force", action="store_true",
                        help="ignora el cache por sha256 y reprocesa todo")
    parser.add_argument("--solo-extract", action="store_true",
                        help="solo la capa bronze")
    parser.add_argument("--sqlite", action="store_true",
                        help="vuelca las tablas gold a SQLite")
    args = parser.parse_args()
    correr(forzar=args.force, solo_extract=args.solo_extract, sqlite=args.sqlite)


if __name__ == "__main__":
    main()
