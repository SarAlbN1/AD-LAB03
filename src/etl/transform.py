"""Capa silver: bronze crudo -> renglones y facturas limpios.

Aqui viven **todas** las reglas de negocio del laboratorio, y cada una queda explicita
para poder defenderla en el reporte:

1. **Clave de factura** ``archivo|caja|tipo_doc|consecutivo``. El consecutivo se
   reinicia por caja y por tipo de documento, asi que `FACNRO` por si solo no
   identifica una factura.
2. **Nombre canonico de producto**: se normaliza el texto y se canoniza por `CODIGO`,
   porque 23 codigos aparecen con mas de una descripcion. Sin esto el mismo producto
   cuenta como dos items y su soporte se reparte a la mitad.
3. **Hora estimada**: los DBF no traen hora. Se estima desde el orden fisico del
   registro dentro de cada caja (ver `src.parse.asignar_hora_estimada`). Queda marcada
   con `hora_estimada=True` para que ninguna lectura la confunda con un dato medido.
4. **Tipos de documento**: se conservan todos (FA, CO, AS, AE, DC). El filtro queda
   para el analisis, que tiene la columna `tipo_doc`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd

from src.etl import paths
from src.parse import JORNADA, fecha_desde_nombre, normalizar_producto


def construir_catalogo(bronze: pd.DataFrame) -> pd.DataFrame:
    """``dim_producto``: un nombre canonico por codigo, mas trazas de la ambiguedad.

    El nombre canonico es la descripcion mas frecuente del codigo. `n_descripciones`
    deja constancia de cuantas variantes tenia, que es un dato de calidad del maestro
    de productos de la tienda.
    """
    tabla = bronze[["CODIGO", "DESCRI"]].copy()
    tabla["codigo"] = tabla["CODIGO"].fillna("").astype(str).str.strip()
    tabla["nombre"] = tabla["DESCRI"].fillna("").map(normalizar_producto)
    tabla = tabla[(tabla["codigo"] != "") & (tabla["nombre"] != "")]

    variantes: dict[str, Counter] = defaultdict(Counter)
    for codigo, nombre in zip(tabla["codigo"], tabla["nombre"]):
        variantes[codigo][nombre] += 1

    filas = []
    for codigo, cuenta in variantes.items():
        canonico, apariciones = cuenta.most_common(1)[0]
        filas.append(
            {
                "codigo": codigo,
                "producto": canonico,
                "n_descripciones": len(cuenta),
                "apariciones": int(sum(cuenta.values())),
                "variantes": " | ".join(sorted(n for n in cuenta if n != canonico)),
                "apariciones_canonico": int(apariciones),
            }
        )
    return pd.DataFrame(filas).sort_values("apariciones", ascending=False).reset_index(
        drop=True
    )


def _hora_estimada(facturas: pd.DataFrame, jornada: tuple[int, int]) -> pd.Series:
    """Reparte las facturas de cada caja sobre la jornada, en orden de facturacion.

    Devuelve la marca de tiempo estimada. Es un proxy ordinal: sirve para la forma de
    la curva a lo largo del dia, no para afirmar horas absolutas.
    """
    inicio, fin = jornada
    duracion = (fin - inicio) * 3600

    marcas = pd.Series(pd.NaT, index=facturas.index, dtype="datetime64[ns]")
    for (archivo, _caja), grupo in facturas.groupby(["archivo", "caja"], observed=True):
        dia = fecha_desde_nombre(archivo)
        base = datetime(dia.year, dia.month, dia.day, inicio)
        orden = grupo.sort_values("orden_fisico").index
        n = len(orden)
        for rango, idx in enumerate(orden):
            # +0.5 centra cada factura en su porcion de jornada: ni la primera cae en
            # la apertura exacta ni la ultima en el cierre.
            marcas.loc[idx] = base + timedelta(seconds=duracion * (rango + 0.5) / n)
    return marcas


def transformar(bronze: pd.DataFrame, dim_producto: pd.DataFrame,
                jornada: tuple[int, int] = JORNADA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bronze -> ``(line_items, facturas)`` de la capa silver."""
    df = bronze.copy()

    df["codigo"] = df["CODIGO"].fillna("").astype(str).str.strip()
    nombre_directo = df["DESCRI"].fillna("").map(normalizar_producto)
    canonico = df["codigo"].map(dim_producto.set_index("codigo")["producto"])
    # El canonico del catalogo manda; si el codigo viene vacio, se usa la descripcion.
    df["producto"] = canonico.fillna(nombre_directo)
    df.loc[df["producto"].isna(), "producto"] = nombre_directo

    sin_producto = int((df["producto"].fillna("").str.len() == 0).sum())
    if sin_producto:
        print(f"[transform] {sin_producto} renglones sin producto identificable, descartados")
        df = df[df["producto"].fillna("").str.len() > 0]

    df["caja"] = df["CODCAJ"].fillna(0).astype("int64").astype(str)
    df["tipo_doc"] = df["CODTRA"].fillna("").astype(str).str.strip()
    df["consecutivo"] = df["FACNRO"].fillna(0).astype("int64")
    df["archivo"] = df["_archivo"]
    df["fecha"] = df["archivo"].map(lambda a: fecha_desde_nombre(a))
    df["factura"] = (
        df["archivo"].str.replace(".DBF", "", regex=False)
        + "|" + df["caja"] + "|" + df["tipo_doc"] + "|" + df["consecutivo"].astype(str)
    )

    df["cantidad"] = df["CANT"].fillna(0.0).astype(float)
    df["precio_unitario"] = df["PRECIO"].fillna(0.0).astype(float)
    df["subtotal"] = df["TOTAL"].fillna(0.0).astype(float)
    df["iva"] = df["IVA"].fillna(0.0).astype(float)
    df["descuento"] = df["VLRDESC"].fillna(0.0).astype(float)
    df["tasa_iva"] = df["TASIVA"].fillna(0.0).astype(float)
    df["unidad"] = df["UNIDAD"].fillna("").astype(str).str.strip()
    df["costo"] = df["COSPRO"].fillna(0.0).astype(float)

    # Control de integridad: lo que deberia valer el renglon vs lo que dice el DBF.
    # TOTAL viene redondeado al peso, de ahi la tolerancia.
    df["descuadre"] = (
        df["cantidad"] * df["precio_unitario"] - df["descuento"] - df["subtotal"]
    ).abs()
    df["cuadra"] = df["descuadre"] <= 1.0

    columnas_items = [
        "factura", "archivo", "fecha", "caja", "tipo_doc", "consecutivo", "_fila",
        "producto", "codigo", "cantidad", "precio_unitario", "subtotal", "iva",
        "descuento", "tasa_iva", "unidad", "costo", "descuadre", "cuadra",
    ]
    items = df[columnas_items].rename(columns={"_fila": "orden_fisico"})
    items = items.reset_index(drop=True)
    items["linea"] = items.groupby("factura").cumcount() + 1

    # ------------------------------------------------------------------ facturas
    agrupado = items.groupby("factura", sort=False)
    facturas = agrupado.agg(
        archivo=("archivo", "first"),
        fecha=("fecha", "first"),
        caja=("caja", "first"),
        tipo_doc=("tipo_doc", "first"),
        consecutivo=("consecutivo", "first"),
        orden_fisico=("orden_fisico", "min"),
        n_lineas=("producto", "size"),
        n_productos=("producto", "nunique"),
        total_items=("cantidad", "sum"),
        total=("subtotal", "sum"),
        total_impuesto=("iva", "sum"),
        total_descuento=("descuento", "sum"),
        cuadra=("cuadra", "all"),
    ).reset_index()
    facturas["es_unitaria"] = facturas["n_productos"] == 1

    facturas["timestamp"] = _hora_estimada(facturas, jornada)
    facturas["hora"] = facturas["timestamp"].dt.hour
    facturas["hora_estimada"] = True

    # La marca de tiempo se calcula una sola vez, a nivel factura, y baja al renglon.
    items = items.merge(facturas[["factura", "timestamp", "hora"]], on="factura",
                        how="left")

    return items, facturas


def guardar_silver(items: pd.DataFrame, facturas: pd.DataFrame) -> None:
    paths.SILVER.mkdir(parents=True, exist_ok=True)
    items.to_parquet(paths.SILVER_LINE_ITEMS, index=False)
    facturas.to_parquet(paths.SILVER_FACTURAS, index=False)
    print(f"[transform] silver -> {len(items):,} renglones, {len(facturas):,} facturas")


def leer_silver() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(paths.SILVER_LINE_ITEMS),
        pd.read_parquet(paths.SILVER_FACTURAS),
    )
