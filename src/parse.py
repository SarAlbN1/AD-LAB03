"""De los .DBF crudos a objetos `Factura`, y de ahi a DataFrames con cache en parquet.

Decisiones de este modulo, todas visibles para el reporte:

1. **Clave de factura**: ``archivo|CODCAJ|CODTRA|FACNRO``. El consecutivo se reinicia
   por caja y por tipo de documento, asi que FACNRO solo no identifica una factura.
2. **Tipos de documento**: se incluyen todos los CODTRA (FA, CO, AS, AE, DC). El modulo
   expone la columna `tipo_doc` para poder repetir el analisis solo con FA como anexo
   de sensibilidad.
3. **Hora estimada**: los DBF **no traen fecha-hora** (FECVEN esta vacio en los 147.413
   registros y MESANRO es 0). La fecha se deduce del nombre del archivo y la hora se
   estima con un proxy ordinal. Ver `asignar_hora_estimada`.
4. **Nombre de producto**: se canoniza por CODIGO, porque 23 codigos aparecen con mas
   de una descripcion y de otro modo el mismo producto contaria como dos items
   distintos, repartiendo su soporte.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.dbf_reader import DBFTable
from src.models import Factura, LineItem

# ------------------------------------------------------------------ configuracion

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", RAIZ / "archivos"))
OUTPUT_DIR = RAIZ / "outputs"

#: Jornada supuesta de la tienda para la hora estimada (ver `asignar_hora_estimada`).
JORNADA = (7, 22)

#: Tipo de documento que corresponde a una venta al consumidor final. El analisis
#: trabaja con **todos** los tipos (FA, CO, AS, AE, DC); esta constante existe para el
#: anexo de sensibilidad que mide cuanto cambian los resultados al quedarse solo con
#: las ventas (ver `src.eda.sensibilidad_tipo_documento`).
TIPO_VENTA = "FA"

#: fc01042024.DBF -> 2024-04-01 (ddmmyyyy). Verificado contra la carpeta: los archivos
#: de `lunes/` caen en lunes, los de `domingo/` en domingo.
_PATRON_NOMBRE = re.compile(r"fc(\d{2})(\d{2})(\d{4})", re.IGNORECASE)

_ESPACIOS = re.compile(r"\s+")


# --------------------------------------------------------------------- utilidades


def archivos_dbf(data_dir: Path | None = None) -> list[Path]:
    """Todos los .DBF del conjunto, en orden cronologico."""
    base = data_dir or DATA_DIR
    rutas = sorted(base.glob("*/*.DBF"), key=lambda p: (fecha_desde_nombre(p), p.name))
    if not rutas:
        raise FileNotFoundError(f"No hay .DBF bajo {base}")
    return rutas


def fecha_desde_nombre(path: Path | str) -> date:
    """Extrae la fecha del nombre del archivo (unica fuente de fecha disponible)."""
    m = _PATRON_NOMBRE.search(Path(path).stem)
    if not m:
        raise ValueError(f"Nombre de archivo sin fecha reconocible: {path}")
    dd, mm, yyyy = (int(g) for g in m.groups())
    return date(yyyy, mm, dd)


def sin_acentos(texto: str) -> str:
    """Version comparable de un texto: sin tildes ni virgulillas.

    Se usa solo para agrupar; lo que se muestra conserva los acentos originales.
    """
    desc = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in desc if not unicodedata.combining(c))


def normalizar_producto(descripcion: str) -> str:
    """Limpia una descripcion de producto conservando su forma legible."""
    texto = _ESPACIOS.sub(" ", descripcion.replace("\x00", " ")).strip()
    # Hay descripciones que terminan en punto o en separadores sueltos del sistema
    # de punto de venta: no son parte del nombre.
    texto = texto.strip(" .-_/*")
    return texto.upper()


def clave_producto(descripcion: str) -> str:
    """Clave de agrupacion: sin acentos y sin espacios redundantes."""
    return sin_acentos(normalizar_producto(descripcion))


# ------------------------------------------------------------- lectura y mapeo


def _fila_a_lineitem(fila: dict[str, Any], nombre: str) -> LineItem:
    return LineItem(
        producto=nombre,
        codigo=(fila.get("CODIGO") or "").strip(),
        cantidad=float(fila.get("CANT") or 0.0),
        precio_unitario=float(fila.get("PRECIO") or 0.0),
        subtotal=float(fila.get("TOTAL") or 0.0),
        iva=float(fila.get("IVA") or 0.0),
        descuento=float(fila.get("VLRDESC") or 0.0),
        tasa_iva=float(fila.get("TASIVA") or 0.0),
        unidad=(fila.get("UNIDAD") or "").strip(),
    )


def construir_catalogo(filas: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Mapa ``CODIGO -> nombre canonico`` (la descripcion mas frecuente del codigo).

    Resuelve los codigos que el sistema registro con varias descripciones. Sin esto,
    'JUGO HIT 500 ML' y 'JUGO HIT  500 ML' serian dos productos y cada uno tendria la
    mitad del soporte real.
    """
    por_codigo: dict[str, Counter] = defaultdict(Counter)
    for fila in filas:
        codigo = (fila.get("CODIGO") or "").strip()
        nombre = normalizar_producto(fila.get("DESCRI") or "")
        if codigo and nombre:
            por_codigo[codigo][nombre] += 1
    return {
        codigo: nombres.most_common(1)[0][0] for codigo, nombres in por_codigo.items()
    }


def asignar_hora_estimada(
    facturas: Sequence[Factura], jornada: tuple[int, int] = JORNADA
) -> None:
    """Asigna una **hora estimada** a cada factura (proxy ordinal).

    Los DBF no tienen campo de hora. Lo que si existe es orden: dentro de un archivo
    (un dia) y de una caja, los registros estan grabados en el orden en que se
    facturo y el consecutivo FACNRO crece. Entonces, para cada caja, las N facturas
    del dia se reparten uniformemente sobre la jornada declarada.

    Lo que esto permite afirmar: la **forma** de la distribucion a lo largo del dia y
    el orden relativo entre franjas. Lo que **no** permite: afirmar que un producto se
    vende "a las 3 de la tarde". El reporte declara el supuesto; toda figura derivada
    se rotula "hora estimada (proxy ordinal)".

    Modifica las facturas en sitio.
    """
    inicio, fin = jornada
    duracion = (fin - inicio) * 3600  # segundos de jornada

    grupos: dict[tuple[str, str], list[Factura] ] = defaultdict(list)
    for f in facturas:
        grupos[(f.archivo, f.caja)].append(f)

    for (archivo, _caja), grupo in grupos.items():
        dia = fecha_desde_nombre(archivo)
        base = datetime(dia.year, dia.month, dia.day, inicio)
        n = len(grupo)
        for rango, factura in enumerate(grupo):
            # +0.5 centra cada factura en su porcion de jornada, para que ni la
            # primera caiga exactamente en la apertura ni la ultima en el cierre.
            desplazamiento = duracion * (rango + 0.5) / n
            factura.timestamp = base + timedelta(seconds=desplazamiento)
            factura.hora_estimada = True


def cargar_facturas(
    rutas: Sequence[Path] | None = None,
    jornada: tuple[int, int] = JORNADA,
) -> list[Factura]:
    """Lee los DBF y devuelve las facturas reconstruidas, con hora estimada asignada."""
    rutas = list(rutas) if rutas is not None else archivos_dbf()

    facturas: list[Factura] = []
    descartadas_sin_producto = 0

    for ruta in rutas:
        tabla = DBFTable(ruta)
        filas = list(tabla.records())
        catalogo = construir_catalogo(filas)
        dia = fecha_desde_nombre(ruta)
        # Hora provisional: se reemplaza en asignar_hora_estimada. Mantener el orden
        # fisico de lectura es lo que hace valido el proxy.
        provisional = datetime(dia.year, dia.month, dia.day, jornada[0])

        por_clave: dict[tuple[str, str, int], Factura] = {}
        orden: list[Factura] = []

        for fila in filas:
            codigo = (fila.get("CODIGO") or "").strip()
            nombre = catalogo.get(codigo) or normalizar_producto(fila.get("DESCRI") or "")
            if not nombre:
                descartadas_sin_producto += 1
                continue

            caja = str(fila.get("CODCAJ") or "")
            tipo = (fila.get("CODTRA") or "").strip()
            consecutivo = int(fila.get("FACNRO") or 0)
            clave = (caja, tipo, consecutivo)

            factura = por_clave.get(clave)
            if factura is None:
                factura = Factura(
                    numero=f"{ruta.stem}|{caja}|{tipo}|{consecutivo}",
                    timestamp=provisional,
                    consecutivo=consecutivo,
                    caja=caja,
                    tipo_doc=tipo,
                    archivo=ruta.name,
                )
                por_clave[clave] = factura
                orden.append(factura)

            factura.items.append(_fila_a_lineitem(fila, nombre))

        facturas.extend(orden)

    asignar_hora_estimada(facturas, jornada)
    if descartadas_sin_producto:
        print(
            f"[parse] {descartadas_sin_producto} renglones sin producto identificable, "
            "descartados"
        )
    return facturas


# ------------------------------------------------------------------- DataFrames


def a_dataframes(facturas: Sequence[Factura]):
    """Devuelve ``(df_items, df_facturas)`` a nivel renglon y a nivel factura."""
    import pandas as pd

    from src.etl.calendario import nombre_dia

    filas_items = []
    filas_fact = []
    for f in facturas:
        filas_fact.append(
            {
                "factura": f.numero,
                "consecutivo": f.consecutivo,
                "timestamp": f.timestamp,
                "fecha": f.fecha,
                "hora": f.hora,
                "dia_semana": nombre_dia(f.fecha),
                "caja": f.caja,
                "tipo_doc": f.tipo_doc,
                "archivo": f.archivo,
                "n_lineas": f.n_lineas,
                "n_productos": len(f.canasta),
                "total_items": f.total_items,
                "total": f.total,
                "total_impuesto": f.total_impuesto,
                "total_descuento": f.total_descuento,
                "cuadra": f.cuadra,
                "es_unitaria": f.es_unitaria,
                "hora_estimada": f.hora_estimada,
            }
        )
        for item in f.items:
            filas_items.append(
                {
                    "factura": f.numero,
                    "timestamp": f.timestamp,
                    "fecha": f.fecha,
                    "hora": f.hora,
                    "caja": f.caja,
                    "tipo_doc": f.tipo_doc,
                    "producto": item.producto,
                    "codigo": item.codigo,
                    "cantidad": item.cantidad,
                    "precio_unitario": item.precio_unitario,
                    "subtotal": item.subtotal,
                    "iva": item.iva,
                    "descuento": item.descuento,
                    "tasa_iva": item.tasa_iva,
                    "cuadra": item.cuadra,
                }
            )

    df_items = pd.DataFrame(filas_items)
    df_fact = pd.DataFrame(filas_fact)
    return df_items, df_fact


def cargar(usar_cache: bool = True, jornada: tuple[int, int] = JORNADA):
    """Entrada principal del analisis: ``(df_items, df_facturas)``.

    Es una fachada sobre la capa gold del ETL (`src.etl`). Si las tablas gold no
    existen todavia, corre el pipeline. Asi `eda.py`, `rules.py` y `bench.py` dependen
    de un solo punto de entrada y reciben el dia de la semana ya resuelto por
    `dim_calendario`, en lugar de derivarlo cada uno por su cuenta.
    """
    from src.etl import load as etl_load
    from src.etl import paths as etl_paths

    if usar_cache and etl_paths.GOLD_FACT_LINE_ITEMS.exists():
        return etl_load.leer_gold()

    from src.etl import pipeline

    pipeline.correr(forzar=not usar_cache)
    return etl_load.leer_gold()


def canastas_desde_items(df_items, solo_fa: bool = False) -> list[list[str]]:
    """Canastas (listas de productos distintos) desde el DataFrame de renglones."""
    df = df_items[df_items["tipo_doc"] == TIPO_VENTA] if solo_fa else df_items
    agrupado = df.groupby("factura")["producto"].unique()
    return [list(v) for v in agrupado]


# ------------------------------------------------------------------------- CLI


def main() -> None:
    """`python -m src.parse`: lee los DBF con el modelo de objetos y lo verifica.

    Es la comprobacion del punto 2 del enunciado (objetos LineItem y Factura) y una
    segunda implementacion independiente del ETL: si los conteos de aqui y los del
    pipeline no coinciden, uno de los dos esta mal. Las tablas del analisis las
    produce `python -m src.etl.pipeline`, no este modulo.
    """
    rutas = archivos_dbf()
    print(f"[parse] {len(rutas)} archivos DBF en {DATA_DIR}")

    facturas = cargar_facturas(rutas)
    df_items, df_fact = a_dataframes(facturas)

    n_items = len(df_items)
    n_fact = len(df_fact)
    descuadre = (~df_items["cuadra"]).mean() * 100

    print(f"[parse] line-items          : {n_items:,}")
    print(f"[parse] facturas            : {n_fact:,}")
    print(f"[parse] productos distintos : {df_items['producto'].nunique():,}")
    print(f"[parse] lineas por factura  : media {df_fact['n_lineas'].mean():.2f} "
          f"max {df_fact['n_lineas'].max()}")
    print(f"[parse] facturas unitarias  : {df_fact['es_unitaria'].sum():,} "
          f"({df_fact['es_unitaria'].mean() * 100:.1f}%)")
    print(f"[parse] renglones con descuadre > 1 peso: {descuadre:.2f}%")
    print(f"[parse] rango de fechas     : {df_fact['fecha'].min()} .. {df_fact['fecha'].max()}")
    print(f"[parse] tipos de documento  : "
          f"{df_items['tipo_doc'].value_counts().to_dict()}")
    print("[parse] las tablas del analisis se generan con: python -m src.etl.pipeline")


if __name__ == "__main__":
    main()
