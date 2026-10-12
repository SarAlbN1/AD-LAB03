"""Dimension de calendario: unica fuente de la verdad del dia de la semana.

Por que es un modulo aparte y no una columna derivada al paso: el dia de la semana se
necesita a nivel renglon, a nivel factura y en las tablas comparativas. Derivarlo tres
veces es donde aparecen las inconsistencias (y donde ya habia una: `parse.py` lo
generaba en ingles con `strftime('%A')` y `eda.py` lo traducia despues).

Lo que esta dimension resuelve, y que una columna simple no resolveria: los 13
archivos **no son 13 dias sueltos**, son dos tramos de calendario desbalanceados.

    bloque 1: 2024-03-09 .. 2024-03-15   7 dias consecutivos, los 7 dias de la semana
    bloque 2: 2024-03-24                 un domingo aislado (Domingo de Ramos)
    bloque 3: 2024-04-01 .. 2024-04-06   falta el viernes 05 (DBF perdido) y el domingo 07

Comparar "todos los lunes" contra "todos los viernes" sumando los 13 archivos mezcla
el efecto del dia con el efecto de la semana, y castiga al viernes, que solo tiene una
observacion. Las columnas `panel`, `bloque` y `n_dias_observados` existen para que el
analisis use el tramo balanceado y deje los otros como replicacion y como caso atipico.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
FIN_DE_SEMANA = {"sabado", "domingo"}

#: Semana Santa 2024: Domingo de Ramos 24 de marzo, Domingo de Pascua 31 de marzo.
#: Es contexto de calendario, no algo deducible de los datos, y explica tanto el
#: tamano del archivo del 24 de marzo como la caida de volumen de la semana de abril.
SEMANA_SANTA_2024 = (date(2024, 3, 24), date(2024, 3, 31))

#: Dias de separacion a partir de los cuales un bloque de un solo dia se considera
#: aislado (y por lo tanto no comparable dentro de una semana).
DIAS_AISLAMIENTO = 7

PANEL_COMPLETO = "completa"
PANEL_PARCIAL = "parcial"
PANEL_ATIPICO = "atipico"


def nombre_dia(d: date) -> str:
    """Dia de la semana en espanol, sin depender del locale del contenedor."""
    return DIAS_ES[d.weekday()]


def _bloques(fechas: Sequence[date]) -> dict[date, int]:
    """Asigna a cada fecha el id del tramo de dias consecutivos al que pertenece."""
    asignacion: dict[date, int] = {}
    bloque = 0
    anterior: date | None = None
    for f in fechas:
        if anterior is not None and (f - anterior).days > 1:
            bloque += 1
        asignacion[f] = bloque
        anterior = f
    return asignacion


def _separaciones(fechas: Sequence[date]) -> dict[date, tuple[int, int]]:
    """Dias hasta la fecha observada anterior y la siguiente (inf si no hay)."""
    grande = 10**6
    salida: dict[date, tuple[int, int]] = {}
    for i, f in enumerate(fechas):
        antes = (f - fechas[i - 1]).days if i > 0 else grande
        despues = (fechas[i + 1] - f).days if i < len(fechas) - 1 else grande
        salida[f] = (antes, despues)
    return salida


def _nota(d: date) -> str:
    inicio, fin = SEMANA_SANTA_2024
    if d == inicio:
        return "Domingo de Ramos (inicio de Semana Santa)"
    if inicio < d <= fin:
        return "Semana Santa"
    if fin < d <= fin + timedelta(days=7):
        return "semana siguiente a Semana Santa"
    return ""


def construir(fechas_archivos: Iterable[tuple[date, str]]) -> pd.DataFrame:
    """Construye ``dim_calendario`` a partir de pares ``(fecha, nombre_de_archivo)``.

    Columnas:

    ``fecha``                clave de la dimension
    ``archivo``              linaje: de que DBF salio el dia
    ``dia_semana``           categorica **ordenada** lunes->domingo, en espanol
    ``dia_num``              0=lunes .. 6=domingo, para ordenar graficas
    ``es_fin_semana``        sabado o domingo
    ``tipo_dia``             'habil' | 'fin de semana'
    ``semana_iso``           p.ej. 2024-W11, separa efecto semana de efecto dia
    ``bloque``               id del tramo de dias consecutivos
    ``bloque_n_dias``        dias observados en ese tramo
    ``panel``                'completa' | 'parcial' | 'atipico' (ver docstring)
    ``es_atipico``           dia aislado del resto del calendario
    ``nota``                 contexto (Semana Santa)
    ``n_dias_observados``    cuantas fechas comparten ese dia de la semana; es el
                             denominador de cualquier promedio por dia
    """
    pares = sorted(set(fechas_archivos))
    if not pares:
        raise ValueError("dim_calendario sin fechas")
    fechas = [f for f, _ in pares]
    if len(set(fechas)) != len(fechas):
        raise ValueError("hay dos archivos para la misma fecha; revisar archivos/")

    bloque_de = _bloques(fechas)
    separacion_de = _separaciones(fechas)
    dias_por_bloque = pd.Series(list(bloque_de.values())).value_counts().to_dict()

    filas = []
    for f, archivo in pares:
        bloque = bloque_de[f]
        n_bloque = dias_por_bloque[bloque]
        antes, despues = separacion_de[f]
        aislado = n_bloque == 1 and antes >= DIAS_AISLAMIENTO and despues >= DIAS_AISLAMIENTO

        # Un bloque es panel completo si trae 7 dias consecutivos, es decir los 7 dias
        # de la semana exactamente una vez: el unico tramo donde comparar dias es limpio.
        if n_bloque == 7:
            panel = PANEL_COMPLETO
        elif aislado:
            panel = PANEL_ATIPICO
        else:
            panel = PANEL_PARCIAL

        iso = f.isocalendar()
        filas.append(
            {
                "fecha": f,
                "archivo": archivo,
                "dia_semana": nombre_dia(f),
                "dia_num": f.weekday(),
                "es_fin_semana": nombre_dia(f) in FIN_DE_SEMANA,
                "tipo_dia": "fin de semana" if nombre_dia(f) in FIN_DE_SEMANA else "habil",
                "semana_iso": f"{iso.year}-W{iso.week:02d}",
                "bloque": bloque,
                "bloque_n_dias": n_bloque,
                "panel": panel,
                "es_atipico": aislado,
                "nota": _nota(f),
            }
        )

    dim = pd.DataFrame(filas)
    dim["dia_semana"] = pd.Categorical(dim["dia_semana"], categories=DIAS_ES, ordered=True)
    # Denominador de los promedios por dia: cuantas veces se observo ese dia.
    dim["n_dias_observados"] = dim.groupby("dia_semana", observed=True)["fecha"].transform(
        "count"
    )
    return dim.sort_values("fecha").reset_index(drop=True)


def desde_archivos(rutas: Iterable[Path]) -> pd.DataFrame:
    """Atajo: construye la dimension desde las rutas de los .DBF."""
    from src.parse import fecha_desde_nombre

    return construir((fecha_desde_nombre(r), r.name) for r in rutas)


def resumen(dim: pd.DataFrame) -> pd.DataFrame:
    """Cobertura del calendario, para el control de calidad y el reporte."""
    por_dia = (
        dim.groupby("dia_semana", observed=True)
        .agg(dias_observados=("fecha", "count"),
             fechas=("fecha", lambda s: ", ".join(str(x) for x in sorted(s))))
        .reset_index()
    )
    return por_dia
