"""Contratos de datos del ETL.

Dos severidades:

* **falla**: el dato esta mal leido o el modelo quedo roto. Levanta `ErrorDeContrato`
  y detiene el pipeline, porque seguir produciria un analisis invalido.
* **avisa**: el dato tiene imperfecciones reales del negocio (descuadres de redondeo,
  codigos con dos descripciones, el viernes que falta). No detiene nada: se cuantifica
  y se publica en ``outputs/tables/00_calidad_datos.csv``, que es material del reporte.

La distincion importa: un descuadre de $1 por redondeo no es un error del laboratorio,
pero una fecha que no corresponde al dia de su carpeta si lo seria.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.etl import paths
from src.etl.calendario import DIAS_ES


class ErrorDeContrato(AssertionError):
    """Un contrato de severidad 'falla' no se cumplio."""


@dataclass
class Reporte:
    """Acumula los resultados de los controles."""

    filas: list[dict] = field(default_factory=list)

    def registrar(self, control: str, severidad: str, ok: bool, valor, detalle: str = "") -> None:
        self.filas.append(
            {
                "control": control,
                "severidad": severidad,
                "resultado": "ok" if ok else ("FALLA" if severidad == "falla" else "aviso"),
                "valor": valor,
                "detalle": detalle,
            }
        )
        if not ok and severidad == "falla":
            raise ErrorDeContrato(f"{control}: {valor} {detalle}".strip())

    def exigir(self, control: str, ok: bool, valor, detalle: str = "") -> None:
        self.registrar(control, "falla", ok, valor, detalle)

    def avisar(self, control: str, ok: bool, valor, detalle: str = "") -> None:
        self.registrar(control, "aviso", ok, valor, detalle)

    def a_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)

    def guardar(self) -> None:
        paths.TABLAS.mkdir(parents=True, exist_ok=True)
        self.a_dataframe().to_csv(paths.CALIDAD, index=False)
        print(f"[validate] calidad -> {paths.CALIDAD.relative_to(paths.RAIZ)}")


# ------------------------------------------------------------------------ bronze


def validar_bronze(manifiesto: pd.DataFrame, bronze: pd.DataFrame,
                   rep: Reporte) -> None:
    """La extraccion no perdio ni invento registros, y el Varchar quedo bien leido."""
    esperados = int(manifiesto["registros_header"].sum())
    obtenidos = len(bronze)
    # El header del DBF cuenta tambien los registros marcados como borrados; el bronze
    # solo trae los vivos. Por eso <=, no ==.
    rep.exigir("bronze: registros <= los del header DBF", obtenidos <= esperados,
               f"{obtenidos:,} de {esperados:,}")
    rep.avisar("bronze: registros borrados omitidos", True, esperados - obtenidos)

    rep.exigir("bronze: un sha256 por archivo",
               manifiesto["sha256"].nunique() == len(manifiesto),
               manifiesto["sha256"].nunique())

    # Si el Varchar de Visual FoxPro no se recorto, su byte de longitud aparece como
    # caracter de control dentro de la descripcion.
    control = bronze["DESCRI"].astype(str).str.contains(r"[\x00-\x1f]", regex=True)
    rep.exigir("bronze: DESCRI sin caracteres de control", not bool(control.any()),
               int(control.sum()), "el Varchar VFP quedo mal recortado")

    rep.exigir("bronze: columna de orden fisico presente", "_fila" in bronze.columns, "_fila")

    for carpeta, grupo in manifiesto.groupby("carpeta"):
        if carpeta in DIAS_ES:
            dias = {DIAS_ES[f.weekday()] for f in grupo["fecha"]}
            rep.exigir(f"calendario: {carpeta}/ contiene solo {carpeta}",
                       dias == {carpeta}, ", ".join(sorted(dias)),
                       "la fecha del nombre se leyo como ddmmyyyy")


def columnas_constantes(bronze: pd.DataFrame, rep: Reporte) -> list[str]:
    """Campos del DBF sin informacion (un solo valor en los 13 archivos)."""
    ignorar = {"_archivo", "_sha256", "_fila"}
    constantes = [
        c for c in bronze.columns
        if c not in ignorar and bronze[c].nunique(dropna=False) <= 1
    ]
    rep.avisar("bronze: campos sin informacion", True, len(constantes),
               ", ".join(constantes))
    return constantes


# ------------------------------------------------------------------------ silver


def validar_silver(items: pd.DataFrame, facturas: pd.DataFrame, rep: Reporte) -> None:
    """Integridad del modelo de facturas y validez de la hora estimada."""
    rep.exigir("silver: sin renglones huerfanos",
               set(items["factura"]) <= set(facturas["factura"]),
               len(set(items["factura"]) - set(facturas["factura"])))

    rep.exigir("silver: n_lineas coincide con los renglones",
               bool((facturas.set_index("factura")["n_lineas"]
                     .eq(items.groupby("factura").size())).all()),
               "consistente")

    rep.exigir("silver: cantidades no negativas",
               bool((items["cantidad"] >= 0).all()),
               int((items["cantidad"] < 0).sum()))

    rep.exigir("silver: todo renglon tiene producto",
               bool(items["producto"].str.len().gt(0).all()),
               int(items["producto"].str.len().eq(0).sum()))

    # Hora estimada: tiene que caer en la jornada y respetar el orden de facturacion.
    rep.exigir("silver: hora estimada dentro de la jornada",
               bool(facturas["hora"].between(7, 22).all()),
               f"{facturas['hora'].min()}..{facturas['hora'].max()}")

    desorden = 0
    for _, grupo in facturas.groupby(["archivo", "caja"], observed=True):
        orden = grupo.sort_values("orden_fisico")["timestamp"].tolist()
        if orden != sorted(orden):
            desorden += 1
    rep.exigir("silver: hora estimada monotona por caja", desorden == 0, desorden)

    # Avisos: imperfecciones reales del negocio.
    descuadre = (~items["cuadra"]).mean() * 100
    rep.avisar("silver: renglones con descuadre > $1", descuadre < 5,
               f"{descuadre:.2f}%", "cantidad x precio - descuento vs subtotal")
    rep.avisar("silver: facturas con total <= 0", True,
               int((facturas["total"] <= 0).sum()))
    rep.avisar("silver: facturas de un solo producto", True,
               f"{facturas['es_unitaria'].sum():,} "
               f"({facturas['es_unitaria'].mean() * 100:.1f}%)",
               "no aportan reglas de asociacion")
    rep.avisar("silver: tipos de documento", True,
               ", ".join(f"{k}:{v:,}" for k, v in items["tipo_doc"].value_counts().items()))


# -------------------------------------------------------------------------- gold


def validar_gold(dim_cal: pd.DataFrame, items: pd.DataFrame, facturas: pd.DataFrame,
                 rep: Reporte) -> None:
    """La dimension de calendario llega igual a los dos niveles de hechos."""
    rep.exigir("gold: dim_calendario sin fechas repetidas",
               dim_cal["fecha"].is_unique, len(dim_cal))

    rep.exigir("gold: toda fecha de hechos esta en el calendario",
               set(facturas["fecha"]) <= set(dim_cal["fecha"]),
               len(set(facturas["fecha"]) - set(dim_cal["fecha"])))

    # La misma factura tiene que tener el mismo dia de la semana en los dos niveles:
    # es la prueba de que la dimension se derivo una sola vez.
    cruce = items[["factura", "dia_semana"]].drop_duplicates().merge(
        facturas[["factura", "dia_semana"]], on="factura", suffixes=("_item", "_fact")
    )
    discrepan = int((cruce["dia_semana_item"].astype(str)
                     != cruce["dia_semana_fact"].astype(str)).sum())
    rep.exigir("gold: dia_semana consistente entre renglon y factura",
               discrepan == 0, discrepan)

    rep.exigir("gold: dia_semana sin nulos",
               not bool(facturas["dia_semana"].isna().any()),
               int(facturas["dia_semana"].isna().sum()))

    # Cobertura del calendario: aqui queda registrado el viernes que falta.
    cobertura = dim_cal.groupby("dia_semana", observed=True)["fecha"].count()
    faltantes = cobertura[cobertura < cobertura.max()]
    rep.avisar("gold: cobertura por dia de la semana", faltantes.empty,
               ", ".join(f"{d}:{n}" for d, n in cobertura.items()),
               "dias con menos observaciones: "
               + ", ".join(str(d) for d in faltantes.index) if len(faltantes) else "")

    paneles = dim_cal["panel"].value_counts().to_dict()
    rep.avisar("gold: paneles de calendario", "completa" in paneles,
               ", ".join(f"{k}:{v}" for k, v in paneles.items()),
               "se necesita un panel 'completa' para comparar dias de la semana")
