"""Analisis descriptivo del conjunto de datos (punto 3 del enunciado).

Genera las tablas en ``outputs/tables/*.csv`` y las figuras en ``outputs/figs/*.png``
que cita el reporte. Toda salida con eje horario lleva el rotulo
"hora estimada (proxy ordinal)": la hora no esta en los datos, se estima (ver
`src.parse.asignar_hora_estimada`).

Uso: ``python -m src.eda``
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.parse import OUTPUT_DIR, TIPO_VENTA, cargar

FIGS = OUTPUT_DIR / "figs"
TABLES = OUTPUT_DIR / "tables"

ROTULO_HORA = "hora estimada (proxy ordinal, jornada 07:00-22:00)"


def _preparar_dirs() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def _guardar(fig, nombre: str) -> Path:
    destino = FIGS / nombre
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)
    print(f"[eda] figura -> {destino.relative_to(OUTPUT_DIR.parent)}")
    return destino


def _tabla(df: pd.DataFrame, nombre: str) -> Path:
    destino = TABLES / nombre
    df.to_csv(destino, index=False)
    print(f"[eda] tabla  -> {destino.relative_to(OUTPUT_DIR.parent)}")
    return destino


# --------------------------------------------------------------- 3.a generalidades


def resumen_general(df_items: pd.DataFrame, df_fact: pd.DataFrame) -> pd.DataFrame:
    """Generalidades del conjunto: volumen, canasta, ticket, calidad de los datos."""
    n_lineas = df_fact["n_lineas"]
    filas = [
        ("Registros (line-items)", f"{len(df_items):,}"),
        ("Facturas", f"{len(df_fact):,}"),
        ("Archivos / dias cubiertos", f"{df_fact['archivo'].nunique()} / "
                                      f"{df_fact['fecha'].nunique()}"),
        ("Rango de fechas", f"{df_fact['fecha'].min()} a {df_fact['fecha'].max()}"),
        ("Productos distintos", f"{df_items['producto'].nunique():,}"),
        ("Renglones por factura (media)", f"{n_lineas.mean():.2f}"),
        ("Renglones por factura (mediana)", f"{n_lineas.median():.0f}"),
        ("Renglones por factura (maximo)", f"{n_lineas.max()}"),
        ("Unidades por factura (media)", f"{df_fact['total_items'].mean():.2f}"),
        ("Facturas de un solo producto", f"{df_fact['es_unitaria'].sum():,} "
                                         f"({df_fact['es_unitaria'].mean()*100:.1f}%)"),
        ("Ticket promedio", f"${df_fact['total'].mean():,.0f}"),
        ("Ticket mediano", f"${df_fact['total'].median():,.0f}"),
        ("Venta total", f"${df_fact['total'].sum():,.0f}"),
        ("Impuesto total", f"${df_fact['total_impuesto'].sum():,.0f}"),
        ("Cajas", f"{df_items['caja'].nunique()}"),
        ("Tipos de documento", ", ".join(
            f"{k}:{v:,}" for k, v in df_items["tipo_doc"].value_counts().items())),
        ("Renglones con descuadre > $1", f"{(~df_items['cuadra']).sum():,} "
                                         f"({(~df_items['cuadra']).mean()*100:.2f}%)"),
    ]
    tabla = pd.DataFrame(filas, columns=["metrica", "valor"])
    _tabla(tabla, "01_resumen_general.csv")
    return tabla


def distribucion_por_hora(df_fact: pd.DataFrame) -> pd.DataFrame:
    """Compras a lo largo de la jornada (3.a), con una advertencia que hay que leer.

    **El conteo de facturas por hora es constante por construccion.** Los DBF no traen
    hora; la hora estimada reparte las facturas de cada caja de forma uniforme sobre la
    jornada, asi que "cuantas facturas por hora" solo devuelve el supuesto que se
    metio, no un hecho de la tienda. La columna `facturas` se publica para dejarlo a la
    vista, no como hallazgo.

    Lo que si se puede leer: el **valor** de la compra segun la posicion dentro de la
    jornada. El proxy conserva el orden real de facturacion, de modo que el eje es la
    posicion en la jornada, y el ticket promedio en cada posicion no esta determinado
    por la construccion.

    Con la hora real (si el docente facilita el maestro de facturas) esta misma funcion
    produce la distribucion verdadera sin cambiarle una linea: la hora es una columna.
    """
    por_hora = (
        df_fact.groupby("hora")
        .agg(
            facturas=("factura", "count"),
            unidades=("total_items", "sum"),
            valor=("total", "sum"),
            ticket_promedio=("total", "mean"),
            canasta_promedio=("n_productos", "mean"),
        )
        .reset_index()
    )
    por_hora["nota"] = "facturas por hora: constante por construccion del proxy"
    _tabla(por_hora, "02_distribucion_por_hora.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(por_hora["hora"], por_hora["ticket_promedio"] / 1000, color="#c44e52",
             marker="o", linewidth=1.6)
    ax1.set_xlabel(f"Posicion en la jornada — {ROTULO_HORA}")
    ax1.set_ylabel("Ticket promedio (miles $)")
    ax1.set_title("Valor de la compra a lo largo del dia")
    ax1.grid(alpha=0.25)

    ax2.bar(por_hora["hora"], por_hora["facturas"], color="#b0b0b0")
    ax2.set_xlabel(f"Posicion en la jornada — {ROTULO_HORA}")
    ax2.set_ylabel("Facturas")
    ax2.set_title("Facturas por hora: plano por construccion")
    ax2.text(0.5, 0.5, "no es un hallazgo:\nlo impone el proxy",
             transform=ax2.transAxes, ha="center", va="center", fontsize=8,
             style="italic", color="#555555")

    _guardar(fig, "02_distribucion_por_hora.png")
    return por_hora


def distribucion_canasta(df_fact: pd.DataFrame) -> pd.DataFrame:
    """Histograma del tamano de canasta: explica por que los soportes son bajos."""
    conteo = (
        df_fact["n_productos"].value_counts().sort_index().rename("facturas").reset_index()
    )
    conteo.columns = ["productos_en_factura", "facturas"]
    conteo["porcentaje"] = conteo["facturas"] / conteo["facturas"].sum() * 100
    _tabla(conteo, "03_tamano_canasta.csv")

    recorte = conteo[conteo["productos_en_factura"] <= 12]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar(recorte["productos_en_factura"], recorte["facturas"], color="#55a868")
    ax.set_xlabel("Productos distintos en la factura")
    ax.set_ylabel("Facturas")
    ax.set_title("Tamano de canasta (truncado en 12; maximo real: "
                 f"{conteo['productos_en_factura'].max()})")
    _guardar(fig, "03_tamano_canasta.png")
    return conteo


def por_dia_semana(df_fact: pd.DataFrame) -> pd.DataFrame:
    """Comportamiento por dia de la semana.

    El dia viene ya resuelto de `dim_calendario` (categorica ordenada, en espanol), no
    se traduce aqui. Las metricas de volumen se expresan **por dia observado**: el
    viernes solo tiene una fecha en el conjunto y el domingo tres, asi que sumar
    acumulado lo castigaria. La comparacion fina entre dias esta en `src.dow`.
    """
    tabla = (
        df_fact.groupby("dia_semana", observed=True)
        .agg(
            dias_observados=("fecha", "nunique"),
            facturas=("factura", "count"),
            valor=("total", "sum"),
            ticket_promedio=("total", "mean"),
            canasta_promedio=("n_productos", "mean"),
        )
        .reset_index()
    )
    tabla["facturas_por_dia"] = tabla["facturas"] / tabla["dias_observados"]
    tabla["valor_por_dia"] = tabla["valor"] / tabla["dias_observados"]
    _tabla(tabla, "04_por_dia_semana.csv")
    return tabla


# ------------------------------------------------------- 3.b / 3.c top de productos


def top_productos(df_items: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Productos mas vendidos por numero de facturas y por unidades (3.b y 3.c)."""
    total_facturas = df_items["factura"].nunique()
    agrupado = (
        df_items.groupby("producto")
        .agg(
            facturas=("factura", "nunique"),
            unidades=("cantidad", "sum"),
            valor=("subtotal", "sum"),
        )
        .sort_values("facturas", ascending=False)
    )
    agrupado["soporte"] = agrupado["facturas"] / total_facturas
    agrupado = agrupado.reset_index()
    _tabla(agrupado, "05_productos_ranking_completo.csv")

    top = agrupado.head(n).copy()
    top["pct_del_lider"] = top["facturas"] / top["facturas"].iloc[0] * 100
    top["acumulado_pct"] = top["facturas"].cumsum() / agrupado["facturas"].sum() * 100
    _tabla(top, "06_top10.csv")

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    pos = range(len(top))
    ax1.bar(pos, top["facturas"], color="#4878a8")
    ax1.set_xticks(list(pos))
    ax1.set_xticklabels(
        [p[:18] for p in top["producto"]], rotation=35, ha="right", fontsize=7
    )
    ax1.set_ylabel("Facturas en que aparece")

    ax2 = ax1.twinx()
    ax2.plot(list(pos), top["acumulado_pct"], color="#c44e52", marker="o", linewidth=1.6)
    ax2.set_ylabel("% acumulado sobre el total de apariciones", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")

    ax1.set_title(f"TOP {n} de productos y concentracion acumulada (Pareto)")
    _guardar(fig, "06_top10_pareto.png")
    return top


def producto_lider_por_hora(df_items: pd.DataFrame) -> pd.DataFrame:
    """Producto mas vendido en cada hora estimada (3.b)."""
    conteo = (
        df_items.groupby(["hora", "producto"])
        .agg(facturas=("factura", "nunique"), unidades=("cantidad", "sum"))
        .reset_index()
    )
    idx = conteo.groupby("hora")["facturas"].idxmax()
    lider = conteo.loc[idx].sort_values("hora").reset_index(drop=True)
    lider["nota"] = ROTULO_HORA
    _tabla(lider, "07_lider_por_hora.csv")
    return lider


# ----------------------------------------------------- 3.d producto menos vendido


def productos_menos_vendidos(
    df_items: pd.DataFrame, df_fact: pd.DataFrame, n: int = 15
) -> pd.DataFrame:
    """Cola larga: en que hora aparecio, si se vendio solo y con que se acompano (3.d).

    `acompanantes` lista los productos que estuvieron en la misma factura. Es la
    pregunta que anticipa las reglas de asociacion: un producto que solo se vende
    solo no puede participar de ningun combo.
    """
    tamano = df_fact.set_index("factura")["n_productos"]
    por_producto = (
        df_items.groupby("producto")
        .agg(facturas=("factura", "nunique"), unidades=("cantidad", "sum"))
        .sort_values(["facturas", "unidades"])
    )
    raros = por_producto.head(n).index.tolist()

    detalle = []
    for producto in raros:
        filas = df_items[df_items["producto"] == producto]
        facturas = filas["factura"].unique()
        tamanos = tamano.reindex(facturas)
        acompanantes: Counter = Counter()
        for fac in facturas:
            otros = df_items[(df_items["factura"] == fac)
                             & (df_items["producto"] != producto)]["producto"].unique()
            acompanantes.update(otros)
        detalle.append(
            {
                "producto": producto,
                "facturas": len(facturas),
                "unidades": float(filas["cantidad"].sum()),
                "precio_promedio": float(filas["precio_unitario"].mean()),
                "horas_aparicion": ", ".join(str(h) for h in sorted(filas["hora"].unique())),
                "veces_vendido_solo": int((tamanos == 1).sum()),
                "veces_acompanado": int((tamanos > 1).sum()),
                "acompanantes_top": ", ".join(p for p, _ in acompanantes.most_common(3)),
                "nota_hora": ROTULO_HORA,
            }
        )
    tabla = pd.DataFrame(detalle)
    _tabla(tabla, "08_menos_vendidos.csv")
    return tabla


# ------------------------------------- anexo: tipos de documento y sensibilidad


def documentos_por_tipo(df_items: pd.DataFrame, df_fact: pd.DataFrame) -> pd.DataFrame:
    """Perfil de cada tipo de documento (`CODTRA`).

    El analisis usa **todos** los tipos, no solo las facturas de venta. Esta tabla
    existe para que esa decision quede cuantificada en vez de implicita: los tipos que
    no son FA son el 1,9% de los documentos pero mueven tickets de dos ordenes de
    magnitud mas, asi que pesan sobre cualquier metrica de valor.
    """
    tabla = (
        df_fact.groupby("tipo_doc")
        .agg(
            documentos=("factura", "count"),
            ticket_promedio=("total", "mean"),
            ticket_mediano=("total", "median"),
            ticket_maximo=("total", "max"),
            lineas_promedio=("n_lineas", "mean"),
            valor_total=("total", "sum"),
        )
        .reset_index()
        .sort_values("documentos", ascending=False)
    )
    tabla["pct_documentos"] = (tabla["documentos"] / len(df_fact) * 100).round(2)
    tabla["pct_valor"] = (tabla["valor_total"] / df_fact["total"].sum() * 100).round(2)
    tabla["es_venta"] = tabla["tipo_doc"] == TIPO_VENTA
    tabla = tabla.round(1)
    _tabla(tabla, "00b_documentos_por_tipo.csv")
    return tabla


def documentos_repetidos(df_items: pd.DataFrame, df_fact: pd.DataFrame,
                         minimo_lineas: int = 5) -> pd.DataFrame:
    """Documentos que reaparecen identicos en varias fechas.

    Mismo tipo, mismo numero de renglones y mismo total al peso, en dias distintos. En
    una venta al consumidor eso seria una casualidad extraordinaria; en un documento de
    inventario es lo esperable. El caso mas claro del conjunto es un `AS` de 26
    renglones de cerveza por $13.954.166 que aparece identico en cinco fechas.
    """
    firma = (
        df_fact[df_fact["n_lineas"] >= minimo_lineas]
        .assign(total_redondeado=lambda d: d["total"].round(0))
        .groupby(["tipo_doc", "consecutivo", "n_lineas", "total_redondeado"])
        .agg(fechas=("fecha", "nunique"), documentos=("factura", "count"),
             archivos=("archivo", lambda s: ", ".join(sorted(set(s)))))
        .reset_index()
    )
    repetidos = firma[firma["fechas"] > 1].sort_values(
        ["total_redondeado", "fechas"], ascending=False
    )
    _tabla(repetidos, "00c_documentos_repetidos.csv")
    return repetidos


def sensibilidad_tipo_documento(df_fact: pd.DataFrame) -> pd.DataFrame:
    """Cuanto cambian las metricas de valor al quedarse solo con las ventas (FA).

    Es el anexo de sensibilidad de la decision de alcance: deja ver de un vistazo que
    parte de cada cifra viene de documentos que no son ventas al consumidor.
    """
    ventas = df_fact[df_fact["tipo_doc"] == TIPO_VENTA]

    def metricas(df: pd.DataFrame) -> dict:
        return {
            "documentos": len(df),
            "ticket promedio": df["total"].mean(),
            "ticket mediano": df["total"].median(),
            "valor total": df["total"].sum(),
            "impuesto total": df["total_impuesto"].sum(),
            "canasta promedio": df["n_productos"].mean(),
        }

    todos, solo_fa = metricas(df_fact), metricas(ventas)
    filas = []
    for clave in todos:
        a, b = todos[clave], solo_fa[clave]
        filas.append(
            {
                "metrica": clave,
                "todos_los_codtra": round(a, 2),
                "solo_fa": round(b, 2),
                "diferencia_pct": round((a / b - 1) * 100, 1) if b else None,
            }
        )
    tabla = pd.DataFrame(filas)
    _tabla(tabla, "00d_sensibilidad_tipo_documento.csv")
    return tabla


def integridad(df_items: pd.DataFrame, df_fact: pd.DataFrame) -> pd.DataFrame:
    """Control de calidad: descuadre entre subtotales y cantidad*precio (extra)."""
    desc = df_items.assign(
        descuadre=(df_items["cantidad"] * df_items["precio_unitario"]
                   - df_items["descuento"] - df_items["subtotal"]).abs()
    )
    tabla = pd.DataFrame(
        [
            ("renglones", len(desc)),
            ("renglones con descuadre > $1", int((~df_items["cuadra"]).sum())),
            ("descuadre promedio ($)", round(float(desc["descuadre"].mean()), 4)),
            ("descuadre maximo ($)", round(float(desc["descuadre"].max()), 2)),
            ("facturas con algun descuadre", int((~df_fact["cuadra"]).sum())),
        ],
        columns=["metrica", "valor"],
    )
    _tabla(tabla, "09_integridad.csv")
    return tabla


# ------------------------------------------------------------------------- CLI


def main() -> None:
    _preparar_dirs()
    df_items, df_fact = cargar()
    print(f"[eda] {len(df_items):,} renglones, {len(df_fact):,} facturas")

    resumen = resumen_general(df_items, df_fact)
    print(resumen.to_string(index=False))

    distribucion_por_hora(df_fact)
    distribucion_canasta(df_fact)
    por_dia_semana(df_fact)
    top = top_productos(df_items)
    print("\n[eda] TOP 10:")
    print(top[["producto", "facturas", "unidades", "soporte", "pct_del_lider"]]
          .to_string(index=False))
    producto_lider_por_hora(df_items)
    productos_menos_vendidos(df_items, df_fact)
    integridad(df_items, df_fact)

    tipos = documentos_por_tipo(df_items, df_fact)
    print("\n[eda] documentos por tipo (el analisis usa todos):")
    print(tipos[["tipo_doc", "documentos", "pct_documentos", "ticket_promedio",
                 "ticket_maximo", "pct_valor", "es_venta"]].to_string(index=False))

    repetidos = documentos_repetidos(df_items, df_fact)
    if len(repetidos):
        print("\n[eda] documentos identicos en varias fechas (top 5 por valor):")
        print(repetidos.head(5)[["tipo_doc", "consecutivo", "n_lineas",
                                 "total_redondeado", "fechas"]].to_string(index=False))

    sens = sensibilidad_tipo_documento(df_fact)
    print("\n[eda] sensibilidad: todos los CODTRA vs solo ventas (FA):")
    print(sens.to_string(index=False))
    print("\n[eda] listo")


if __name__ == "__main__":
    main()
