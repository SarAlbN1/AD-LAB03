"""Comparativa de dia de la semana (del ingles *day of week*, dow).

El problema que resuelve este modulo: los 13 archivos **no son 13 dias comparables**.
Son un tramo de 7 dias consecutivos (09-15 marzo), una semana con huecos
(01-06 abril, sin el viernes 05) y un domingo aislado (24 marzo, Domingo de Ramos).
Un `groupby('dia_semana').sum()` sobre los 13 mezcla el efecto del dia con el efecto
de la semana y castiga al viernes, que solo tiene una observacion.

El diseno, que `dim_calendario` ya materializa en la columna `panel`:

* **panel 'completa'** (09-15 marzo): cada dia de la semana exactamente una vez. Es el
  panel donde comparar dias es limpio. De aqui sale la comparativa principal.
* **panel 'parcial'** (01-06 abril): sirve de **replicacion**. Si el patron de dias se
  repite en otra semana, el efecto es del dia; si no, era de esa semana.
* **panel 'atipico'** (24 marzo): se analiza aparte, como caso de interes.

La columna comparativa es un **indice**: ``valor_del_dia / promedio_de_los_dias``.
1.00 es el promedio, 1.20 es un 20% arriba. Es adimensional, asi que el mismo numero
se lee igual para pesos, conteos y tamanos de canasta, y cabe en una tabla de dos
columnas del reporte.

Uso: ``python -m src.dow``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from src.etl.calendario import DIAS_ES, PANEL_ATIPICO, PANEL_COMPLETO, PANEL_PARCIAL
from src.etl.load import leer_dim_calendario, leer_gold
from src.parse import OUTPUT_DIR
from src.rules import SOPORTE_ABSOLUTO_MINIMO, minar

FIGS = OUTPUT_DIR / "figs"
TABLAS = OUTPUT_DIR / "tables"

#: Metricas que se comparan entre dias. El indice se calcula sobre cada una.
METRICAS = ("facturas", "valor", "ticket_promedio", "canasta_promedio", "unidades")

ROTULO_HORA = "hora estimada (proxy ordinal, jornada 07:00-22:00)"


def _tabla(df: pd.DataFrame, nombre: str) -> None:
    TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS / nombre, index=False)
    print(f"[dow] tabla  -> outputs/tables/{nombre}  ({len(df)} filas)")


def _guardar(fig, nombre: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / nombre, dpi=200)
    plt.close(fig)
    print(f"[dow] figura -> outputs/figs/{nombre}")


# --------------------------------------------------------------------- agregados


def metricas_por_dia(facturas: pd.DataFrame) -> pd.DataFrame:
    """Nivel por dia de la semana: una fila por dia, sin indices todavia."""
    tabla = (
        facturas.groupby("dia_semana", observed=True)
        .agg(
            fechas=("fecha", "nunique"),
            facturas=("factura", "count"),
            valor=("total", "sum"),
            unidades=("total_items", "sum"),
            ticket_promedio=("total", "mean"),
            canasta_promedio=("n_productos", "mean"),
            pct_unitarias=("es_unitaria", "mean"),
        )
        .reset_index()
    )
    tabla["pct_unitarias"] = (tabla["pct_unitarias"] * 100).round(1)
    # Si un dia tuviera mas de una fecha observada, el volumen se expresa por dia
    # observado; de lo contrario sumariamos semanas distintas.
    for col in ("facturas", "valor", "unidades"):
        tabla[col] = tabla[col] / tabla["fechas"]
    return tabla


def agregar_indices(tabla: pd.DataFrame,
                    metricas: tuple[str, ...] = METRICAS) -> pd.DataFrame:
    """Agrega la columna comparativa: ``indice_<metrica>`` = dia / promedio de dias.

    Por construccion el promedio de cada columna de indices es exactamente 1.00, lo
    que sirve de comprobacion aritmetica (ver `main`).
    """
    salida = tabla.copy()
    for metrica in metricas:
        if metrica not in salida.columns:
            continue
        promedio = salida[metrica].mean()
        salida[f"indice_{metrica}"] = (salida[metrica] / promedio).round(3)
        salida[f"delta_pct_{metrica}"] = ((salida[metrica] / promedio - 1) * 100).round(1)
    salida["rank_facturas"] = salida["facturas"].rank(ascending=False).astype(int)
    return salida


def comparativa_principal(facturas: pd.DataFrame) -> pd.DataFrame:
    """Comparativa sobre el panel completo (09-15 marzo), la limpia."""
    panel = facturas[facturas["panel"] == PANEL_COMPLETO]
    tabla = agregar_indices(metricas_por_dia(panel))
    tabla.insert(1, "es_fin_semana", tabla["dia_semana"].isin(["sabado", "domingo"]))
    tabla.insert(2, "tipo_dia", np.where(tabla["es_fin_semana"], "fin de semana", "habil"))
    return tabla


def test_uniformidad(facturas: pd.DataFrame) -> pd.DataFrame:
    """Chi-cuadrado: el trafico esta realmente desigual entre dias, o es azar.

    Sin este contraste el indice es solo descriptivo. La hipotesis nula es que las
    facturas se reparten por igual entre los siete dias.
    """
    panel = facturas[facturas["panel"] == PANEL_COMPLETO]
    observado = (
        panel.groupby("dia_semana", observed=True)["factura"].count().reindex(DIAS_ES)
    )
    chi2, p = chisquare(observado.values)
    return pd.DataFrame(
        [
            ("panel", PANEL_COMPLETO),
            ("dias", len(observado)),
            ("facturas", int(observado.sum())),
            ("hipotesis nula", "las facturas se reparten por igual entre los 7 dias"),
            ("chi2", round(float(chi2), 1)),
            ("grados de libertad", len(observado) - 1),
            ("p_valor", f"{p:.3e}"),
            ("se rechaza al 1%", bool(p < 0.01)),
            ("lectura", "el dia de la semana si afecta el trafico"
                        if p < 0.01 else "no hay evidencia de efecto del dia"),
        ],
        columns=["clave", "valor"],
    )


def replicacion(facturas: pd.DataFrame) -> pd.DataFrame:
    """El mismo indice en la semana de marzo y en la de abril.

    Solo sobre los dias presentes en **ambos** paneles: la semana de abril no tiene
    viernes ni domingo, y comparar indices calculados sobre conjuntos de dias
    distintos no diria nada.
    """
    completo = facturas[facturas["panel"] == PANEL_COMPLETO]
    parcial = facturas[facturas["panel"] == PANEL_PARCIAL]

    dias_comunes = sorted(
        set(completo["dia_semana"].astype(str)) & set(parcial["dia_semana"].astype(str)),
        key=DIAS_ES.index,
    )

    partes = {}
    for nombre, subconjunto in (("marzo", completo), ("abril", parcial)):
        recorte = subconjunto[subconjunto["dia_semana"].astype(str).isin(dias_comunes)]
        tabla = agregar_indices(metricas_por_dia(recorte))
        partes[nombre] = tabla.set_index(tabla["dia_semana"].astype(str))

    marzo, abril = partes["marzo"], partes["abril"]
    salida = pd.DataFrame({"dia_semana": dias_comunes})
    salida["facturas_marzo"] = [marzo.loc[d, "facturas"] for d in dias_comunes]
    salida["facturas_abril"] = [abril.loc[d, "facturas"] for d in dias_comunes]
    salida["indice_marzo"] = [marzo.loc[d, "indice_facturas"] for d in dias_comunes]
    salida["indice_abril"] = [abril.loc[d, "indice_facturas"] for d in dias_comunes]
    salida["delta_indice"] = (salida["indice_abril"] - salida["indice_marzo"]).round(3)
    # El patron se replica si el dia queda del mismo lado del promedio en las dos
    # semanas: arriba en las dos, o abajo en las dos.
    salida["mismo_lado_del_promedio"] = (
        (salida["indice_marzo"] - 1).apply(np.sign)
        == (salida["indice_abril"] - 1).apply(np.sign)
    )
    salida["caida_volumen_pct"] = (
        (salida["facturas_abril"] / salida["facturas_marzo"] - 1) * 100
    ).round(1)
    return salida


def diagnostico_replicacion(rep: pd.DataFrame) -> pd.DataFrame:
    """Dice si la semana de abril sirve de replicacion, y cuando no, por que.

    No basta con publicar los dos indices lado a lado: hay que decir si son
    comparables. Dos senales de que no lo son:

    * **Correlacion de rangos** entre los indices de las dos semanas. Si es negativa o
      cercana a cero, el orden de los dias no se repite.
    * **Gradiente de la caida de volumen**. Si la caida porcentual crece de forma
      monotona con el dia de la semana, lo que cambia no es la demanda del dia sino la
      completitud de los archivos: huele a exportaciones parciales, no a conducta de
      clientes. Los archivos de abril pesan 0,8 / 1,7 / 2,5 / 3,3 / 4,7 MB de lunes a
      sabado, en el mismo orden que la caida, lo que refuerza esa lectura.
    """
    from scipy.stats import spearmanr

    rho, p = spearmanr(rep["indice_marzo"], rep["indice_abril"])
    caidas = rep["caida_volumen_pct"].tolist()
    monotona = all(b > a for a, b in zip(caidas, caidas[1:]))
    coinciden = int(rep["mismo_lado_del_promedio"].sum())

    usable = bool(rho > 0.5 and not monotona)
    filas = [
        ("dias comparables", len(rep)),
        ("dias con el mismo lado del promedio", f"{coinciden} de {len(rep)}"),
        ("correlacion de rangos (spearman)", round(float(rho), 3)),
        ("p_valor", f"{p:.3f}"),
        ("caida de volumen mediana", f"{rep['caida_volumen_pct'].median():.1f}%"),
        ("caida monotona por dia", monotona),
        ("abril sirve de replicacion", usable),
        (
            "lectura",
            "el patron de dias se replica en la otra semana" if usable else
            "la semana de abril no replica el efecto del dia: su volumen cae entre "
            f"{min(caidas):.0f}% y {max(caidas):.0f}% con un gradiente que sigue el "
            "orden de los dias, lo que apunta a archivos incompletos mas que a "
            "demanda. Sirve para mostrar el cambio de nivel, no para validar el "
            "efecto del dia",
        ),
    ]
    return pd.DataFrame(filas, columns=["clave", "valor"])


# ------------------------------------------------------------- producto por dia


def producto_por_dia(items: pd.DataFrame, facturas: pd.DataFrame,
                     minimo_facturas: int = SOPORTE_ABSOLUTO_MINIMO) -> pd.DataFrame:
    """Sesgo de dia por producto: una fila por producto, un indice por dia.

    ``indice_<dia>`` = soporte del producto ese dia / su soporte global. Un valor de
    1.6 el domingo significa que el producto aparece en un 60% mas de las canastas de
    los domingos que en una canasta promedio.

    ``amplitud`` (pico/valle) ordena la tabla: arriba quedan los productos cuyo consumo
    depende mas del dia, que son los candidatos a promocion por dia y material para la
    pregunta 8 del taller.
    """
    panel_items = items[items["panel"] == PANEL_COMPLETO]
    panel_fact = facturas[facturas["panel"] == PANEL_COMPLETO]

    facturas_por_dia = (
        panel_fact.groupby("dia_semana", observed=True)["factura"].count()
    )
    total_facturas = int(facturas_por_dia.sum())

    presencia = (
        panel_items.groupby(["producto", "dia_semana"], observed=True)["factura"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=DIAS_ES, fill_value=0)
    )
    totales = presencia.sum(axis=1)
    presencia = presencia[totales >= minimo_facturas]
    totales = totales[totales >= minimo_facturas]

    soporte_global = totales / total_facturas
    soporte_dia = presencia.div(facturas_por_dia.reindex(DIAS_ES), axis=1)
    indices = soporte_dia.div(soporte_global, axis=0).round(3)
    indices.columns = [f"indice_{d}" for d in DIAS_ES]

    salida = pd.concat(
        [
            pd.DataFrame(
                {
                    "producto": totales.index,
                    "facturas": totales.values,
                    "soporte_global": soporte_global.round(5).values,
                }
            ).set_index(totales.index),
            indices,
        ],
        axis=1,
    )
    columnas_indice = list(indices.columns)
    salida["dia_pico"] = indices.idxmax(axis=1).str.replace("indice_", "", regex=False)
    salida["indice_pico"] = indices.max(axis=1)
    salida["dia_valle"] = indices.idxmin(axis=1).str.replace("indice_", "", regex=False)
    salida["indice_valle"] = indices.min(axis=1)
    salida["dias_sin_ventas"] = (indices == 0).sum(axis=1)
    # Un producto que no se vendio ningun dia de la semana tiene valle 0 y amplitud
    # infinita. Es el sesgo de dia mas fuerte posible, no un dato faltante, asi que
    # queda como inf y la tabla lo ordena primero.
    with np.errstate(divide="ignore"):
        salida["amplitud"] = (salida["indice_pico"] / salida["indice_valle"]).round(2)
    salida["sesgo_fin_semana"] = (
        salida[["indice_sabado", "indice_domingo"]].mean(axis=1)
        / salida[columnas_indice].drop(columns=["indice_sabado", "indice_domingo"]).mean(axis=1)
    ).round(3)
    return salida.sort_values("amplitud", ascending=False).reset_index(drop=True)


# --------------------------------------------------- reglas por tipo de dia


#: Soporte absoluto exigido a una regla para entrar a la comparacion por segmento.
#: El minimo general (30 facturas) alcanza para no leer ruido, pero al partir el panel
#: en habiles y fin de semana cada segmento se queda con una fraccion de esas
#: facturas, asi que la regla tiene que traer mas masa para seguir siendo medible.
MINIMO_FACTURAS_SEGMENTO = 100


def reglas_por_tipo_dia(items: pd.DataFrame, top: int = 15,
                        minimo_facturas: int = MINIMO_FACTURAS_SEGMENTO) -> pd.DataFrame:
    """Las reglas principales recalculadas en dias habiles y en fin de semana.

    Responde si una regla es estructural (se sostiene en los dos segmentos) o es de
    fin de semana. Entra a la pregunta 7 del taller.

    Tres decisiones en la seleccion de las reglas de referencia, todas para evitar que
    la tabla se llene de ruido:

    1. **Se ordena por `leverage`, no por `lift`.** El lift crudo premia a las reglas
       sostenidas por un punado de facturas: con 30 facturas de 46.554 se alcanzan
       lifts de 1.400 que no describen a ningun cliente. `leverage` es
       ``soporte - soporte_esperado``, asi que pondera por la masa de la regla y deja
       arriba lo que de verdad pasa seguido. Las de lift extremo y soporte minimo son
       justamente las espurias de la pregunta 4, y viven en `23_p4_espurias.csv`.
    2. **Consecuente de un solo producto.** Sin esto la tabla se llena de
       permutaciones del mismo itemset: un itemset de 5 productos genera decenas de
       reglas que dicen lo mismo.
    3. **Se deduplica por itemset.** De las reglas que cubren el mismo conjunto de
       productos se conserva la de mayor `leverage`.
    """
    panel = items[items["panel"] == PANEL_COMPLETO]
    _, reglas_panel, n_panel = minar(panel)

    candidatas = reglas_panel[
        (reglas_panel["facturas"] >= minimo_facturas)
        & (reglas_panel["k_consecuente"] == 1)
    ].copy()
    candidatas["itemset"] = [
        " + ".join(sorted(set(a.split(" + ")) | set(c.split(" + "))))
        for a, c in zip(candidatas["antecedente"], candidatas["consecuente"])
    ]
    candidatas = (
        candidatas.sort_values("leverage", ascending=False)
        .drop_duplicates("itemset", keep="first")
    )

    referencia = (
        candidatas.nlargest(top, "leverage")[
            ["antecedente", "consecuente", "support", "confidence", "lift", "leverage",
             "facturas"]
        ]
        .rename(columns={"support": "support_panel", "confidence": "confidence_panel",
                         "lift": "lift_panel", "leverage": "leverage_panel",
                         "facturas": "facturas_panel"})
        .reset_index(drop=True)
    )

    segmentos = {}
    for etiqueta, filtro in (("habil", "habil"), ("fin_semana", "fin de semana")):
        recorte = panel[panel["tipo_dia"] == filtro]
        _, reglas, _ = minar(recorte)
        segmentos[etiqueta] = reglas.set_index(["antecedente", "consecuente"])

    salida = referencia.copy()
    for etiqueta, reglas in segmentos.items():
        claves = list(zip(salida["antecedente"], salida["consecuente"]))
        for metrica in ("support", "confidence", "lift"):
            salida[f"{metrica}_{etiqueta}"] = [
                round(float(reglas.loc[k, metrica]), 5) if k in reglas.index else np.nan
                for k in claves
            ]

    salida["delta_lift"] = (salida["lift_fin_semana"] - salida["lift_habil"]).round(3)
    # Estable = la regla aparece en los dos segmentos y en ambos supera el azar.
    salida["estable"] = salida[["lift_habil", "lift_fin_semana"]].notna().all(axis=1) & (
        salida[["lift_habil", "lift_fin_semana"]].min(axis=1) > 1.0
    )
    salida["solo_en"] = np.where(
        salida["lift_habil"].isna() & salida["lift_fin_semana"].notna(), "fin de semana",
        np.where(salida["lift_fin_semana"].isna() & salida["lift_habil"].notna(),
                 "habil", ""),
    )
    print(f"[dow] reglas sobre el panel completo: {len(reglas_panel):,} "
          f"({n_panel:,} transacciones); "
          f"{len(candidatas):,} con >= {minimo_facturas} facturas y consecuente k=1")
    return salida


# ----------------------------------------------------------------- dia atipico


def atipico(items: pd.DataFrame, facturas: pd.DataFrame) -> pd.DataFrame:
    """El domingo atipico (24 marzo, Domingo de Ramos) contra el domingo normal."""
    atip = facturas[facturas["panel"] == PANEL_ATIPICO]
    if atip.empty:
        return pd.DataFrame()
    fecha_atipica = atip["fecha"].iloc[0]
    normal = facturas[
        (facturas["dia_semana"].astype(str) == "domingo")
        & (facturas["fecha"] != fecha_atipica)
    ]

    def resumen(fact: pd.DataFrame, it: pd.DataFrame) -> dict:
        return {
            "facturas": len(fact),
            "valor": float(fact["total"].sum()),
            "ticket_promedio": round(float(fact["total"].mean()), 1),
            "canasta_promedio": round(float(fact["n_productos"].mean()), 3),
            "pct_unitarias": round(float(fact["es_unitaria"].mean() * 100), 1),
            "productos_distintos": int(it["producto"].nunique()),
        }

    it_atip = items[items["fecha"] == fecha_atipica]
    it_normal = items[items["factura"].isin(set(normal["factura"]))]

    a = resumen(atip, it_atip)
    n = resumen(normal, it_normal)
    filas = [
        {"metrica": k, f"domingo_{fecha_atipica}": a[k],
         "domingo_normal": n[k],
         "razon": round(a[k] / n[k], 2) if isinstance(n[k], (int, float)) and n[k] else None}
        for k in a
    ]

    exclusivos = set(it_atip["producto"]) - set(it_normal["producto"])
    filas.append({"metrica": "productos solo ese dia",
                  f"domingo_{fecha_atipica}": len(exclusivos),
                  "domingo_normal": 0, "razon": None})
    top_atip = (
        it_atip.groupby("producto")["factura"].nunique().nlargest(5).index.tolist()
    )
    filas.append({"metrica": "top 5 productos",
                  f"domingo_{fecha_atipica}": " | ".join(top_atip),
                  "domingo_normal": " | ".join(
                      it_normal.groupby("producto")["factura"].nunique()
                      .nlargest(5).index.tolist()),
                  "razon": None})
    return pd.DataFrame(filas)


# -------------------------------------------------------------------- figuras


def figura_indice(tabla: pd.DataFrame) -> None:
    """Una sola figura cuenta toda la comparativa: el indice por dia."""
    orden = [d for d in DIAS_ES if d in set(tabla["dia_semana"].astype(str))]
    datos = tabla.set_index(tabla["dia_semana"].astype(str)).loc[orden]

    colores = ["#c44e52" if d in ("sabado", "domingo") else "#4878a8" for d in orden]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    ax1.bar(orden, datos["indice_facturas"], color=colores)
    ax1.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax1.set_ylabel("Indice (1.00 = promedio de la semana)")
    ax1.set_title("Trafico por dia: facturas")
    ax1.tick_params(axis="x", rotation=35)
    for i, v in enumerate(datos["indice_facturas"]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    ax2.bar(orden, datos["indice_ticket_promedio"], color=colores)
    ax2.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax2.set_title("Valor por dia: ticket promedio")
    ax2.tick_params(axis="x", rotation=35)
    for i, v in enumerate(datos["indice_ticket_promedio"]):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    fig.suptitle("Comparativa por dia de la semana (panel 09-15 marzo; rojo = fin de semana)",
                 fontsize=9)
    _guardar(fig, "15_indice_por_dia.png")


def figura_heatmap(facturas: pd.DataFrame) -> None:
    """Dia x hora estimada, midiendo **ticket promedio**, no cantidad de facturas.

    Por que no el conteo: la hora estimada reparte las facturas de cada caja de forma
    uniforme sobre la jornada, asi que el numero de facturas por hora es constante
    **por construccion del proxy**. Un mapa de calor de conteos saldria liso y no
    diria nada del negocio; solo mostraria el efecto del dia, que ya cuenta la
    figura 15.

    El ticket si es informativo: el proxy ordena las facturas por el orden en que se
    facturaron, de modo que el eje horizontal es la **posicion dentro de la jornada**,
    y el valor de la canasta en cada posicion no esta determinado por la construccion.
    Lo que se lee es como cambia el valor de la compra a lo largo del dia, no cuanta
    gente entra a cada hora.

    Se usa la **mediana y no el promedio** porque el conjunto incluye documentos que no
    son ventas al consumidor: hay un documento AS de inventario de cerveza por
    $13,9 millones repetido en cinco dias, y los tipos AS, CO y AE promedian entre
    $340.000 y $470.000 de ticket contra $6.604 de las facturas FA. Con promedio, cada
    uno de esos documentos pintaba de rojo la celda donde cayera y el mapa mostraba
    cinco outliers en vez del comportamiento de la tienda.
    """
    panel = facturas[facturas["panel"] == PANEL_COMPLETO]
    pivote = (
        panel.pivot_table(index="dia_semana", columns="hora", values="total",
                          aggfunc="median", observed=True)
        .reindex(DIAS_ES)
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    im = ax.imshow(pivote.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivote.index)))
    ax.set_yticklabels(pivote.index)
    ax.set_xticks(range(len(pivote.columns)))
    ax.set_xticklabels(pivote.columns)
    ax.set_xlabel(f"Posicion dentro de la jornada — {ROTULO_HORA}")
    ax.set_title("Ticket mediano por dia y posicion de la jornada")
    fig.colorbar(im, ax=ax, label="Ticket mediano ($)")
    fig.text(0.5, 0.01,
             "Se mide valor y no volumen (el conteo por hora es constante por "
             "construccion del proxy), y con mediana para que los\ndocumentos de "
             "inventario de millones de pesos no pinten la celda donde caen.",
             ha="center", fontsize=6.5, style="italic", linespacing=1.4)
    _guardar(fig, "16_heatmap_dia_hora.png")


# ------------------------------------------------------------------------- CLI


def main() -> None:
    items, facturas = leer_gold()
    dim_cal = leer_dim_calendario()
    print(f"[dow] paneles: " + ", ".join(
        f"{k}:{v}" for k, v in dim_cal["panel"].value_counts().items()))

    principal = comparativa_principal(facturas)
    _tabla(principal, "15_comparativa_dia_semana.csv")
    print("\n[dow] comparativa principal (panel 09-15 marzo):")
    print(principal[["dia_semana", "tipo_dia", "facturas", "ticket_promedio",
                     "canasta_promedio", "indice_facturas", "delta_pct_facturas",
                     "rank_facturas"]].round(2).to_string(index=False))

    # Comprobacion aritmetica del indice: el promedio tiene que ser 1.00.
    promedio = principal["indice_facturas"].mean()
    assert abs(promedio - 1.0) < 0.005, f"indice mal construido: promedio {promedio}"
    print(f"\n[dow] control: promedio de indice_facturas = {promedio:.3f} (debe ser 1.000)")

    _tabla(test_uniformidad(facturas), "15b_test_uniformidad_dia.csv")
    print("\n[dow] chi-cuadrado de uniformidad entre dias:")
    print(test_uniformidad(facturas).to_string(index=False))

    rep = replicacion(facturas)
    _tabla(rep, "16_comparativa_dia_replicacion.csv")
    print("\n[dow] replicacion marzo vs abril:")
    print(rep.round(2).to_string(index=False))

    diag = diagnostico_replicacion(rep)
    _tabla(diag, "16b_diagnostico_replicacion.csv")
    print("\n[dow] diagnostico de la replicacion:")
    print(diag.to_string(index=False))

    ppd = producto_por_dia(items, facturas)
    _tabla(ppd, "17_producto_por_dia.csv")
    print("\n[dow] productos con mas sesgo de dia (top 10 por amplitud):")
    print(ppd.head(10)[["producto", "facturas", "dia_pico", "indice_pico",
                        "dia_valle", "indice_valle", "dias_sin_ventas", "amplitud",
                        "sesgo_fin_semana"]].to_string(index=False))

    rtd = reglas_por_tipo_dia(items)
    _tabla(rtd, "18_reglas_por_tipo_dia.csv")
    print("\n[dow] reglas: habil vs fin de semana:")
    print(rtd[["antecedente", "consecuente", "facturas_panel", "lift_panel",
               "lift_habil", "lift_fin_semana", "delta_lift", "estable",
               "solo_en"]].round(2).to_string(index=False))
    print(f"[dow] reglas estables en los dos segmentos: "
          f"{int(rtd['estable'].sum())} de {len(rtd)}")

    atp = atipico(items, facturas)
    if not atp.empty:
        _tabla(atp, "19_atipico_domingo_ramos.csv")
        print("\n[dow] domingo atipico (24 marzo) vs domingo normal:")
        print(atp.to_string(index=False))

    figura_indice(principal)
    figura_heatmap(facturas)
    print("\n[dow] listo")


if __name__ == "__main__":
    main()
