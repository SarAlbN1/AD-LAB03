"""Mineria de reglas de asociacion y respuestas a las 8 preguntas del taller.

Cada funcion publica produce el artefacto concreto que responde una pregunta; `main`
los escribe todos en ``outputs/tables/``.

Uso: ``python -m src.rules``
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from scipy.stats import chi2_contingency, fisher_exact

from src.parse import OUTPUT_DIR, canastas_desde_items, cargar

TABLES = OUTPUT_DIR / "tables"

#: Parametros del analisis de reglas (distintos del benchmark: aqui interesa el
#: resultado, no el tiempo). Con canasta media de 2.14 productos el soporte util es
#: muy bajo, por eso 5e-4 ~ 34 facturas de 68.809.
SOPORTE_ANALISIS = 0.0005
CONFIANZA_MINIMA = 0.05

#: Una regla sostenida por 3 facturas es ruido aunque su lift sea 400. El soporte
#: absoluto minimo es el filtro que separa asociacion de coincidencia.
SOPORTE_ABSOLUTO_MINIMO = 30
LIFT_INTERESANTE = 1.5

#: Soporte al que se va a buscar reglas espurias (pregunta 4). Tiene que ser mas bajo
#: que `SOPORTE_ANALISIS`: con 0.0005 toda regla ya trae 34 facturas o mas, asi que a
#: ese nivel el conjunto no puede producir una regla de soporte minusculo.
#:
#: 0.0002 son 14 facturas de 68.807, suficiente para que aparezcan reglas de soporte
#: minusculo y lift en los miles. Bajar a 0.0001 produciria mas, pero el espacio de
#: itemsets explota (a 0.0002 ya son 5.070 itemsets y 58.125 reglas) y la pregunta se
#: responde igual: lo que crea las espurias es bajar el umbral, no un valor magico.
SOPORTE_ESPURIAS = 0.0002

#: Articulos que no son consumo sino insumo de armado de un regalo. Aparecen en reglas
#: de confianza altisima y lift de cientos ("si compra caja de regalo, compra
#: empaque"), que son ciertas y triviales. No se filtran: se marcan, para poder
#: mostrarlas por separado de las reglas entre productos de consumo.
ARTICULOS_DE_ARMADO = (
    "EMPAQUE", "CAJA REGALO", "BOLSA", "MUG", "SERPENTINA", "BOMBAS", "PAPEL REGALO",
    "MOÑA", "CINTA", "TARJETA",
)


# ----------------------------------------------------------------- compatibilidad


def reglas_desde_itemsets(itemsets: pd.DataFrame, metric: str = "confidence",
                          min_threshold: float = CONFIANZA_MINIMA) -> pd.DataFrame:
    """`association_rules` tolerante a la firma de la version instalada de mlxtend.

    mlxtend 0.23.2 agrego el parametro obligatorio `num_itemsets`; versiones previas no
    lo aceptan. Esto evita que el laboratorio dependa de una version exacta.
    """
    kwargs = {"metric": metric, "min_threshold": min_threshold}
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = len(itemsets)
    return association_rules(itemsets, **kwargs)


def _etiqueta(conjunto) -> str:
    return " + ".join(sorted(conjunto))


def es_articulo_de_armado(texto: str) -> bool:
    """True si la regla involucra insumos de regalo y no productos de consumo."""
    return any(clave in texto.upper() for clave in ARTICULOS_DE_ARMADO)


def reglas_representativas(reglas: pd.DataFrame, minimo_facturas: int = 0,
                           top: int | None = None, ordenar_por: str = "leverage",
                           consecuente_simple: bool = True) -> pd.DataFrame:
    """Quita el ruido estructural de una tabla de reglas antes de mostrarla.

    Un itemset frecuente de k productos genera decenas de reglas que dicen lo mismo
    repartidas de otra forma: con 5 productos salen mas de 20 permutaciones, todas con
    el mismo soporte y el mismo lift. Sin depurarlas, cualquier "top 20" se llena de
    un solo hallazgo repetido veinte veces.

    Tres criterios:

    * ``consecuente_simple``: se queda con las reglas de un solo producto a la derecha,
      que es la forma en que una regla de asociacion se lee en voz alta.
    * se deduplica por **itemset** (union de antecedente y consecuente), conservando la
      version de mayor `ordenar_por`.
    * se ordena por ``leverage`` y no por ``lift``, porque el lift crudo premia reglas
      de soporte minusculo: con 30 facturas de 68.807 se alcanzan lifts de miles que no
      describen a ningun cliente.
    """
    df = reglas
    if consecuente_simple and "k_consecuente" in df.columns:
        df = df[df["k_consecuente"] == 1]
    if minimo_facturas:
        df = df[df["facturas"] >= minimo_facturas]
    if df.empty:
        return df

    df = df.copy()
    df["itemset"] = [
        " + ".join(sorted(set(a.split(" + ")) | set(c.split(" + "))))
        for a, c in zip(df["antecedente"], df["consecuente"])
    ]
    df = (
        df.sort_values(ordenar_por, ascending=False)
        .drop_duplicates("itemset", keep="first")
    )
    df["es_armado"] = (df["antecedente"] + " + " + df["consecuente"]).map(
        es_articulo_de_armado
    )
    return df.head(top) if top else df


def _con_etiquetas(reglas: pd.DataFrame, n_transacciones: int) -> pd.DataFrame:
    df = reglas.copy()
    df["antecedente"] = df["antecedents"].apply(_etiqueta)
    df["consecuente"] = df["consequents"].apply(_etiqueta)
    df["k_antecedente"] = df["antecedents"].apply(len)
    df["k_consecuente"] = df["consequents"].apply(len)
    df["facturas"] = (df["support"] * n_transacciones).round().astype(int)
    # Probabilidad de que la combinacion ocurra por azar si los productos fueran
    # independientes: P(A) * P(B). El lift es justamente observado / esperado.
    df["soporte_esperado_azar"] = df["antecedent support"] * df["consequent support"]
    return df


# -------------------------------------------------------------------- extraccion


def minar(df_items: pd.DataFrame, min_support: float = SOPORTE_ANALISIS,
          solo_fa: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Devuelve ``(itemsets, reglas, n_transacciones)`` usando fp-growth.

    Se usa fp-growth porque el benchmark demuestra que entrega el mismo resultado que
    apriori en menos tiempo; para el analisis no hay razon para pagar el costo de
    apriori.
    """
    from mlxtend.preprocessing import TransactionEncoder

    transacciones = canastas_desde_items(df_items, solo_fa=solo_fa)
    te = TransactionEncoder()
    matriz = te.fit(transacciones).transform(transacciones, sparse=False)
    onehot = pd.DataFrame(matriz, columns=te.columns_, dtype=bool)

    itemsets = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    reglas = reglas_desde_itemsets(itemsets)
    reglas = _con_etiquetas(reglas, len(transacciones))
    return itemsets, reglas, len(transacciones)


# -------------------------------------------------------- preguntas del taller


def p1_reglas_k1(reglas: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """P1: reglas predecibles con un producto a cada lado (k=1).

    La columna `es_armado` separa dos clases de regla predecible que conviene no
    mezclar: la trivial de insumos ("si compra caja de regalo, compra empaque", con
    confianza de 0,95) y la de consumo, que es la que pide el enunciado con su ejemplo
    de avena -> pandebono.
    """
    k1 = reglas[(reglas["k_antecedente"] == 1) & (reglas["k_consecuente"] == 1)]
    k1 = k1[k1["facturas"] >= SOPORTE_ABSOLUTO_MINIMO].copy()
    k1["es_armado"] = (k1["antecedente"] + " + " + k1["consecuente"]).map(
        es_articulo_de_armado
    )
    cols = ["antecedente", "consecuente", "facturas", "support", "confidence", "lift",
            "es_armado"]
    return k1.sort_values("confidence", ascending=False).head(n)[cols]


def p3_efecto_parametros(df_items: pd.DataFrame,
                         soportes=(0.005, 0.002, 0.001, 0.0005, 0.0002)) -> pd.DataFrame:
    """P3: como cambia el resultado al mover el soporte (el tiempo lo mide src.bench)."""
    filas = []
    for soporte in soportes:
        itemsets, reglas, n_trans = minar(df_items, min_support=soporte)
        filas.append(
            {
                "min_support": soporte,
                "facturas_minimas": round(soporte * n_trans),
                "itemsets": len(itemsets),
                "itemsets_k2+": int((itemsets["itemsets"].apply(len) >= 2).sum()),
                "reglas": len(reglas),
                "reglas_con_soporte_abs_30": int(
                    (reglas["facturas"] >= SOPORTE_ABSOLUTO_MINIMO).sum()
                ) if len(reglas) else 0,
                "lift_maximo": round(float(reglas["lift"].max()), 2) if len(reglas) else None,
            }
        )
    return pd.DataFrame(filas)


def p4_espurias(df_items: pd.DataFrame, n: int = 15,
                min_support: float = SOPORTE_ESPURIAS) -> pd.DataFrame:
    """P4: reglas espurias, y con que parametros aparecen.

    Una regla espuria es la que tiene lift enorme y soporte minusculo: describe una
    coincidencia de un punado de facturas, no una conducta. La pregunta del taller pide
    ademas **con cuales parametros** aparecen, y la respuesta es concreta: no aparecen
    al soporte del analisis.

    Con ``min_support = 0.0005`` (34 facturas de 68.807) el conjunto no puede producir
    ninguna, porque el umbral ya garantiza 34 facturas por regla. Hay que bajar a
    ``0.0001`` (7 facturas) para que el algoritmo las devuelva. Esa es la respuesta: el
    soporte minimo es el parametro que las crea o las elimina.

    El criterio de descarte que se aplica en el resto del laboratorio es el soporte
    **absoluto** (>= 30 facturas), no el relativo, porque no depende del tamano del
    conjunto.
    """
    _, reglas, n_trans = minar(df_items, min_support=min_support)
    sospechosas = reglas[reglas["facturas"] < SOPORTE_ABSOLUTO_MINIMO].copy()
    sospechosas = reglas_representativas(
        sospechosas, ordenar_por="lift", consecuente_simple=True
    )
    sospechosas["min_support_usado"] = min_support
    sospechosas["facturas_minimas_del_umbral"] = round(min_support * n_trans)
    sospechosas["aparece_con_soporte_analisis"] = False
    cols = ["antecedente", "consecuente", "facturas", "support", "confidence", "lift",
            "leverage", "zhangs_metric", "min_support_usado",
            "facturas_minimas_del_umbral", "aparece_con_soporte_analisis", "es_armado"]
    cols = [c for c in cols if c in sospechosas.columns]
    return sospechosas.sort_values("lift", ascending=False).head(n)[cols]


def p5_reglas_k_mayor_1(reglas: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """P5: reglas con mas de un producto en el antecedente (itemsets de k > 2).

    Se deduplica por itemset: de otro modo la tabla serian veinte formas de escribir
    la misma combinacion de cinco productos.
    """
    multiples = reglas[reglas["k_antecedente"] > 1]
    representativas = reglas_representativas(multiples, ordenar_por="leverage", top=n)
    cols = ["antecedente", "consecuente", "facturas", "support", "confidence", "lift",
            "leverage", "es_armado"]
    cols = [c for c in cols if c in representativas.columns]
    return representativas[cols]


def p6_probabilidad_azar(reglas: pd.DataFrame, df_items: pd.DataFrame,
                         n_transacciones: int, n: int = 10) -> pd.DataFrame:
    """P6: probabilidad de que dos productos coincidan por azar, y test estadistico.

    Si A y B fueran independientes, ``P(A and B) = P(A) * P(B)``. El cociente entre lo
    observado y eso es el lift. Ademas se contrasta la tabla 2x2 con chi-cuadrado (y
    Fisher cuando alguna celda es pequena), que es lo que permite decir si la
    diferencia es significativa y no un accidente del tamano de muestra.
    """
    candidatas = reglas[
        (reglas["k_antecedente"] == 1)
        & (reglas["k_consecuente"] == 1)
        & (reglas["facturas"] >= SOPORTE_ABSOLUTO_MINIMO)
    ].sort_values("lift", ascending=False).head(n)

    canastas = df_items.groupby("factura")["producto"].apply(set)
    filas = []
    for _, regla in candidatas.iterrows():
        a = next(iter(regla["antecedents"]))
        b = next(iter(regla["consequents"]))
        ambos = int(sum(1 for c in canastas if a in c and b in c))
        solo_a = int(sum(1 for c in canastas if a in c and b not in c))
        solo_b = int(sum(1 for c in canastas if b in c and a not in c))
        ninguno = n_transacciones - ambos - solo_a - solo_b
        tabla = [[ambos, solo_a], [solo_b, ninguno]]

        chi2, p_chi2, _, _ = chi2_contingency(tabla)
        _, p_fisher = fisher_exact(tabla)
        filas.append(
            {
                "producto_a": a,
                "producto_b": b,
                "p_a": round(float(regla["antecedent support"]), 5),
                "p_b": round(float(regla["consequent support"]), 5),
                "p_a_por_p_b_esperado_azar": round(float(regla["soporte_esperado_azar"]), 6),
                "p_conjunto_observado": round(float(regla["support"]), 6),
                "veces_mas_que_el_azar_lift": round(float(regla["lift"]), 2),
                "facturas_con_ambos": ambos,
                "chi2": round(float(chi2), 1),
                "p_valor_chi2": f"{p_chi2:.2e}",
                "p_valor_fisher": f"{p_fisher:.2e}",
                "significativa_al_1pct": bool(p_fisher < 0.01),
            }
        )
    return pd.DataFrame(filas)


def p7_interesantes(reglas: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """P7: reglas con soporte suficiente, lift sobre el azar y algo que contar.

    Tres exclusiones, en orden de importancia:

    1. Se deduplica por itemset y se ordena por `leverage` (ver
       `reglas_representativas`). Sin esto la tabla entera era un unico hallazgo
       -- una combinacion de cinco productos presente en 40 facturas -- repetido
       veinte veces con los mismos numeros.
    2. Se marcan con `es_armado` las reglas entre insumos de regalo (empaque, caja,
       mug). Son ciertas y predecibles, pero no dicen nada del consumo.
    3. Se descartan los pares que son dos presentaciones del mismo producto, que
       comparten la primera palabra del nombre.
    """
    def distintos(fila) -> bool:
        a = _etiqueta(fila["antecedents"]).split()[:1]
        b = _etiqueta(fila["consequents"]).split()[:1]
        return a != b

    filtradas = reglas[
        (reglas["facturas"] >= SOPORTE_ABSOLUTO_MINIMO)
        & (reglas["lift"] >= LIFT_INTERESANTE)
    ]
    filtradas = filtradas[filtradas.apply(distintos, axis=1)]
    representativas = reglas_representativas(filtradas, ordenar_por="leverage", top=n)
    cols = ["antecedente", "consecuente", "facturas", "support", "confidence", "lift",
            "leverage", "es_armado"]
    cols = [c for c in cols if c in representativas.columns]
    return representativas[cols]


def sensibilidad_tipo_doc(df_items: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Anexo: las reglas principales con todos los CODTRA vs solo FA.

    Confirma que incluir cotizaciones y los otros tipos de documento no cambia el
    ranking de reglas, es decir que la decision de alcance no sesga el resultado.
    """
    _, todas, _ = minar(df_items, solo_fa=False)
    _, solo_fa, _ = minar(df_items, solo_fa=True)

    clave = ["antecedente", "consecuente"]
    top_todas = todas[todas["facturas"] >= SOPORTE_ABSOLUTO_MINIMO].nlargest(n, "lift")
    comparado = top_todas[clave + ["support", "confidence", "lift"]].merge(
        solo_fa[clave + ["support", "confidence", "lift"]],
        on=clave, how="left", suffixes=("_todos", "_solo_fa"),
    )
    comparado["delta_lift"] = (comparado["lift_solo_fa"] - comparado["lift_todos"]).round(3)
    return comparado


# ------------------------------------------------------------------------- CLI


def _guardar(df: pd.DataFrame, nombre: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    destino = TABLES / nombre
    df.to_csv(destino, index=False)
    print(f"[rules] {nombre:38s} {len(df):5d} filas")
    return destino


def main() -> None:
    df_items, df_fact = cargar()
    itemsets, reglas, n_trans = minar(df_items)
    print(f"[rules] {n_trans:,} transacciones, soporte {SOPORTE_ANALISIS} "
          f"(>= {round(SOPORTE_ANALISIS * n_trans)} facturas)")
    print(f"[rules] {len(itemsets):,} itemsets frecuentes, {len(reglas):,} reglas")

    _guardar(reglas.drop(columns=["antecedents", "consequents"]), "20_reglas_completas.csv")
    _guardar(p1_reglas_k1(reglas), "21_p1_reglas_k1.csv")
    _guardar(p3_efecto_parametros(df_items), "22_p3_efecto_parametros.csv")
    _guardar(p4_espurias(df_items), "23_p4_espurias.csv")
    _guardar(p5_reglas_k_mayor_1(reglas), "24_p5_reglas_k_mayor_1.csv")
    _guardar(p6_probabilidad_azar(reglas, df_items, n_trans), "25_p6_probabilidad_azar.csv")
    _guardar(p7_interesantes(reglas), "26_p7_interesantes.csv")
    _guardar(sensibilidad_tipo_doc(df_items), "27_sensibilidad_tipo_doc.csv")
    print("[rules] listo")


if __name__ == "__main__":
    main()
