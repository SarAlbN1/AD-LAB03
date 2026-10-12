"""Genera los fragmentos LaTeX que incluye el reporte, desde los CSV de `outputs/`.

El reporte no lleva ni una cifra escrita a mano: cada tabla sale de los datos. Si el
ETL o el benchmark cambian, `make report` produce un PDF con los numeros nuevos sin
tocar el .tex. Esto evita el error clasico del informe: una cifra del texto que quedo
de una corrida anterior.

Uso: ``python -m src.report_tables`` -> escribe ``report/tablas/*.tex``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.parse import OUTPUT_DIR

TABLAS = OUTPUT_DIR / "tables"
DESTINO = OUTPUT_DIR.parent / "report" / "tablas"

#: Caracteres que LaTeX interpreta y hay que escapar en texto venido de los datos.
_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escapar(valor) -> str:
    """Escapa un valor para que LaTeX no lo interprete.

    Hace falta de verdad: hay productos con `&` en el nombre y cifras con `$` y `%`.
    """
    texto = "" if pd.isna(valor) else str(valor)
    return "".join(_ESCAPES.get(c, c) for c in texto)


def _acortar(texto: str, maximo: int = 26) -> str:
    """Recorta nombres de producto para que la tabla quepa en una columna."""
    texto = str(texto)
    return texto if len(texto) <= maximo else texto[: maximo - 1] + "…"


#: Caracteres que puede tener una cifra ya formateada (miles, decimales, signos).
_CARACTERES_NUMERICOS = set("0123456789.,$%×-+e ")


def a_castellano(texto: str) -> str:
    """Pasa una cifra de notación inglesa a española: 1,234.5 -> 1.234,5.

    pandas y Python formatean con coma de miles y punto decimal. El informe está en
    español, donde es al revés, y mezclar las dos notaciones en el mismo documento
    ---'147,411 renglones' junto a '5.978 facturas'--- hace ambiguo si un separador es
    de miles o decimal. La conversión solo toca celdas que son enteramente numéricas,
    para no alterar nombres de producto.
    """
    if not texto or not set(texto) <= _CARACTERES_NUMERICOS:
        return texto
    return texto.translate(str.maketrans({",": ".", ".": ","}))


def tabular(df: pd.DataFrame, columnas: dict[str, str], alineacion: str,
            acortar: tuple[str, ...] = (), formato: dict | None = None) -> str:
    """Convierte un DataFrame en un `tabular` de LaTeX listo para incluir.

    `columnas` mapea nombre en el CSV -> encabezado en el PDF, y fija el orden.
    """
    formato = formato or {}
    filas = []
    for _, fila in df.iterrows():
        celdas = []
        for col in columnas:
            valor = fila[col]
            if col in formato:
                valor = formato[col](valor)
            if col in acortar:
                valor = _acortar(valor)
            celdas.append(a_castellano(escapar(valor)))
        filas.append(" & ".join(celdas) + r" \\")

    # Los encabezados los escribe este modulo, no vienen de los datos: son LaTeX
    # valido y no se escapan. Escaparlos imprimia literalmente "$\Delta$\%".
    encabezado = " & ".join(rf"\textbf{{{v}}}" for v in columnas.values())
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{alineacion}}}",
            r"\toprule",
            encabezado + r" \\",
            r"\midrule",
            *filas,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def _leer(nombre: str) -> pd.DataFrame | None:
    ruta = TABLAS / nombre
    if not ruta.exists():
        print(f"[report] falta {nombre}, se omite su tabla")
        return None
    return pd.read_csv(ruta)


def _escribir(nombre: str, contenido: str) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / nombre).write_text(contenido + "\n", encoding="utf-8")
    print(f"[report] tabla -> report/tablas/{nombre}")


def _p_valor_latex(texto: str) -> str:
    """Convierte '1.536e-280' en '1{,}5\\times 10^{-280}', legible en modo matematico."""
    try:
        mantisa, exponente = str(texto).lower().split("e")
        return rf"{float(mantisa):.1f}".replace(".", "{,}") + rf"\times 10^{{{int(exponente)}}}"
    except (ValueError, AttributeError):
        return str(texto)


def _miles(valor) -> str:
    try:
        return f"{float(valor):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return str(valor)


def tabla_dia_semana() -> None:
    df = _leer("15_comparativa_dia_semana.csv")
    if df is None:
        return
    _escribir(
        "dia_semana.tex",
        tabular(
            df,
            {
                "dia_semana": "Día",
                "facturas": "Facturas",
                "indice_facturas": "Índice",
                "delta_pct_facturas": r"$\Delta$\%",
                "canasta_promedio": "Canasta",
            },
            "lrrrr",
            formato={
                "facturas": _miles,
                "canasta_promedio": lambda v: f"{float(v):.2f}",
            },
        ),
    )


def tabla_benchmark() -> None:
    df = _leer("12_benchmark_comparativa.csv")
    if df is None:
        return
    # El reporte tiene dos paginas: entra una sola familia de corridas. Se eligen las
    # de `max_len` sin limite, que muestran el costo real de cada algoritmo; la grilla
    # completa queda en outputs/tables/.
    #
    # Las celdas de apriori sin valor **no se descartan**: significan que el algoritmo
    # agoto la memoria a ese soporte, que es el resultado principal del experimento.
    # Borrar esas filas dejaria la tabla contando solo los casos en que apriori pudo.
    sin_limite = df[df["max_len"].isna()]
    df = (sin_limite if len(sin_limite) >= 3 else df).head(6)

    def tiempo(v) -> str:
        return "sin memoria" if pd.isna(v) else f"{float(v):.2f}"

    def speedup(v) -> str:
        return "---" if pd.isna(v) else f"{float(v):.1f}×"
    _escribir(
        "benchmark.tex",
        tabular(
            df,
            {
                "min_support": "Soporte",
                "max_len": r"max\_len",
                "apriori": "apriori (s)",
                "fpgrowth": "fp-growth (s)",
                "speedup_fpgrowth": "Speedup",
            },
            "llrrr",
            formato={
                "apriori": tiempo,
                "fpgrowth": tiempo,
                "speedup_fpgrowth": speedup,
                "max_len": lambda v: "sin límite" if pd.isna(v) else f"{int(float(v))}",
            },
        ),
    )


def tabla_parametros() -> None:
    df = _leer("22_p3_efecto_parametros.csv")
    if df is None:
        return
    _escribir(
        "parametros.tex",
        tabular(
            df,
            {
                "min_support": "Soporte",
                "facturas_minimas": "Facturas mín.",
                "itemsets": "Itemsets",
                "reglas": "Reglas",
                "reglas_con_soporte_abs_30": r"Reglas $\geq$30",
                "lift_maximo": "Lift máx.",
            },
            "lrrrrr",
            formato={"itemsets": _miles, "reglas": _miles,
                     "reglas_con_soporte_abs_30": _miles},
        ),
    )


def tabla_reglas() -> None:
    df = _leer("26_p7_interesantes.csv")
    if df is None:
        return
    if "es_armado" in df.columns:
        df = df[~df["es_armado"]]
    _escribir(
        "reglas.tex",
        tabular(
            df.head(8),
            {
                "antecedente": "Antecedente",
                "consecuente": "Consecuente",
                "facturas": "Fact.",
                "confidence": "Conf.",
                "lift": "Lift",
            },
            "llrrr",
            acortar=("antecedente", "consecuente"),
            formato={
                "confidence": lambda v: f"{float(v):.2f}",
                "lift": lambda v: f"{float(v):.2f}",
            },
        ),
    )


def tabla_top10() -> None:
    df = _leer("06_top10.csv")
    if df is None:
        return
    _escribir(
        "top10.tex",
        tabular(
            df.head(6),   # el reporte tiene dos paginas; el ranking completo esta en outputs/
            {
                "producto": "Producto",
                "facturas": "Facturas",
                "soporte": "Soporte",
                "pct_del_lider": r"\% del líder",
            },
            "lrrr",
            acortar=("producto",),
            formato={
                "facturas": _miles,
                "soporte": lambda v: f"{float(v) * 100:.1f}\\%",
                "pct_del_lider": lambda v: f"{float(v):.0f}",
            },
        ),
    )


def macros_cifras() -> None:
    """Macros LaTeX con las cifras que el texto cita en prosa.

    Asi una frase como "los 13 archivos aportan 147.411 renglones" no se escribe a
    mano: se define aqui desde el CSV y el texto usa el macro.
    """
    resumen = _leer("01_resumen_general.csv")
    if resumen is None:
        return
    valores = dict(zip(resumen["metrica"], resumen["valor"]))

    def macro(nombre: str, valor: str) -> str:
        return rf"\newcommand{{\{nombre}}}{{{a_castellano(str(valor))}}}"

    lineas = [
        macro("cifraRenglones", escapar(valores.get("Registros (line-items)", ""))),
        macro("cifraFacturas", escapar(valores.get("Facturas", ""))),
        macro("cifraProductos", escapar(valores.get("Productos distintos", ""))),
        macro("cifraCanasta", escapar(valores.get("Renglones por factura (media)", ""))),
        macro("cifraUnitarias", escapar(valores.get("Facturas de un solo producto", ""))),
        macro("cifraTicketMediano", escapar(valores.get("Ticket mediano", ""))),
        macro("cifraTicketPromedio", escapar(valores.get("Ticket promedio", ""))),
    ]

    uniformidad = _leer("15b_test_uniformidad_dia.csv")
    if uniformidad is not None:
        u = dict(zip(uniformidad["clave"], uniformidad["valor"]))
        lineas.append(macro("cifraChiCuadrado", escapar(u.get("chi2", ""))))
        lineas.append(macro("cifraPValor", _p_valor_latex(u.get("p_valor", ""))))
        lineas.append(macro("cifraFacturasPanel", escapar(_miles(u.get("facturas", 0)))))

    equivalencia = _leer("13_equivalencia_itemsets.csv")
    if equivalencia is not None and "identicos" in equivalencia.columns:
        # Solo cuentan las combinaciones que ambos algoritmos completaron: donde
        # apriori se quedo sin memoria no hay nada que comparar.
        comparables = int(equivalencia["comparable"].sum())
        lineas.append(macro("cifraEquivalentes", str(int(equivalencia["identicos"].sum()))))
        lineas.append(macro("cifraCombinaciones", str(comparables)))
        lineas.append(macro("cifraSinCompletar", str(len(equivalencia) - comparables)))

    bench = _leer("10_benchmark_crudo.csv")
    if bench is not None and "estado" in bench.columns:
        sin_memoria = bench[(bench["algoritmo"] == "apriori")
                            & (bench["estado"] == "memoria")]
        total_apriori = int((bench["algoritmo"] == "apriori").sum())
        lineas.append(macro("cifraAprioriSinMemoria", str(len(sin_memoria))))
        lineas.append(macro("cifraAprioriCorridas", str(total_apriori)))

    _escribir("cifras.tex", "\n".join(lineas))


def main() -> None:
    tabla_dia_semana()
    tabla_top10()
    tabla_parametros()
    tabla_reglas()
    tabla_benchmark()
    macros_cifras()
    print("[report] fragmentos listos en report/tablas/")


if __name__ == "__main__":
    main()
