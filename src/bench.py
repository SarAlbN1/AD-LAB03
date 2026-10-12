"""Comparacion de rendimiento apriori vs fp-growth (punto 5 del enunciado).

Diseno experimental
-------------------
* **Misma entrada** para los dos algoritmos: una sola matriz one-hot construida una
  vez con `TransactionEncoder`. La unica variable del experimento es la implementacion.
* **20 combinaciones de parametros** (punto 5.d): 5 soportes minimos x 4 valores de
  `max_len`, aplicadas a cada algoritmo.
* **3 repeticiones** por combinacion; se reporta la **mediana** porque una sola
  medicion queda a merced del planificador del sistema operativo.
* Cada corrida se ejecuta en un **proceso hijo** con limite de tiempo. Asi una
  configuracion que explota (apriori con soporte muy bajo) queda registrada como
  censurada en vez de colgar el laboratorio. El hijo tambien da una medida limpia de
  memoria pico (`ru_maxrss`), que incluye lo que asigna numpy y que `tracemalloc` no ve.
* **Prueba de equivalencia**: de cada corrida se guarda una firma SHA-256 de sus
  itemsets frecuentes. Si apriori y fp-growth dan la misma firma para los mismos
  parametros, entonces difieren en tiempo pero no en resultado, y eso queda demostrado
  en vez de supuesto.

Uso: ``python -m src.bench`` (fuera de Jupyter: el kernel anade overhead que
contaminaria las mediciones).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import platform
import resource
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, hmine
from mlxtend.preprocessing import TransactionEncoder

from src.parse import OUTPUT_DIR, canastas_desde_items, cargar
from src.rules import reglas_desde_itemsets

FIGS = OUTPUT_DIR / "figs"
TABLES = OUTPUT_DIR / "tables"
CACHE_ONEHOT = OUTPUT_DIR / "onehot.parquet"

#: Grilla del punto 5.d: 5 x 4 = 20 combinaciones por algoritmo.
SOPORTES = (0.0002, 0.0005, 0.001, 0.002, 0.005)
MAX_LEN = (2, 3, 4, None)
REPETICIONES = 3

#: Segundos por corrida antes de declararla censurada.
TIMEOUT_S = 600

#: Umbral de confianza para contar reglas en cada corrida.
MIN_CONFIANZA = 0.05

ALGORITMOS = {
    "apriori": apriori,
    "fpgrowth": fpgrowth,
}
#: Implementaciones extra, solo como contraste de familia algoritmica (no las pide el
#: enunciado): fpmax da itemsets maximales y hmine es otra estrategia sin arbol.
ALGORITMOS_EXTRA = {
    "fpmax": fpmax,
    "hmine": hmine,
}


@dataclass
class Corrida:
    algoritmo: str
    min_support: float
    max_len: int | None
    repeticion: int
    estado: str  # ok | timeout | error
    tiempo_s: float | None
    tiempo_reglas_s: float | None
    memoria_mb: float | None
    n_itemsets: int | None
    n_reglas: int | None
    firma: str | None
    detalle: str = ""


# ------------------------------------------------------------------ one-hot comun


def construir_onehot(usar_cache: bool = True) -> pd.DataFrame:
    """Matriz booleana facturas x productos, entrada unica de los dos algoritmos.

    La produce el ETL (`src.etl.load.construir_onehot`); aqui solo se lee. Si no esta,
    se construye sobre la marcha para que el benchmark pueda correr solo.
    """
    from src.etl import paths as etl_paths

    if usar_cache and etl_paths.GOLD_ONEHOT.exists():
        return pd.read_parquet(etl_paths.GOLD_ONEHOT)
    if usar_cache and CACHE_ONEHOT.exists():
        return pd.read_parquet(CACHE_ONEHOT)

    df_items, _ = cargar()
    transacciones = canastas_desde_items(df_items)
    te = TransactionEncoder()
    matriz = te.fit(transacciones).transform(transacciones, sparse=False)
    df = pd.DataFrame(matriz, columns=te.columns_, dtype=bool)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_ONEHOT, index=False)
    print(f"[bench] one-hot {df.shape[0]:,} x {df.shape[1]:,} -> {CACHE_ONEHOT.name}")
    return df


def firma_itemsets(itemsets: pd.DataFrame) -> str:
    """Firma estable del resultado: permite comparar algoritmos sin guardar todo.

    Se redondea el soporte a 10 decimales para que una diferencia de ultimo bit en el
    punto flotante no se confunda con una diferencia de resultado.
    """
    piezas = sorted(
        "|".join(sorted(items)) + f"={round(float(sup), 10)}"
        for items, sup in zip(itemsets["itemsets"], itemsets["support"])
    )
    return hashlib.sha256("\n".join(piezas).encode()).hexdigest()[:16]


# --------------------------------------------------------------- corrida aislada


def _worker(cola, nombre: str, df: pd.DataFrame, min_support: float,
            max_len: int | None, funcion) -> None:
    """Ejecuta una corrida en el proceso hijo y devuelve sus metricas."""
    try:
        t0 = time.perf_counter_ns()
        itemsets = funcion(df, min_support=min_support, use_colnames=True,
                          max_len=max_len)
        tiempo = (time.perf_counter_ns() - t0) / 1e9

        n_reglas, tiempo_reglas = 0, 0.0
        # fpmax devuelve itemsets maximales: association_rules no aplica sobre ellos.
        if nombre not in ALGORITMOS_EXTRA and len(itemsets):
            t1 = time.perf_counter_ns()
            reglas = reglas_desde_itemsets(
                itemsets, metric="confidence", min_threshold=MIN_CONFIANZA
            )
            tiempo_reglas = (time.perf_counter_ns() - t1) / 1e9
            n_reglas = len(reglas)

        cola.put(
            {
                "estado": "ok",
                "tiempo_s": tiempo,
                "tiempo_reglas_s": tiempo_reglas,
                "memoria_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "n_itemsets": len(itemsets),
                "n_reglas": n_reglas,
                "firma": firma_itemsets(itemsets) if len(itemsets) else "vacio",
                "detalle": "",
            }
        )
    except Exception as exc:  # pragma: no cover - se registra, no se propaga
        cola.put({"estado": "error", "detalle": f"{type(exc).__name__}: {exc}"})


def ejecutar_corrida(nombre: str, funcion, df: pd.DataFrame, min_support: float,
                     max_len: int | None, repeticion: int,
                     timeout: int = TIMEOUT_S) -> Corrida:
    """Lanza una corrida con limite de tiempo. 'fork' comparte la matriz sin copiarla."""
    ctx = mp.get_context("fork")
    cola = ctx.Queue()
    proc = ctx.Process(target=_worker,
                       args=(cola, nombre, df, min_support, max_len, funcion))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return Corrida(nombre, min_support, max_len, repeticion, "timeout",
                       None, None, None, None, None, None,
                       f"supero {timeout}s")

    # Si el sistema operativo mato al hijo (exitcode negativo = senal), casi siempre es
    # el OOM killer: la corrida pidio mas memoria de la que el contenedor concede. Eso
    # no es un fallo del laboratorio sino un resultado del experimento, y por eso se
    # registra con estado propio en vez de confundirse con una excepcion de Python.
    if proc.exitcode is not None and proc.exitcode < 0:
        return Corrida(nombre, min_support, max_len, repeticion, "memoria",
                       None, None, None, None, None, None,
                       f"el SO termino el proceso (senal {-proc.exitcode}); "
                       "memoria insuficiente")

    resultado = cola.get() if not cola.empty() else {"estado": "error",
                                                     "detalle": "sin resultado"}
    if resultado["estado"] != "ok":
        return Corrida(nombre, min_support, max_len, repeticion, "error",
                       None, None, None, None, None, None,
                       resultado.get("detalle", ""))
    return Corrida(nombre, min_support, max_len, repeticion, "ok",
                   resultado["tiempo_s"], resultado["tiempo_reglas_s"],
                   resultado["memoria_mb"], resultado["n_itemsets"],
                   resultado["n_reglas"], resultado["firma"])


# ------------------------------------------------------------------- experimento


def correr_grilla(df: pd.DataFrame, algoritmos: dict | None = None,
                  repeticiones: int = REPETICIONES,
                  timeout: int = TIMEOUT_S) -> pd.DataFrame:
    """Recorre la grilla completa y devuelve una fila por corrida.

    Se avanza de soporte alto a soporte bajo. Si un algoritmo se cae por timeout en un
    soporte, los soportes mas bajos del mismo `max_len` se marcan censurados sin
    ejecutarlos: en apriori el costo crece al bajar el soporte, nunca baja.
    """
    algoritmos = algoritmos or ALGORITMOS
    filas: list[Corrida] = []
    total = len(algoritmos) * len(SOPORTES) * len(MAX_LEN) * repeticiones
    hecho = 0

    for nombre, funcion in algoritmos.items():
        for max_len in MAX_LEN:
            censurado = False
            motivo_censura = "timeout"
            for soporte in sorted(SOPORTES, reverse=True):
                for rep in range(1, repeticiones + 1):
                    hecho += 1
                    if censurado:
                        filas.append(Corrida(nombre, soporte, max_len, rep, motivo_censura,
                                             None, None, None, None, None, None,
                                             "censurado: un soporte mayor ya fallo"))
                        continue
                    corrida = ejecutar_corrida(nombre, funcion, df, soporte, max_len,
                                               rep, timeout)
                    filas.append(corrida)
                    etiqueta = f"max_len={max_len}" if max_len else "max_len=None"
                    print(f"[bench] {hecho:3d}/{total} {nombre:9s} sup={soporte:<7} "
                          f"{etiqueta:12s} rep={rep} -> {corrida.estado} "
                          f"{corrida.tiempo_s if corrida.tiempo_s else ''}")
                    if corrida.estado in ("timeout", "memoria"):
                        censurado = True
                        motivo_censura = corrida.estado

    return pd.DataFrame([asdict(f) for f in filas])


def resumir(crudo: pd.DataFrame) -> pd.DataFrame:
    """Mediana por combinacion y speedup de fp-growth frente a apriori."""
    ok = crudo[crudo["estado"] == "ok"]
    agg = (
        ok.groupby(["algoritmo", "min_support", "max_len"], dropna=False)
        .agg(
            tiempo_mediano_s=("tiempo_s", "median"),
            tiempo_min_s=("tiempo_s", "min"),
            tiempo_max_s=("tiempo_s", "max"),
            tiempo_reglas_s=("tiempo_reglas_s", "median"),
            memoria_mb=("memoria_mb", "max"),
            n_itemsets=("n_itemsets", "max"),
            n_reglas=("n_reglas", "max"),
            corridas=("tiempo_s", "count"),
        )
        .reset_index()
    )
    pivote = agg.pivot_table(index=["min_support", "max_len"], columns="algoritmo",
                             values="tiempo_mediano_s", dropna=False)
    if {"apriori", "fpgrowth"} <= set(pivote.columns):
        pivote["speedup_fpgrowth"] = pivote["apriori"] / pivote["fpgrowth"]
    comparativa = pivote.reset_index()
    return agg, comparativa


def verificar_equivalencia(crudo: pd.DataFrame) -> pd.DataFrame:
    """Compara las firmas de apriori y fp-growth para cada combinacion.

    Esta tabla es la evidencia de la pregunta 2 del taller.
    """
    ok = crudo[(crudo["estado"] == "ok") & (crudo["firma"].notna())]
    firmas = ok.pivot_table(index=["min_support", "max_len"], columns="algoritmo",
                            values="firma", aggfunc="first", dropna=False)
    columnas = [c for c in ("apriori", "fpgrowth") if c in firmas.columns]
    firmas = firmas[columnas].reset_index()
    if len(columnas) == 2:
        # Una combinacion solo es comparable si los dos algoritmos la completaron.
        # Donde apriori se quedo sin memoria no hay nada que contrastar, y contar esas
        # filas como "no identicas" diria algo falso sobre la equivalencia.
        firmas["comparable"] = firmas["apriori"].notna() & firmas["fpgrowth"].notna()
        firmas["identicos"] = firmas["comparable"] & (
            firmas["apriori"] == firmas["fpgrowth"]
        )
    return firmas


def graficar(agg: pd.DataFrame) -> None:
    """Costo y resultado frente al soporte minimo, en escala log-log.

    El eje x va de soporte alto a soporte bajo, que es el sentido en que el problema se
    vuelve dificil. Donde un algoritmo no completo, su curva simplemente **termina**:
    no se rellena ni se extrapola, y la zona se sombrea y se rotula, porque de otro
    modo la interrupcion se leeria como datos faltantes en vez de como el resultado que
    es --- apriori agota la memoria cuando hay muchos items frecuentes, porque
    materializa los candidatos de cada nivel.
    """
    base = agg[agg["max_len"].isna()]
    if base.empty:  # si max_len=None quedo censurado, se usa el mayor disponible
        base = agg[agg["max_len"] == agg["max_len"].max()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    completados: dict[str, set] = {}
    for nombre, grupo in base.groupby("algoritmo"):
        grupo = grupo.sort_values("min_support")
        completados[nombre] = set(grupo["min_support"])
        ax1.plot(grupo["min_support"], grupo["tiempo_mediano_s"], marker="o",
                 label=nombre)
        ax2.plot(grupo["min_support"], grupo["n_reglas"], marker="s", label=nombre)

    for eje, titulo, etiqueta_y in (
        (ax1, "Costo de ejecucion", "Tiempo mediano (s)"),
        (ax2, "Reglas obtenidas", "Reglas encontradas"),
    ):
        eje.set_xscale("log")
        eje.set_yscale("log")
        eje.set_xlabel("min_support")
        eje.set_ylabel(etiqueta_y)
        eje.set_title(titulo)
        eje.grid(True, which="both", alpha=0.2)
        eje.invert_xaxis()
        # Las marcas se fijan despues de la escala logaritmica: al reves, el
        # localizador del eje log las sobrescribe y solo queda una potencia de diez.
        eje.set_xticks(list(SOPORTES))
        eje.set_xticklabels([f"{s:g}" for s in SOPORTES], fontsize=7)
        eje.tick_params(axis="x", which="minor", bottom=False)
        eje.legend(loc="upper left", fontsize=8)

    faltan = set(SOPORTES) - completados.get("apriori", set(SOPORTES))
    if faltan and "apriori" in completados:
        corte = max(faltan)
        for eje in (ax1, ax2):
            eje.axvspan(min(SOPORTES) * 0.85, corte * 1.18, color="#c44e52", alpha=0.08)
            # El rotulo va abajo y centrado en la banda, lejos de la leyenda.
            centro = (corte * min(SOPORTES)) ** 0.5
            inferior, superior = eje.get_ylim()
            eje.text(centro, inferior * (superior / inferior) ** 0.06,
                     "apriori: sin memoria", fontsize=7.5, color="#c44e52",
                     ha="center", va="bottom", style="italic")

    fig.tight_layout()
    destino = FIGS / "10_benchmark.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)
    print(f"[bench] figura -> {destino}")


def entorno() -> pd.DataFrame:
    """Contexto de la medicion, para que el reporte declare donde se midio."""
    import mlxtend
    import numpy

    filas = [
        ("python", platform.python_version()),
        ("plataforma", platform.platform()),
        ("maquina", platform.machine()),
        ("cpus", str(mp.cpu_count())),
        ("mlxtend", mlxtend.__version__),
        ("pandas", pd.__version__),
        ("numpy", numpy.__version__),
        ("timeout_s", str(TIMEOUT_S)),
        ("repeticiones", str(REPETICIONES)),
    ]
    return pd.DataFrame(filas, columns=["clave", "valor"])


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = construir_onehot()
    print(f"[bench] matriz {df.shape[0]:,} transacciones x {df.shape[1]:,} productos")

    crudo = correr_grilla(df)
    crudo.to_csv(TABLES / "10_benchmark_crudo.csv", index=False)

    agg, comparativa = resumir(crudo)
    agg.to_csv(TABLES / "11_benchmark_por_combinacion.csv", index=False)
    comparativa.to_csv(TABLES / "12_benchmark_comparativa.csv", index=False)

    equivalencia = verificar_equivalencia(crudo)
    equivalencia.to_csv(TABLES / "13_equivalencia_itemsets.csv", index=False)

    entorno().to_csv(TABLES / "14_entorno.csv", index=False)
    graficar(agg)

    print("\n[bench] comparativa (tiempo mediano en segundos):")
    print(comparativa.to_string(index=False))
    if "identicos" in equivalencia.columns:
        comparables = int(equivalencia["comparable"].sum())
        iguales = int(equivalencia["identicos"].sum())
        print(f"\n[bench] itemsets identicos en {iguales} de {comparables} "
              f"combinaciones comparables "
              f"({len(equivalencia) - comparables} sin comparar: un algoritmo no completo)")
    print(f"[bench] corridas totales: {len(crudo)} "
          + ", ".join(f"{k}={v}" for k, v in crudo["estado"].value_counts().items()))

    sin_completar = crudo[crudo["estado"].isin(["timeout", "memoria"])]
    if len(sin_completar):
        print("\n[bench] corridas que no completaron, por algoritmo y causa:")
        print(sin_completar.groupby(["algoritmo", "estado"]).size().to_string())


if __name__ == "__main__":
    main()
