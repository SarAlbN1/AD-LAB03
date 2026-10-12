"""Capa bronze: los .DBF pasados a parquet **sin ninguna regla de negocio**.

Lo unico que se hace aqui es decodificar el formato (ver `src.dbf_reader`) y agregar
columnas de linaje. Nada se normaliza, nada se filtra, nada se agrupa: si mas adelante
hay que revisar una decision de limpieza, el bronze permite volver al dato original sin
releer 110 MB de DBF.

Dos columnas de linaje importan:

``_archivo``  de que DBF viene la fila.
``_fila``     posicion fisica del registro dentro del DBF. Es la **unica** traza del
              orden en que se facturo, y de ahi sale la hora estimada. Si se pierde
              este orden, el proxy horario deja de tener sentido.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.dbf_reader import DBFTable
from src.etl import paths

# El bronze guarda los 50 campos del DBF tal cual, incluidos los que parecen vacios.
# Descartarlos aqui seria una decision de negocio, y esta capa no toma ninguna: el
# parquet con codificacion de diccionario deja una columna constante en casi nada, y
# `src.etl.validate.columnas_constantes` reporta cuales no aportan.


def sha256(ruta: Path, bloque: int = 1 << 20) -> str:
    """Huella del archivo de origen: es la clave del cache del pipeline."""
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for trozo in iter(lambda: fh.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def destino_bronze(ruta: Path) -> Path:
    return paths.BRONZE / f"{ruta.stem}.parquet"


def extraer(ruta: Path, forzar: bool = False) -> dict:
    """Pasa un .DBF a parquet bronze y devuelve su fila de manifiesto."""
    from src.parse import fecha_desde_nombre

    tabla = DBFTable(ruta)
    huella = sha256(ruta)
    salida = destino_bronze(ruta)

    fila_manifiesto = {
        "archivo": ruta.name,
        "carpeta": ruta.parent.name,
        "fecha": fecha_desde_nombre(ruta),
        "sha256": huella,
        "bytes": ruta.stat().st_size,
        "version_dbf": f"0x{tabla.version:02x}",
        "registros_header": tabla.record_count,
        "campos": len(tabla.fields),
        "bronze": salida.name,
        "extraido_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if salida.exists() and not forzar:
        previo = pd.read_parquet(salida, columns=["_sha256"]).head(1)
        if len(previo) and previo["_sha256"].iloc[0] == huella:
            fila_manifiesto["registros_bronze"] = len(pd.read_parquet(salida, columns=["_fila"]))
            fila_manifiesto["estado"] = "cache"
            return fila_manifiesto

    filas = []
    for i, registro in enumerate(tabla.records()):
        registro["_fila"] = i
        filas.append(registro)

    df = pd.DataFrame(filas)
    df["_archivo"] = ruta.name
    df["_sha256"] = huella
    paths.BRONZE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida, index=False)

    fila_manifiesto["registros_bronze"] = len(df)
    fila_manifiesto["estado"] = "extraido"
    return fila_manifiesto


def extraer_todos(rutas: list[Path] | None = None, forzar: bool = False) -> pd.DataFrame:
    """Extrae los 13 archivos y escribe el manifiesto."""
    from src.parse import archivos_dbf

    rutas = rutas if rutas is not None else archivos_dbf()
    paths.preparar()

    filas = []
    for ruta in rutas:
        fila = extraer(ruta, forzar=forzar)
        filas.append(fila)
        print(f"[extract] {fila['archivo']:18s} {fila['registros_bronze']:>7,} filas "
              f"({fila['estado']})")

    manifiesto = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)
    manifiesto.to_csv(paths.MANIFIESTO, index=False)
    print(f"[extract] manifiesto -> {paths.MANIFIESTO.relative_to(paths.RAIZ)} "
          f"({len(manifiesto)} archivos, "
          f"{manifiesto['registros_bronze'].sum():,} registros)")
    return manifiesto


def leer_bronze(archivos: list[str] | None = None) -> pd.DataFrame:
    """Concatena el bronze respetando el orden fisico de cada archivo."""
    rutas = sorted(paths.BRONZE.glob("*.parquet"))
    if archivos is not None:
        permitidos = {Path(a).stem for a in archivos}
        rutas = [r for r in rutas if r.stem in permitidos]
    if not rutas:
        raise FileNotFoundError(f"no hay parquet en {paths.BRONZE}; correr extract")
    partes = [pd.read_parquet(r) for r in rutas]
    return pd.concat(partes, ignore_index=True).sort_values(
        ["_archivo", "_fila"]
    ).reset_index(drop=True)


def main() -> None:
    extraer_todos()


if __name__ == "__main__":
    main()
