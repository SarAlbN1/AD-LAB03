"""Lector de tablas DBF de Visual FoxPro (version 0x32).

Por que un lector propio y no `dbfread`: los archivos de este laboratorio guardan la
descripcion del producto en un campo de tipo ``V`` (Varchar de Visual FoxPro). En ese
tipo el ultimo byte del campo almacena la longitud util del texto, y `dbfread` lo
devuelve como parte del dato. El resultado es basura al final de cada descripcion:

    'LUCKY DAIQUIRI X 10                         ... B'   <- la 'B' es 0x42 == 66

donde 66 es la longitud, no una letra. `tests/test_dbf_reader.py` deja constancia de
la diferencia contra `dbfread`.

Uso:

    >>> from src.dbf_reader import DBFTable
    >>> t = DBFTable('archivos/lunes/fc01042024.DBF')
    >>> t.record_count
    1168
    >>> next(iter(t))['DESCRI']
    'LUCKY DAIQUIRI X 10'
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

# Marcas del formato DBF.
_HEADER_LEN = 32
_FIELD_DESC_LEN = 32
_FIELD_TERMINATOR = 0x0D
_DELETED_MARK = b"*"

# Codificacion de los .DBF de la tienda: no declaran code page util, y el contenido
# real son descripciones en espanol con tildes y 'N' con virgulilla de un sistema DOS
# latino. latin-1 nunca falla al decodificar (mapea los 256 bytes) y preserva esos
# caracteres.
_ENCODING = "latin-1"

#: Campo de banderas de nulos que Visual FoxPro agrega al final de la tabla.
NULL_FLAGS_FIELD = "_NullFlags"


@dataclass(frozen=True)
class DBFField:
    """Descriptor de una columna de la tabla."""

    name: str
    type: str
    length: int
    decimals: int
    flags: int
    offset: int  # desplazamiento dentro del registro, ya sin el byte de borrado

    @property
    def is_varlength(self) -> bool:
        """Visual FoxPro marca con 0x04 los campos de longitud variable."""
        return bool(self.flags & 0x04)


class DBFTable:
    """Tabla DBF leida en memoria.

    El archivo completo se carga de una vez (el mas grande del laboratorio son 20 MB)
    porque el parseo por registro sobre bytes en memoria es un orden de magnitud mas
    rapido que ir al disco 28.000 veces.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as fh:
            header = fh.read(_HEADER_LEN)
            if len(header) < _HEADER_LEN:
                raise ValueError(f"{self.path.name}: header truncado")

            self.version: int = header[0]
            yy, mm, dd = header[1], header[2], header[3]
            self.last_update = (2000 + yy, mm, dd)
            self.record_count, self._header_length, self._record_length = struct.unpack(
                "<IHH", header[4:12]
            )

            self.fields: list[DBFField] = []
            offset = 0  # el byte 0 del registro es la marca de borrado
            while True:
                desc = fh.read(_FIELD_DESC_LEN)
                if not desc or desc[0] == _FIELD_TERMINATOR:
                    break
                name = desc[0:11].split(b"\x00")[0].decode(_ENCODING)
                field = DBFField(
                    name=name,
                    type=chr(desc[11]),
                    length=desc[16],
                    decimals=desc[17],
                    flags=desc[18],
                    offset=offset,
                )
                self.fields.append(field)
                offset += field.length

            fh.seek(self._header_length)
            self._data = fh.read(self.record_count * self._record_length)

        self._by_name = {f.name: f for f in self.fields}

    # ------------------------------------------------------------------ metadatos

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def __len__(self) -> int:
        return self.record_count

    def __repr__(self) -> str:  # pragma: no cover - ayuda en el notebook
        return (
            f"DBFTable({self.path.name!r}, version=0x{self.version:02x}, "
            f"records={self.record_count}, fields={len(self.fields)})"
        )

    # ------------------------------------------------------------------- lectura

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.records()

    def records(self, include_deleted: bool = False) -> Iterator[dict[str, Any]]:
        """Itera los registros como diccionarios ``{nombre_campo: valor}``."""
        rlen = self._record_length
        data = self._data
        fields = [f for f in self.fields if f.name != NULL_FLAGS_FIELD]
        for i in range(self.record_count):
            raw = data[i * rlen : (i + 1) * rlen]
            if len(raw) < rlen:
                break  # archivo truncado: se ignora el resto
            if not include_deleted and raw[0:1] == _DELETED_MARK:
                continue
            body = raw[1:]
            yield {f.name: self._decode(f, body) for f in fields}

    def raw_record(self, index: int) -> bytes:
        """Registro sin decodificar, util para los tests del parser."""
        rlen = self._record_length
        return self._data[index * rlen : (index + 1) * rlen]

    def field(self, name: str) -> DBFField:
        return self._by_name[name]

    # ------------------------------------------------------------- decodificacion

    def _decode(self, field: DBFField, body: bytes) -> Any:
        raw = body[field.offset : field.offset + field.length]
        ftype = field.type

        if ftype == "V":
            return self._decode_varchar(raw)
        if ftype in ("C", "M"):
            return raw.decode(_ENCODING).strip()
        if ftype == "N":
            return self._decode_number(raw, field.decimals)
        if ftype == "L":
            ch = raw[:1].decode(_ENCODING).upper()
            return True if ch in "TY" else False if ch in "FN" else None
        if ftype == "D":
            text = raw.decode(_ENCODING).strip()
            return text or None
        if ftype == "I":
            return struct.unpack("<i", raw[:4])[0] if len(raw) >= 4 else None
        # Tipos que estos archivos no usan (T, Y, B, F, G, P...). Se devuelve el texto
        # crudo en lugar de fallar, para que un archivo distinto no rompa el laboratorio.
        return raw.decode(_ENCODING).strip()

    @staticmethod
    def _decode_varchar(raw: bytes) -> str:
        """Decodifica un Varchar de Visual FoxPro.

        El ultimo byte del campo guarda la longitud util. En estos archivos esa
        longitud a veces sobra (incluye el relleno que tenia el valor original), asi
        que se recorta a ``longitud`` y despues se quitan los espacios: el resultado es
        correcto en ambos casos. Si el byte no es una longitud plausible se descarta
        solo ese byte.
        """
        if not raw:
            return ""
        declared = raw[-1]
        if 0 < declared < len(raw):
            return raw[:declared].decode(_ENCODING).strip()
        return raw[:-1].decode(_ENCODING).strip()

    @staticmethod
    def _decode_number(raw: bytes, decimals: int) -> float | int | None:
        """Los numericos DBF son texto alineado a la derecha; vacio significa nulo."""
        text = raw.decode(_ENCODING).strip()
        if not text or text in ("-", "."):
            return None
        try:
            value = Decimal(text)
        except InvalidOperation:
            return None
        # Decimal primero y luego float: evita que '4873.9500' se vuelva 4873.949999.
        return float(value) if decimals else int(value)


def read_dbf(path: str | Path) -> list[dict[str, Any]]:
    """Atajo: devuelve todos los registros vivos de una tabla."""
    return list(DBFTable(path).records())
