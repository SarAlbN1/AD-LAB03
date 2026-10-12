"""Pruebas del lector DBF y del armado de facturas.

La prueba central es `test_varchar_sin_byte_de_longitud`: documenta con un caso real
por que este laboratorio no usa `dbfread`.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.dbf_reader import DBFTable
from src.parse import (
    archivos_dbf,
    cargar_facturas,
    construir_catalogo,
    fecha_desde_nombre,
    normalizar_producto,
)

ARCHIVO_CHICO = "fc01042024.DBF"  # 1.168 registros: el mas pequeno del conjunto


def _ruta_chica() -> Path:
    for ruta in archivos_dbf():
        if ruta.name == ARCHIVO_CHICO:
            return ruta
    raise unittest.SkipTest(f"no se encontro {ARCHIVO_CHICO}")


class TestDBFTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ruta = _ruta_chica()
        cls.tabla = DBFTable(cls.ruta)

    def test_header(self) -> None:
        self.assertEqual(self.tabla.version, 0x32)  # Visual FoxPro
        self.assertEqual(self.tabla.record_count, 1168)
        self.assertEqual(len(self.tabla.fields), 51)

    def test_campos_esperados(self) -> None:
        for campo in ("FACNRO", "CODTRA", "CODIGO", "DESCRI", "CANT", "PRECIO",
                      "TOTAL", "IVA", "CODCAJ"):
            self.assertIn(campo, self.tabla.field_names)

    def test_descri_es_varchar(self) -> None:
        self.assertEqual(self.tabla.field("DESCRI").type, "V")
        self.assertEqual(self.tabla.field("DESCRI").length, 254)

    def test_varchar_sin_byte_de_longitud(self) -> None:
        """El ultimo byte del Varchar es la longitud, no texto.

        El primer registro trae 'LUCKY DAIQUIRI X 10' y un byte final 0x42 (66). Un
        lector que no interprete el tipo V deja ese 0x42 como una 'B' pegada al nombre.
        """
        primero = next(iter(self.tabla))
        self.assertEqual(primero["DESCRI"], "LUCKY DAIQUIRI X 10")
        self.assertFalse(primero["DESCRI"].endswith("B"))

        crudo = self.tabla.raw_record(0)
        campo = self.tabla.field("DESCRI")
        inicio = 1 + campo.offset
        bytes_campo = crudo[inicio : inicio + campo.length]
        self.assertEqual(bytes_campo[-1], 66)  # el byte de longitud
        ingenuo = bytes_campo.decode("latin-1").strip()
        self.assertTrue(ingenuo.endswith("B"), "el caso que motiva el lector propio")

    def test_numericos_con_decimales(self) -> None:
        primero = next(iter(self.tabla))
        self.assertAlmostEqual(primero["PRECIO"], 4873.95, places=4)
        self.assertAlmostEqual(primero["CANT"], 1.0, places=6)
        self.assertEqual(primero["FACNRO"], 61642)  # sin decimales -> int
        self.assertIsInstance(primero["FACNRO"], int)

    def test_ninguna_descripcion_con_caracteres_de_control(self) -> None:
        for fila in self.tabla:
            self.assertNotRegex(fila["DESCRI"], r"[\x00-\x1f]")

    def test_contraste_con_dbfread(self) -> None:
        """dbfread lee los mismos registros pero no recorta el Varchar."""
        try:
            from dbfread import DBF
        except ImportError:  # pragma: no cover
            self.skipTest("dbfread no instalado")

        externo = list(DBF(str(self.ruta), encoding="latin-1", load=True))
        propio = list(self.tabla.records(include_deleted=True))
        self.assertEqual(len(externo), len(propio))
        # Los numericos coinciden...
        self.assertEqual(externo[0]["FACNRO"], propio[0]["FACNRO"])
        # ...y la descripcion es donde difieren.
        self.assertTrue(str(externo[0]["DESCRI"]).strip().endswith("B"))
        self.assertEqual(propio[0]["DESCRI"], "LUCKY DAIQUIRI X 10")


class TestParseo(unittest.TestCase):
    def test_fecha_desde_nombre(self) -> None:
        self.assertEqual(str(fecha_desde_nombre("fc01042024.DBF")), "2024-04-01")
        self.assertEqual(str(fecha_desde_nombre("fc24032024.DBF")), "2024-03-24")

    def test_carpeta_coincide_con_dia_de_semana(self) -> None:
        """Valida la lectura ddmmyyyy: lunes/ debe caer en lunes, domingo/ en domingo."""
        dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
        for ruta in archivos_dbf():
            carpeta = ruta.parent.name
            if carpeta in dias:
                self.assertEqual(dias[fecha_desde_nombre(ruta).weekday()], carpeta,
                                 f"{ruta.name} no cae en {carpeta}")

    def test_normalizar_producto(self) -> None:
        self.assertEqual(normalizar_producto("  jugo   hit  500 ml "), "JUGO HIT 500 ML")
        self.assertEqual(normalizar_producto("PASTELES. "), "PASTELES")

    def test_catalogo_canoniza_codigos(self) -> None:
        filas = [
            {"CODIGO": "1", "DESCRI": "JUGO HIT 500 ML"},
            {"CODIGO": "1", "DESCRI": "JUGO HIT 500 ML"},
            {"CODIGO": "1", "DESCRI": "jugo  hit  500 ml"},
        ]
        catalogo = construir_catalogo(filas)
        self.assertEqual(catalogo["1"], "JUGO HIT 500 ML")


class TestFacturas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facturas = cargar_facturas([_ruta_chica()])

    def test_se_agruparon_lineas(self) -> None:
        total_lineas = sum(f.n_lineas for f in self.facturas)
        self.assertLess(len(self.facturas), total_lineas)
        self.assertGreater(len(self.facturas), 0)

    def test_canasta_sin_repetidos(self) -> None:
        for f in self.facturas:
            self.assertEqual(len(f.canasta), len(set(f.canasta)))
            self.assertLessEqual(len(f.canasta), f.n_lineas)

    def test_totales_coinciden_con_los_renglones(self) -> None:
        f = max(self.facturas, key=lambda x: x.n_lineas)
        self.assertAlmostEqual(f.total, sum(i.subtotal for i in f.items), places=2)
        self.assertAlmostEqual(f.total_impuesto, sum(i.iva for i in f.items), places=2)

    def test_hora_estimada_dentro_de_la_jornada(self) -> None:
        for f in self.facturas:
            self.assertGreaterEqual(f.hora, 7)
            self.assertLessEqual(f.hora, 22)
            self.assertTrue(f.hora_estimada)

    def test_hora_estimada_es_monotona_por_caja(self) -> None:
        """El proxy debe respetar el orden de facturacion dentro de cada caja."""
        por_caja: dict[str, list] = {}
        for f in self.facturas:
            por_caja.setdefault(f.caja, []).append(f)
        for caja, grupo in por_caja.items():
            tiempos = [f.timestamp for f in grupo]
            self.assertEqual(tiempos, sorted(tiempos), f"caja {caja} fuera de orden")


if __name__ == "__main__":
    unittest.main()
