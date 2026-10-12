"""Pruebas del ETL: calendario, capas y contratos.

Las pruebas del calendario no tocan disco: construyen la dimension con fechas
sinteticas, incluidas las tres formas que tiene este conjunto (semana completa, dia
aislado, semana con huecos).
"""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from src.etl import calendario, paths
from src.etl.validate import ErrorDeContrato, Reporte


def _pares(fechas: list[date]) -> list[tuple[date, str]]:
    return [(f, f"fc{f.day:02d}{f.month:02d}{f.year}.DBF") for f in fechas]


SEMANA_COMPLETA = [date(2024, 3, d) for d in range(9, 16)]       # sab 09 .. vie 15
AISLADO = [date(2024, 3, 24)]                                     # Domingo de Ramos
SEMANA_CON_HUECOS = [date(2024, 4, d) for d in (1, 2, 3, 4, 6)]   # falta vie 05


class TestCalendario(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dim = calendario.construir(
            _pares(SEMANA_COMPLETA + AISLADO + SEMANA_CON_HUECOS)
        )

    def test_una_fila_por_fecha(self) -> None:
        self.assertEqual(len(self.dim), 13)
        self.assertTrue(self.dim["fecha"].is_unique)

    def test_dia_en_espanol_y_ordenado(self) -> None:
        self.assertEqual(
            list(self.dim["dia_semana"].cat.categories),
            ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"],
        )
        self.assertTrue(self.dim["dia_semana"].cat.ordered)
        fila = self.dim[self.dim["fecha"] == date(2024, 3, 11)].iloc[0]
        self.assertEqual(fila["dia_semana"], "lunes")
        self.assertEqual(fila["dia_num"], 0)

    def test_fin_de_semana(self) -> None:
        sabado = self.dim[self.dim["fecha"] == date(2024, 3, 9)].iloc[0]
        self.assertTrue(sabado["es_fin_semana"])
        self.assertEqual(sabado["tipo_dia"], "fin de semana")
        lunes = self.dim[self.dim["fecha"] == date(2024, 3, 11)].iloc[0]
        self.assertFalse(lunes["es_fin_semana"])
        self.assertEqual(lunes["tipo_dia"], "habil")

    def test_panel_completo_son_los_siete_dias_de_marzo(self) -> None:
        completo = self.dim[self.dim["panel"] == calendario.PANEL_COMPLETO]
        self.assertEqual(len(completo), 7)
        self.assertEqual(set(completo["fecha"]), set(SEMANA_COMPLETA))
        # El panel completo tiene cada dia de la semana exactamente una vez: es lo que
        # permite comparar dias sin confundirlos con el efecto de la semana.
        self.assertEqual(sorted(completo["dia_num"]), list(range(7)))

    def test_dia_aislado_marcado_como_atipico(self) -> None:
        fila = self.dim[self.dim["fecha"] == date(2024, 3, 24)].iloc[0]
        self.assertTrue(fila["es_atipico"])
        self.assertEqual(fila["panel"], calendario.PANEL_ATIPICO)
        self.assertIn("Ramos", fila["nota"])

    def test_semana_con_huecos_es_parcial(self) -> None:
        parcial = self.dim[self.dim["panel"] == calendario.PANEL_PARCIAL]
        self.assertEqual(set(parcial["fecha"]), set(SEMANA_CON_HUECOS))
        # El sabado 06 queda solo por el viernes 05 que falta, pero no esta aislado
        # del resto del calendario, asi que no es atipico.
        sabado = self.dim[self.dim["fecha"] == date(2024, 4, 6)].iloc[0]
        self.assertFalse(sabado["es_atipico"])

    def test_n_dias_observados_es_el_denominador(self) -> None:
        """El viernes aparece una sola vez; el domingo, tres."""
        por_dia = self.dim.set_index("fecha")["n_dias_observados"]
        self.assertEqual(por_dia[date(2024, 3, 15)], 1)   # viernes: solo el 15 de marzo
        self.assertEqual(por_dia[date(2024, 3, 10)], 2)   # domingos: 10 y 24 de marzo

    def test_nota_de_semana_santa(self) -> None:
        abril = self.dim[self.dim["fecha"] == date(2024, 4, 1)].iloc[0]
        self.assertIn("Semana Santa", abril["nota"])

    def test_rechaza_fechas_duplicadas(self) -> None:
        with self.assertRaises(ValueError):
            calendario.construir(
                [(date(2024, 3, 9), "a.DBF"), (date(2024, 3, 9), "b.DBF")]
            )


class TestReporteDeContratos(unittest.TestCase):
    def test_avisar_no_detiene(self) -> None:
        rep = Reporte()
        rep.avisar("algo imperfecto", False, 42)
        self.assertEqual(rep.filas[0]["resultado"], "aviso")

    def test_exigir_detiene(self) -> None:
        rep = Reporte()
        with self.assertRaises(ErrorDeContrato):
            rep.exigir("algo roto", False, 0)

    def test_dataframe_con_columnas_esperadas(self) -> None:
        rep = Reporte()
        rep.avisar("x", True, 1)
        df = rep.a_dataframe()
        for col in ("control", "severidad", "resultado", "valor", "detalle"):
            self.assertIn(col, df.columns)


@unittest.skipUnless(paths.GOLD_FACT_LINE_ITEMS.exists(),
                     "requiere haber corrido el ETL (make etl)")
class TestCapasGold(unittest.TestCase):
    """Pruebas sobre las tablas ya materializadas."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.etl.load import leer_dim_calendario, leer_gold

        cls.items, cls.facturas = leer_gold()
        cls.dim_cal = leer_dim_calendario()

    def test_conteos_conocidos(self) -> None:
        """Cifras de referencia del conjunto. Si cambian, el ETL rompio algo.

        El bronze trae los 147.413 registros vivos de los 13 DBF. De ahi el silver
        descarta 2 renglones sin producto identificable (sin CODIGO y sin DESCRI, que
        no pueden entrar a una canasta), y con ellos se van las 2 facturas que solo
        tenian ese renglon.
        """
        manifiesto = pd.read_csv(paths.MANIFIESTO)
        self.assertEqual(int(manifiesto["registros_bronze"].sum()), 147_413)
        self.assertEqual(len(self.items), 147_411)
        self.assertEqual(len(self.facturas), 68_807)

    def test_productos_canonizados(self) -> None:
        """La canonizacion por CODIGO reduce 1.280 descripciones a 1.265 productos.

        Es el efecto buscado: las variantes de escritura del mismo codigo dejan de
        repartirse el soporte entre dos items distintos.
        """
        self.assertEqual(self.items["producto"].nunique(), 1_265)

    def test_calendario_de_trece_dias(self) -> None:
        self.assertEqual(len(self.dim_cal), 13)
        self.assertEqual(self.dim_cal["dia_semana"].nunique(), 7)
        viernes = self.dim_cal[self.dim_cal["dia_semana"] == "viernes"]
        self.assertEqual(int(viernes["n_dias_observados"].iloc[0]), 1,
                         "falta el DBF del viernes 05 de abril")

    def test_dia_semana_en_los_dos_niveles(self) -> None:
        for df in (self.items, self.facturas):
            self.assertIn("dia_semana", df.columns)
            self.assertFalse(df["dia_semana"].isna().any())

    def test_dia_semana_consistente_entre_niveles(self) -> None:
        cruce = (
            self.items[["factura", "dia_semana"]].drop_duplicates()
            .merge(self.facturas[["factura", "dia_semana"]], on="factura",
                   suffixes=("_i", "_f"))
        )
        self.assertTrue(
            (cruce["dia_semana_i"].astype(str) == cruce["dia_semana_f"].astype(str)).all()
        )

    def test_hora_estimada_en_jornada(self) -> None:
        self.assertTrue(self.facturas["hora"].between(7, 22).all())
        self.assertTrue(self.facturas["hora_estimada"].all())

    def test_manifiesto_con_trece_huellas(self) -> None:
        manifiesto = pd.read_csv(paths.MANIFIESTO)
        self.assertEqual(len(manifiesto), 13)
        self.assertEqual(manifiesto["sha256"].nunique(), 13)

    def test_onehot_coincide_con_facturas(self) -> None:
        onehot = pd.read_parquet(paths.GOLD_ONEHOT)
        self.assertEqual(len(onehot), self.items["factura"].nunique())
        self.assertEqual(onehot.shape[1], self.items["producto"].nunique())


if __name__ == "__main__":
    unittest.main()
