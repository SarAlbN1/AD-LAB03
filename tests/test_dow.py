"""Pruebas de la comparativa de dia de la semana.

La prueba que mas importa es `test_indices_promedian_uno`: la columna comparativa es
un indice relativo al promedio, asi que su media tiene que ser exactamente 1.00. Es la
comprobacion aritmetica de que el indice esta bien construido.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.dow import METRICAS, agregar_indices, diagnostico_replicacion
from src.etl import paths


class TestIndice(unittest.TestCase):
    def setUp(self) -> None:
        self.tabla = pd.DataFrame(
            {
                "dia_semana": ["lunes", "martes", "miercoles"],
                "facturas": [100.0, 200.0, 300.0],
                "valor": [1000.0, 2000.0, 3000.0],
                "ticket_promedio": [10.0, 10.0, 10.0],
                "canasta_promedio": [2.0, 2.0, 2.0],
                "unidades": [50.0, 100.0, 150.0],
            }
        )

    def test_indices_promedian_uno(self) -> None:
        salida = agregar_indices(self.tabla)
        for metrica in METRICAS:
            promedio = salida[f"indice_{metrica}"].mean()
            self.assertAlmostEqual(promedio, 1.0, places=2,
                                   msg=f"indice_{metrica} promedia {promedio}")

    def test_indice_es_valor_sobre_promedio(self) -> None:
        salida = agregar_indices(self.tabla)
        # facturas: 100, 200, 300 -> promedio 200 -> indices 0.5, 1.0, 1.5
        self.assertEqual(list(salida["indice_facturas"]), [0.5, 1.0, 1.5])
        self.assertEqual(list(salida["delta_pct_facturas"]), [-50.0, 0.0, 50.0])

    def test_metrica_constante_da_indice_uno(self) -> None:
        """Un dia que se comporta como el promedio tiene indice 1.00 exacto."""
        salida = agregar_indices(self.tabla)
        self.assertTrue((salida["indice_ticket_promedio"] == 1.0).all())

    def test_rank(self) -> None:
        salida = agregar_indices(self.tabla)
        self.assertEqual(list(salida["rank_facturas"]), [3, 2, 1])


class TestDiagnosticoReplicacion(unittest.TestCase):
    def _diagnostico(self, rep: pd.DataFrame) -> dict:
        salida = diagnostico_replicacion(rep)
        return dict(zip(salida["clave"], salida["valor"]))

    def test_detecta_caida_monotona(self) -> None:
        """Una caida que crece con el dia delata archivos incompletos, no demanda."""
        rep = pd.DataFrame(
            {
                "dia_semana": ["lunes", "martes", "miercoles"],
                "indice_marzo": [0.9, 1.0, 1.1],
                "indice_abril": [0.3, 0.6, 2.1],
                "mismo_lado_del_promedio": [True, False, True],
                "caida_volumen_pct": [-90.0, -80.0, -70.0],
            }
        )
        d = self._diagnostico(rep)
        self.assertTrue(d["caida monotona por dia"])
        self.assertFalse(d["abril sirve de replicacion"])

    def test_acepta_replicacion_consistente(self) -> None:
        rep = pd.DataFrame(
            {
                "dia_semana": ["lunes", "martes", "miercoles"],
                "indice_marzo": [0.9, 1.0, 1.1],
                "indice_abril": [0.88, 1.02, 1.10],
                "mismo_lado_del_promedio": [True, True, True],
                "caida_volumen_pct": [-20.0, -25.0, -18.0],
            }
        )
        d = self._diagnostico(rep)
        self.assertFalse(d["caida monotona por dia"])
        self.assertTrue(d["abril sirve de replicacion"])


@unittest.skipUnless((paths.TABLAS / "15_comparativa_dia_semana.csv").exists(),
                     "requiere haber corrido la comparativa (make dow)")
class TestSalidas(unittest.TestCase):
    def test_comparativa_tiene_los_siete_dias(self) -> None:
        tabla = pd.read_csv(paths.TABLAS / "15_comparativa_dia_semana.csv")
        self.assertEqual(len(tabla), 7)
        self.assertAlmostEqual(tabla["indice_facturas"].mean(), 1.0, places=2)
        # Cada dia del panel completo se observa una sola vez: es lo que lo hace
        # comparable.
        self.assertTrue((tabla["fechas"] == 1).all())

    def test_producto_por_dia_tiene_un_indice_por_dia(self) -> None:
        tabla = pd.read_csv(paths.TABLAS / "17_producto_por_dia.csv")
        for dia in ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado",
                    "domingo"):
            self.assertIn(f"indice_{dia}", tabla.columns)
        self.assertTrue((tabla["amplitud"] >= 1).all())

    def test_reglas_por_tipo_dia_sin_permutaciones(self) -> None:
        """Consecuente de un solo producto y un itemset por fila."""
        tabla = pd.read_csv(paths.TABLAS / "18_reglas_por_tipo_dia.csv")
        self.assertFalse(tabla["consecuente"].str.contains(r"\+").any())
        itemsets = [
            frozenset(a.split(" + ")) | frozenset(c.split(" + "))
            for a, c in zip(tabla["antecedente"], tabla["consecuente"])
        ]
        self.assertEqual(len(itemsets), len(set(itemsets)))


if __name__ == "__main__":
    unittest.main()
