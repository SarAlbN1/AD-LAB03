"""Objetos de dominio del laboratorio: line-item y factura (punto 2 del enunciado).

Los archivos DBF traen **una fila por line-item**, no una por factura (el enunciado
describe otro formato). La factura se reconstruye agrupando las filas que comparten
caja, tipo de documento y numero dentro del mismo archivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

# Tolerancia al comparar la suma de subtotales contra el total de la factura: el DBF
# redondea TOTAL al peso, asi que un descuadre de hasta 1 peso por line-item es
# aritmetica de redondeo, no un error de datos.
TOLERANCIA_PESOS = 1.0


@dataclass(slots=True)
class LineItem:
    """Un renglon de factura: que producto, cuanto, a que precio.

    Campos que pide el enunciado: nombre del producto, cantidad de items vendidos,
    precio por unidad y valor del subtotal de la compra.
    """

    producto: str
    codigo: str
    cantidad: float
    precio_unitario: float
    subtotal: float
    iva: float = 0.0
    descuento: float = 0.0
    tasa_iva: float = 0.0
    unidad: str = ""

    @property
    def subtotal_calculado(self) -> float:
        """Lo que deberia valer el renglon segun cantidad y precio."""
        return self.cantidad * self.precio_unitario - self.descuento

    @property
    def descuadre(self) -> float:
        return abs(self.subtotal - self.subtotal_calculado)

    @property
    def cuadra(self) -> bool:
        return self.descuadre <= TOLERANCIA_PESOS


@dataclass(slots=True)
class Factura:
    """Una factura de venta con sus renglones.

    Campos que pide el enunciado: numero o consecutivo, fecha o marca de tiempo,
    total, total de impuesto y cantidad total de items.
    """

    numero: str
    timestamp: datetime
    items: list[LineItem] = field(default_factory=list)
    # Contexto que traen los datos y que el analisis aprovecha.
    consecutivo: int = 0
    caja: str = ""
    tipo_doc: str = ""
    archivo: str = ""
    #: True cuando `timestamp` viene de la hora estimada (proxy ordinal) y no de un
    #: campo real del DBF. En estos archivos es siempre True: ver src/parse.py.
    hora_estimada: bool = True

    # ------------------------------------------------------------------ agregados

    @property
    def fecha(self) -> date:
        return self.timestamp.date()

    @property
    def hora(self) -> int:
        return self.timestamp.hour

    @property
    def total(self) -> float:
        """Total de la factura: suma de los subtotales de sus renglones."""
        return sum(i.subtotal for i in self.items)

    @property
    def total_impuesto(self) -> float:
        return sum(i.iva for i in self.items)

    @property
    def total_descuento(self) -> float:
        return sum(i.descuento for i in self.items)

    @property
    def total_items(self) -> float:
        """Cantidad total de unidades vendidas en la factura."""
        return sum(i.cantidad for i in self.items)

    @property
    def n_lineas(self) -> int:
        """Numero de renglones, que es el tamano de la canasta para reglas."""
        return len(self.items)

    # -------------------------------------------------------- canasta para reglas

    @property
    def canasta(self) -> list[str]:
        """Productos distintos de la factura.

        Las reglas de asociacion trabajan sobre presencia/ausencia, asi que un producto
        repetido en dos renglones de la misma factura cuenta una sola vez.
        """
        vistos: dict[str, None] = {}
        for item in self.items:
            vistos.setdefault(item.producto, None)
        return list(vistos)

    @property
    def es_unitaria(self) -> bool:
        """Factura de un solo producto: no aporta ninguna regla de asociacion."""
        return len(self.canasta) == 1

    @property
    def cuadra(self) -> bool:
        return all(i.cuadra for i in self.items)


def canastas(facturas: Iterable[Factura], minimo: int = 1) -> list[list[str]]:
    """Lista de canastas lista para `TransactionEncoder` de mlxtend.

    `minimo=2` descarta las facturas de un solo producto. No se usa por defecto: esas
    facturas cuentan en el denominador del soporte y quitarlas inflaria las metricas.
    """
    salida = []
    for f in facturas:
        c = f.canasta
        if len(c) >= minimo:
            salida.append(c)
    return salida
