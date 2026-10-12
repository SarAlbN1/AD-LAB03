# Laboratorio 03 — Reglas de asociación

Análisis de reglas de asociación sobre las facturas de una tienda de conveniencia, y
comparación de rendimiento entre **apriori** y **fp-growth** (mlxtend).

Entregables: `notebooks/lab03.ipynb` y `report/report.pdf`.

## Reproducir

Requiere Docker. Todo corre dentro del contenedor; no hace falta instalar Python ni
TeX en la máquina.

```bash
make build     # imagen con Python 3.12, mlxtend, pandas y TeX Live
make d-all     # tests -> ETL -> EDA -> día de semana -> reglas -> benchmark -> notebook -> PDF
```

Por partes:

| Comando | Qué hace | Duración aprox. |
|---|---|---|
| `make d-test` | 45 pruebas (lector DBF, ETL, contratos, índices) | segundos |
| `make d-etl` | DBF → bronze → silver → gold | ~6 s |
| `make d-eda` | Análisis descriptivo (punto 3) | ~1 min |
| `make d-dow` | Comparativa de día de la semana | ~3 min |
| `make d-rules` | Reglas y las 8 preguntas del taller | ~8 min |
| `make d-bench` | 120 corridas de benchmark (punto 5) | decenas de minutos |
| `make d-report` | Tablas LaTeX + PDF de 2 páginas | ~20 s |
| `make jupyter` | JupyterLab en http://localhost:8888 | — |

`make d-etl` cachea por SHA-256 de cada DBF: la segunda corrida no reprocesa nada.
`make d-etl-force` ignora el caché y reproduce las tablas byte a byte.

### Por qué Docker y no Podman

En macOS ambos levantan una VM, así que la ventaja de Podman de no tener demonio no
aplica. Lo que el laboratorio necesita —un puerto de Jupyter, un bind-mount de solo
lectura y TeX Live— es OCI plano. Docker Desktop monta los DBF más rápido (VirtioFS) y
no remapea UIDs al escribir `outputs/` de vuelta al host. El `Dockerfile` es estándar:
`podman build` lo corre sin cambios si hiciera falta.

## Estructura

```
archivos/              13 DBF de Visual FoxPro (solo lectura, nunca se modifican)
src/
  dbf_reader.py        lector DBF VFP 0x32 (resuelve el Varchar de DESCRI)
  models.py            LineItem y Factura (punto 2 del enunciado)
  parse.py             fachada de carga + modelo de objetos
  etl/                 pipeline bronze -> silver -> gold
    calendario.py      dim_calendario: única fuente del día de la semana
    extract.py         DBF -> parquet crudo + manifiesto con SHA-256
    transform.py       reglas de negocio: canonización, factura, hora estimada
    validate.py        18 contratos de datos (falla duro / avisa)
    load.py            dimensiones, hechos y matriz one-hot
    pipeline.py        orquestación idempotente
  eda.py               análisis descriptivo (punto 3)
  dow.py               comparativa de día de la semana
  rules.py             reglas y respuestas a las 8 preguntas
  bench.py             apriori vs fp-growth (punto 5)
  report_tables.py     CSV -> fragmentos LaTeX
outputs/
  etl/{bronze,silver,gold}/   capas del ETL
  tables/*.csv                todas las tablas del análisis
  figs/*.png                  figuras
report/report.tex      fuente del PDF (2 páginas, 12pt, Times, 2 columnas)
tests/                 45 pruebas
```

## Decisiones que conviene conocer antes de leer los resultados

**El formato no es el que describe el enunciado.** Los archivos son DBF de Visual
FoxPro con una línea por *line-item*, no texto con una línea por factura. La factura se
reconstruye agrupando por archivo, caja, tipo de documento y consecutivo.

**`DESCRI` es un `Varchar` de FoxPro**: su último byte guarda la longitud del texto. Un
lector que no lo interprete devuelve `'LUCKY DAIQUIRI X 10 … B'`, donde esa `B` es el
byte 66. `dbfread` no lo maneja; por eso el lector es propio, y
`tests/test_dbf_reader.py` deja constancia de la diferencia.

**No hay hora en los datos.** `FECVEN` está vacío en los 147.413 registros y `MESANRO`
es 0. La fecha sale del nombre del archivo; la hora se **estima** repartiendo las
facturas de cada caja, en su orden real de facturación, sobre una jornada supuesta de
07:00 a 22:00. Sirve para la posición dentro de la jornada, no para horas absolutas, y
**el número de facturas por hora es constante por construcción**: no es un hallazgo.
Si aparece el maestro de facturas con hora real, el análisis se recalcula sin cambiar
código, porque la hora entra como columna.

**Los 13 archivos no son 13 días comparables.** Son dos tramos desiguales:

| Tramo | Días | Estado |
|---|---|---|
| 09–15 marzo | 7 consecutivos | **completo**: cada día de la semana una vez |
| 24 marzo | 1 | atípico: Domingo de Ramos |
| 01–06 abril | 5 | incompleto: falta el viernes 05 (DBF perdido) y el domingo 07 |

La comparativa de día usa solo el tramo completo. Sumar los 13 archivos confundiría el
efecto del día con el de la semana y castigaría al viernes, que tiene una sola
observación.

**Se incluyen los cinco tipos de documento** (`FA`, `CO`, `AS`, `AE`, `DC`). Los que no
son ventas son el 2 % de los documentos pero el 55 % del valor: hay un documento de
inventario de cerveza por $13.954.166 repetido idéntico en cinco fechas. Por eso el
informe cita ticket **mediano** y no promedio; `outputs/tables/00d_sensibilidad_tipo_documento.csv`
cuantifica el efecto sobre cada métrica.

## Datos de referencia

Si el ETL se modifica, estas cifras deben seguir saliendo (los tests las verifican):

| | |
|---|---|
| Registros en bronze | 147.413 |
| Renglones en silver | 147.411 (2 sin producto identificable) |
| Facturas | 68.807 |
| Productos canónicos | 1.265 (de 1.280 descripciones) |
| Renglones por factura | media 2,14 · máx 49 |
| Facturas de un solo producto | 28.203 (41,0 %) |
