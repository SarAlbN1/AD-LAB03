# Plan de desarrollo — ETL + comparativa de día de la semana

Complemento de `PLAN.md`. Cubre dos cambios: formalizar el procesamiento como **ETL
por capas** y agregar la **dimensión comparativa de día de la semana**.

## Estado actual

Ya escrito y con imagen Docker construida (`lab03-reglas-asociacion:1.0`):

| Archivo | Qué hace |
|---|---|
| `src/dbf_reader.py` | Lector DBF Visual FoxPro `0x32`, resuelve el Varchar de `DESCRI` |
| `src/models.py` | `LineItem`, `Factura`, `canastas()` |
| `src/parse.py` | DBF → facturas → DataFrames + caché parquet, hora-proxy |
| `src/eda.py` | Punto 3 del enunciado, 9 tablas y 4 figuras |
| `src/rules.py` | Reglas + las 8 preguntas del taller |
| `src/bench.py` | Grilla 20×2×3 con timeout y firma de equivalencia |
| `tests/test_dbf_reader.py` | 15 pruebas, incl. contraste contra `dbfread` |
| `Makefile`, `docker/` | Orquestación |

Nada se ha ejecutado todavía. `dia_semana` existe, pero flojo: solo a nivel factura,
en inglés (`strftime('%A')`), traducido después en `eda.py`. No sirve para comparar.

## Hallazgo que condiciona todo el diseño

Los 13 archivos **no son 13 días sueltos**: son dos semanas desbalanceadas.

| Semana | Días presentes | Estado |
|---|---|---|
| 09–15 marzo 2024 | sáb 09, dom 10, lun 11, mar 12, mié 13, jue 14, vie 15 | **completa, 7/7** |
| 24 marzo 2024 | dom 24 | aislado — Domingo de Ramos, el archivo más grande (19 MB) |
| 01–06 abril 2024 | lun 01, mar 02, mié 03, jue 04, sáb 06 | incompleta: falta vie 05 (DBF perdido) y dom 07 |

Además el volumen por archivo cae fuerte entre semanas (lunes: 8.7 MB en marzo vs
0.8 MB en abril), y abril 01–06 es la semana **posterior a Semana Santa** (24–31 marzo).

**Consecuencia:** un `groupby('dia_semana').sum()` sobre los 13 archivos confunde el
efecto del día con el efecto de la semana, y castiga al viernes, que solo tiene una
observación. El diseño correcto:

1. **Panel principal**: semana 09–15 marzo. Siete días, una observación cada uno,
   misma semana → la comparación entre días es limpia.
2. **Replicación**: semana 01–06 abril, con los días que tiene. Si el patrón se
   repite, el efecto del día es real y no de esa semana.
3. **24 marzo**: se excluye del panel y se analiza aparte como caso atípico
   (Domingo de Ramos), que es un hallazgo, no un estorbo.
4. Toda métrica comparativa se expresa **por día observado** (promedio), nunca como
   suma acumulada entre semanas.

## Parte 1 — ETL por capas

### Por qué un ETL y no seguir con `parse.cargar()`

- Reparsear los 13 DBF cuesta ~1 min y el notebook, el EDA, las reglas y el benchmark
  lo necesitan; hoy el caché es un parquet monolítico, todo o nada.
- La dimensión de calendario tiene que estar en **dos niveles** (renglón y factura) y
  en las tablas comparativas. Derivarla tres veces es donde aparecen las
  inconsistencias.
- Las validaciones (descuadres, fechas vs carpeta, códigos con varias descripciones)
  hoy están repartidas; en un ETL son un paso con criterio de falla explícito.
- El reporte necesita una sección "Datos y método" defendible: capas, reglas de
  transformación y controles de calidad.

**Sin Airflow ni dbt.** Para 13 archivos y un estudiante son infraestructura que no se
paga: etapas en Python puro, idempotentes, encadenadas por `make`. Es un ETL real
(extract → transform → load, con contratos y linaje), no un framework.

### Capas

```
archivos/*.DBF
   │  extract   (sin lógica de negocio: bytes -> tabla, + manifiesto con sha256)
   ▼
outputs/etl/bronze/<archivo>.parquet        1 fila por registro DBF, tal cual
   │  transform (normalizar, canonizar, calendario, hora-proxy, validar)
   ▼
outputs/etl/silver/line_items.parquet       1 fila por renglón, limpio y tipado
outputs/etl/silver/facturas.parquet         1 fila por factura
   │  load     (modelo analítico)
   ▼
outputs/etl/gold/dim_calendario.parquet     1 fila por fecha  <-- día de la semana
outputs/etl/gold/dim_producto.parquet       1 fila por producto canónico
outputs/etl/gold/fact_line_items.parquet    hechos con claves a las dimensiones
outputs/etl/gold/fact_facturas.parquet
outputs/etl/gold/onehot.parquet             matriz para mlxtend
outputs/etl/lab03.sqlite                    (opcional) las gold en SQL
```

### Archivos nuevos

| Archivo | Contenido |
|---|---|
| `src/etl/__init__.py` | — |
| `src/etl/extract.py` | `extraer(ruta) -> parquet bronze`; manifiesto con `archivo, sha256, n_registros, version_dbf, fecha, bytes`. Salta el trabajo si el sha256 no cambió. |
| `src/etl/calendario.py` | Construye `dim_calendario` y es la **única** fuente de `dia_semana`. |
| `src/etl/transform.py` | Bronze → silver: normalización de producto, catálogo por `CODIGO`, tipado, hora-proxy, unión con el calendario. |
| `src/etl/validate.py` | Contratos de datos; falla duro o avisa según severidad. |
| `src/etl/load.py` | Silver → gold (dimensiones + hechos + one-hot + SQLite opcional). |
| `src/etl/pipeline.py` | `python -m src.etl.pipeline [--force] [--solo-extract]`, log a `outputs/etl/etl_log.csv`. |
| `tests/test_etl.py` | Pruebas de cada capa y de los contratos. |

`src/parse.py` queda como **fachada**: `cargar()` pasa a leer las tablas gold, para no
reescribir `eda.py`, `rules.py` ni `bench.py`. Se conserva `cargar_facturas()` porque
construye los objetos `LineItem`/`Factura` que el enunciado pide explícitamente, y los
tests ya lo cubren.

### `dim_calendario` — la dimensión que resuelve la comparativa

Una fila por fecha observada (13 fechas). Columnas:

| Columna | Ejemplo | Para qué |
|---|---|---|
| `fecha` | 2024-03-11 | clave |
| `archivo` | fc11032024.DBF | linaje |
| `dia_semana` | lunes | categórica **ordenada** lunes→domingo (en español, desde el ETL) |
| `dia_num` | 0 | orden estable para gráficas |
| `es_fin_semana` | False | sáb+dom |
| `tipo_dia` | hábil / fin de semana | agrupación de 2 niveles |
| `semana_iso` | 2024-W11 | separa el efecto semana del efecto día |
| `panel` | completa / parcial / atipico | 09–15 mar / 01–06 abr / 24 mar |
| `es_atipico` | True para 2024-03-24 | Domingo de Ramos |
| `nota` | "post Semana Santa" | contexto para el reporte |
| `n_dias_observados` | 1 | denominador de los promedios |

Se propaga `fecha`, `dia_semana`, `es_fin_semana`, `tipo_dia`, `semana_iso`, `panel`
tanto a `fact_line_items` como a `fact_facturas`. Una sola definición, dos niveles.

### Contratos de validación (`validate.py`)

| Control | Severidad |
|---|---|
| Fecha del nombre coincide con la carpeta del día (ddmmyyyy bien leído) | **falla** |
| `n_registros` del bronze == `record_count` del header DBF | **falla** |
| Ningún `DESCRI` con bytes de control (el Varchar quedó bien recortado) | **falla** |
| Suma de renglones == total de factura, ±$1 | avisa + métrica |
| `CODIGO` con más de una `DESCRI` | avisa + queda en `dim_producto` |
| Renglones sin producto identificable | avisa + cuenta |
| Facturas con `total <= 0` | avisa + cuenta |
| Hora-proxy dentro de la jornada y monótona por caja | **falla** |
| Cobertura: días esperados por semana vs presentes | avisa (deja constancia del viernes faltante) |

Resultado a `outputs/tables/00_calidad_datos.csv` → va al reporte como control de
calidad, que el enunciado no pide y suma.

## Parte 2 — Comparativa de día de la semana

Módulo nuevo `src/dow.py` (`python -m src.dow`), que consume las tablas gold.

### Tablas

**`15_comparativa_dia_semana.csv`** — una fila por día, solo panel completo
(09–15 marzo), métricas por día observado:

| Columna | Qué es |
|---|---|
| `dia_semana`, `es_fin_semana`, `tipo_dia` | de `dim_calendario` |
| `facturas`, `unidades`, `valor`, `ticket_promedio`, `canasta_promedio`, `pct_unitarias` | nivel |
| `indice_facturas`, `indice_valor`, `indice_ticket`, `indice_canasta` | **la columna comparativa**: valor del día / promedio de los 7 días. 1.00 = igual al promedio, 1.20 = 20% arriba |
| `delta_pct_vs_promedio` | lo mismo en puntos porcentuales, para el texto |
| `rank` | posición del día en esa métrica |

El índice es la forma correcta de comparar: adimensional, comparable entre métricas de
escalas distintas (pesos vs conteos) y legible en una tabla de dos columnas.

**`16_comparativa_dia_replicacion.csv`** — el mismo índice calculado en la semana de
abril, al lado del de marzo, con `coincide_signo` → responde si el patrón de día es
estable o fue de esa semana.

**`17_producto_por_dia.csv`** — producto × día: soporte del día, soporte global e
`indice_dia = soporte_dia / soporte_global`. Con filtro de soporte absoluto ≥ 30 para
no leer ruido. Saca los productos con sesgo de día (p. ej. algo que se vende el
domingo y no el martes), que es material directo para la hipótesis y para la
pregunta 8 (comportamiento del consumidor).

**`18_reglas_por_tipo_dia.csv`** — las reglas del top general, con su `support`,
`confidence` y `lift` recalculados en hábil vs fin de semana, más `delta_lift`.
Responde si una regla es estructural o es de fin de semana. Esto va a la pregunta 7.

**`19_atipico_domingo_ramos.csv`** — 24 marzo contra el domingo normal (10 marzo):
ticket, canasta, top 10 y qué productos aparecen solo ese día.

### Figuras

- `15_indice_por_dia.png`: barras del índice por día, línea de referencia en 1.00,
  barras de fin de semana en otro color. Una sola figura cuenta toda la comparativa.
- `16_heatmap_dia_hora.png`: día × hora estimada, intensidad = facturas. Cruza las dos
  dimensiones temporales y deja ver si el pico horario se mueve el fin de semana.
  Rotulada "hora estimada (proxy ordinal)".

### Cambios en los módulos existentes

- `src/eda.py`: `por_dia_semana()` pasa a usar `dim_calendario` (quita `DIAS_ES` y
  `ORDEN_DIAS`, que quedan sin razón de existir) y a reportar promedios por día
  observado en vez de sumas.
- `src/rules.py`: `minar()` acepta un filtro de calendario (`panel`, `tipo_dia`) para
  poder minar reglas por segmento sin duplicar código.
- `src/bench.py`: sin cambios; el benchmark usa el one-hot completo a propósito —
  medir rendimiento quiere la matriz más grande posible.

## Orden de trabajo

1. `src/etl/calendario.py` + `dim_calendario` (lo que desbloquea el resto).
2. `src/etl/extract.py` + manifiesto; correr y verificar los 13 bronze.
3. `src/etl/transform.py` + `validate.py` → silver + tabla de calidad.
4. `src/etl/load.py` + `pipeline.py` → gold; `parse.cargar()` apuntando a gold.
5. `tests/test_etl.py` verde junto con los tests ya existentes.
6. `src/dow.py` → las 5 tablas y 2 figuras comparativas.
7. Ajustar `eda.py` y `rules.py`; correr `make d-eda`, `make d-rules`.
8. `make d-bench` (el paso largo, se deja corriendo).
9. Notebook + reporte: la comparativa de día entra como una subsección de resultados
   y alimenta la hipótesis.

## Metas del Makefile

```
make d-etl          # pipeline completo (extract + transform + load)
make d-etl-force    # ignora el caché por sha256 y reprocesa
make d-dow          # comparativa de día de la semana
```
`make all` pasa a ser: `test → etl → eda → dow → rules → bench → notebook → report`.

## Verificación

- `make d-etl` deja 13 parquet en `bronze/`, manifiesto con 13 sha256 distintos, y
  `silver/line_items.parquet` con **147.413** filas y `facturas.parquet` con **68.809**
  (las cifras ya medidas: si no cuadran, el ETL rompió algo).
- `dim_calendario` tiene exactamente 13 filas, 7 valores de `dia_semana`, `viernes`
  con `n_dias_observados = 1` y `2024-03-24` con `es_atipico = True`.
- Segunda corrida de `make d-etl` sin cambios: no reprocesa nada (caché por sha256) y
  termina en segundos.
- `make d-etl-force` reproduce byte a byte los mismos conteos.
- `fact_line_items` y `fact_facturas` dan el mismo `dia_semana` para la misma factura
  (prueba de consistencia entre niveles).
- `make d-dow` → los 5 CSV y 2 PNG; los `indice_*` del panel completo promedian 1.00
  (prueba aritmética de que el índice está bien construido).
- `make d-test` verde: tests del DBF, del ETL y de los contratos.
