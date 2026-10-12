# Laboratorio 03 — Reglas de asociación (apriori vs fp-growth)

## Context

`lab03_reglas_asociacion_2026-10-10.md` pide un laboratorio de reglas de asociación
sobre facturas de retail, con dos entregables (notebook `.ipynb` + PDF de 2 páginas,
Times New Roman 12pt, 2 columnas) para el **viernes 16 de octubre, 23:59**. La petición
es hacerlo de la mejor manera posible, incluyendo lo opcional, y en contenedor.

El proyecto hoy solo tiene el enunciado y `archivos/` con los datos crudos. No hay código.

**Lo que ya se midió sobre los datos reales** (no son supuestos; medido con un parser
`struct` sobre los 13 archivos):

| Hecho | Valor |
|---|---|
| Archivos | 13 `.DBF` Visual FoxPro ver `0x32`, esquema idéntico (51 campos, `rlen=713`) |
| Line-items (sin borrados) | **147,413** |
| Facturas (`archivo+CODCAJ+CODTRA+FACNRO`) | **68,809** |
| Ítems por factura | media **2.14**, máx **49**, 27,470 (40%) con un solo ítem |
| Productos distintos (`DESCRI`) | **1,280** |
| `CODIGO` con más de una `DESCRI` | 23 de 1,275 |
| `CODTRA` | FA 138,546 · CO 3,903 · AS 2,824 · AE 2,122 · DC 16 · vacío 2 |
| Cantidades negativas | 0 |

**Cinco discrepancias entre enunciado y datos, que este plan resuelve:**

1. **No hay campo de fecha-hora.** `FECVEN` vacío en los 147,413 registros, `MESANRO=0`.
   Los puntos 3.a, 3.b y 3.d piden análisis horario → se resuelve con hora-proxy
   declarada (ver Fase 4).
2. **`viernes/fc05042024` solo tiene el `.CDX`** (índice, 27 KB); falta su `.DBF`.
   Ese día queda incompleto; se documenta, no se rellena.
3. El enunciado describe **texto plano "una línea por factura"**; la realidad es DBF con
   **una línea por line-item** → hay que *agrupar* para armar la factura, no *dividir*.
4. **`DESCRI` es `Varchar` VFP (`V`, 254)**: el último byte del campo guarda la longitud.
   Sin recortarlo aparece basura (`'LUCKY DAIQUIRI X 10 … B'`). `dbfread` no lo maneja.
5. **`xlmtend` del enunciado es `mlxtend`.**

Consecuencia técnica clave: canasta media **2.14** ítems → los soportes útiles están en
1e-4…1e-2. A esos soportes apriori explota combinatoriamente, que es exactamente lo que
hace informativa la comparación contra fp-growth — pero exige timeout por corrida.

## Decisiones confirmadas con el usuario

- **Alcance:** los 13 archivos (7 días de la semana), no un solo día.
- **Hora:** proxy ordinal + limitación declarada en el reporte.
- **`CODTRA`:** **se incluyen todos** los tipos de documento. Se añade un anexo de
  sensibilidad "solo FA" (es una línea de filtro) para mostrar que las reglas
  principales no dependen de esa decisión.

## Runtime: Docker (no Podman) — argumento

**Docker.** Razonado para este caso, no por costumbre:

- Docker CLI **29.7.2 ya está instalado** (`/usr/local/bin/docker`), el demonio está
  apagado → basta abrir Docker Desktop. Podman exigiría instalar y
  `podman machine init` antes de nada.
- Host: macOS **arm64, 10 CPU, 16 GB**. En Mac **ambos corren una VM Linux**; la ventaja
  "Podman no necesita demonio/VM" no existe aquí.
- Lo que el lab necesita es OCI plano: puerto Jupyter 8888, bind-mount read-only de
  ~110 MB de DBF, TeX Live, versiones fijadas. Las ventajas reales de Podman — rootless,
  sin demonio, pods, systemd, `podman kube generate` — **no aplican**.
- Dos fricciones concretas donde Docker gana: el bind-mount VirtioFS de Docker Desktop
  lee los DBF más rápido, y al escribir `outputs/` al host **no hay remapeo de UID**
  (Podman rootless deja artefactos con dueño raro que hay que `chown`).
- **Podman sería mejor si**: empresa >250 empleados o >10 M USD (Docker Desktop exige
  licencia paga) o si hubiera que exportar a Kubernetes. Ninguno es el caso.
- **Sin lock-in:** el `Dockerfile` es OCI estándar → `podman build` lo corre sin cambios.
- Medición: la VM añade overhead igual a ambos algoritmos, la comparación relativa sigue
  válida. Aun así el contenedor se fija con `cpus: 6`, `mem_limit: 10g` y el reporte
  declara host, imagen y digest.

## Archivos a crear

```
Laboratiorio03/
├── archivos/                 # crudo, montado :ro, nunca se modifica
├── docker/Dockerfile         # python:3.12-slim + mlxtend + TeX Live (Times)
├── docker/docker-compose.yml # servicios: jupyter | bench | report
├── requirements.txt          # pins exactos
├── src/dbf_reader.py         # lector DBF VFP 0x32 (maneja tipo V + _NullFlags)
├── src/models.py             # dataclasses LineItem, Factura
├── src/parse.py              # DBF -> list[Factura] + normalización + caché parquet
├── src/eda.py                # punto 3 del enunciado
├── src/bench.py              # punto 5: grilla 20 corridas x 2 algoritmos
├── notebooks/lab03.ipynb     # ENTREGABLE 1
├── outputs/{figs,tables}/    # PNG + CSV citados por el reporte
├── report/report.tex         # ENTREGABLE 2 -> report.pdf
└── Makefile                  # make up|parse|eda|bench|notebook|report|all
```

`src/` separado del notebook a propósito: el benchmark corre sin kernel de Jupyter (su
overhead contaminaría los tiempos) y el notebook entregable queda legible.

Nota: ya existe `/Users/sara/Downloads/Laboratiorio03/PLAN.md` (versión preliminar de
este plan, escrita antes de entrar a plan mode). Se reemplaza por este contenido al
empezar, o se borra.

## Fases

### F1 — Entorno Docker
- `python:3.12-slim`; pins: `pandas==2.2.3`, `mlxtend==0.23.4`, `numpy==2.1.3`,
  `matplotlib==3.9.2`, `pyarrow==18.1.0`, `jupyterlab==4.3.4`, `scipy`, `tabulate`,
  `dbfread==2.0.7` (solo para contraste del parser).
- Capa TeX **mínima**: `texlive-latex-recommended` + `texlive-fonts-recommended`
  (traen `mathptmx` = Times). No `texlive-full` (4 GB inútiles).
- `docker-compose.yml`: `./archivos:/data:ro`, resto `:rw`, `cpus: 6`, `mem_limit: 10g`.
- Criterio de salida: `make up` entra al shell, `import mlxtend` y `pdflatex -v` responden.

### F2 — Lector DBF (punto 2)
`src/dbf_reader.py`, parser propio por `struct` (ya validado en exploración):
- Header: `nrec`, `hlen`, `rlen` desde `<IHH` en bytes 4:12; tabla de campos de 32 bytes
  hasta `0x0D`.
- **Tipo `V` (Varchar VFP)**: longitud en el último byte del campo; fallback `rstrip()`.
  Esta es la parte que `dbfread` no cubre y la razón del parser propio.
- `N` con decimales → `Decimal` → `float`; `C` → `latin-1` + `strip`.
- Salta registros borrados (primer byte `b'*'`).
- Validación cruzada contra `dbfread` en una muestra; donde discrepe, gana el parser
  propio y se documenta por qué.

### F3 — Modelo de dominio (punto 2, literal del enunciado)
`src/models.py` + `src/parse.py`:
- `LineItem`: `producto`, `codigo`, `cantidad` (`CANT`), `precio_unitario` (`PRECIO`),
  `subtotal` (`TOTAL`), `iva` (`IVA`), `descuento` (`VLRDESC`).
- `Factura`: `numero` (`CODCAJ`-`CODTRA`-`FACNRO`), `timestamp`, `total`, `total_iva`,
  `total_items`, `items: list[LineItem]`, más `fecha`, `caja`, `tipo_doc`.
- Normalización de producto: upper, colapso de espacios, acentos quitados solo para
  comparar (no para mostrar); canonización por `CODIGO` en los 23 códigos con varias
  `DESCRI`, tomando la descripción más frecuente.
- Control de calidad: `Σ subtotales` vs `TOTAL` con tolerancia ±1 peso (el DBF redondea);
  se reporta la tasa de descuadre como métrica de integridad.
- Caché `outputs/facturas.parquet` → no re-parsear 147k registros en cada ejecución.

### F4 — Hora-proxy (desbloquea 3.a / 3.b / 3.d)
- Columna `timestamp` alimentada por una **hora estimada**: orden físico del registro +
  `FACNRO` consecutivo por caja (monótono dentro del día), mapeado linealmente a jornada
  **07:00–22:00**.
- Toda figura y tabla derivada se rotula **"hora estimada (proxy ordinal)"**.
- El reporte lleva una línea de limitación con su impacto: el ranking por franja es
  relativo al supuesto de jornada; la *forma* de la curva es interpretable, las horas
  absolutas no.
- En paralelo (acción humana, no bloqueante): preguntar al docente si existe el archivo
  cabecera/maestro con `HORA`. Si llega, el análisis horario se recalcula exacto sin
  tocar el código — por eso la hora entra como columna, no cableada.

### F5 — Análisis descriptivo (punto 3)
`src/eda.py`, salidas a `outputs/figs/*.png` y `outputs/tables/*.csv`:
- **3.a**: registros, facturas, media/máx ítems por factura, histograma de tamaño de
  canasta, distribución de cantidad y de valor por franja (proxy), ticket promedio, IVA
  total, desglose por caja y **por día de la semana** (los 13 archivos cubren los 7 días:
  es la variable temporal que sí existe en los datos).
- **3.b**: producto más vendido global por nº de facturas y por unidades (`EMPANADAS`
  lidera con 7,448 apariciones) y por franja.
- **3.c**: TOP 10 con curva de Pareto y % acumulado; relación 1º vs 10º.
- **3.d**: productos de 1 sola aparición (`BOTELLON AGUA`, `CORTA UÑAS`, …); para cada
  uno: franja, si se vendió solo (`len(items)==1`) o acompañado y con qué, precio vs
  mediana de la canasta.

### F6 — Hipótesis (punto 4)
Formulada **después** del EDA y falsable, cada una con el número que la decide:
- **H1**: la canasta está dominada por consumo inmediato → las reglas fuertes serán
  snack→bebida, no cross-category.
- **H2**: con canasta media 2.14 y 40% de facturas de 1 ítem, el soporte alcanzable para
  pares es muy bajo → los combos "más vendido + menos vendido" no tendrán soporte
  suficiente y solo aparecerán vía `lift`, con riesgo de regla espuria.
- **H3**: la cola larga aparece sobre todo en facturas de 1 ítem → no son candidatos a
  combo, son compras de misión única.

### F7 — Reglas y benchmark (punto 5)
`src/bench.py`:
- Entrada **común** a ambos algoritmos: `TransactionEncoder` sobre canastas
  deduplicadas (mismo producto dos veces en una factura = un ítem). Matriz
  68,809 × 1,280 `bool` ≈ 88 MB — cabe sin problema en 10 GB.
- `mlxtend.frequent_patterns.apriori` y `fpgrowth`, misma entrada y mismos parámetros:
  la única variable es la implementación. Luego `association_rules` guardando `support`,
  `confidence`, `lift`, `leverage`, `conviction`, `zhangs_metric`.
- **Diseño de las 20 corridas (punto 5.d)** — grilla cruzada:
  `min_support ∈ {0.0002, 0.0005, 0.001, 0.002, 0.005}` × `max_len ∈ {2, 3, 4, None}`
  = 20 combinaciones × 2 algoritmos. Cada combinación se repite 3 veces; se reporta la
  **mediana** (defensa contra ruido del scheduler) más min/max.
- Medición: `time.perf_counter_ns`, warm-up descartado, `tracemalloc` para memoria pico,
  `len(itemsets)` y `len(rules)` por corrida; `timeout` de 600 s registrado como
  **censurado** en vez de colgar el notebook. Resultados a `outputs/tables/bench.csv`.
- **Prueba de equivalencia**: para los mismos parámetros, los itemsets de apriori y
  fp-growth deben ser idénticos como conjunto →
  `assert set(map(frozenset, a.itemsets)) == set(map(frozenset, f.itemsets))`.
  Ese assert es lo que permite afirmar "difieren en tiempo, no en resultado" con
  evidencia en vez de con fe.
- Tabla comparativa: parámetros × algoritmo → tiempo mediano, speedup fp-growth/apriori,
  nº itemsets, nº reglas, memoria pico.
- Gráficas: tiempo vs `min_support` (log-log, dos curvas) y nº de reglas vs `min_support`.

### F8 — Las 8 preguntas, cada una contra un artefacto
1. **k=1 predecibles** → tabla de reglas 1→1 ordenada por confianza, con lectura.
2. **apriori vs fp-growth** → el assert de equivalencia + curva de tiempos: la diferencia
   es de rendimiento y escala, no de salida.
3. **Efecto de parámetros** → la grilla de F7: soporte bajo = más reglas y coste
   superlineal en apriori; `max_len` recorta el blow-up.
4. **Reglas espurias** → casos con `lift` alto y soporte de 2-3 facturas. Criterio de
   descarte: soporte **absoluto** ≥ 30 facturas + `leverage`/`zhang`. Se muestra una
   regla espuria concreta y con qué parámetros aparece.
5. **k>1** → itemsets de 3+ ítems con `max_len=None` al soporte más bajo viable.
6. **Probabilidad al azar** → `P(A)·P(B)` (independencia esperada) vs `support(A∪B)`
   observado; eso *es* el `lift`. Cálculo explícito con un par real + test chi²/Fisher
   sobre la tabla 2×2 (*extra: da rigor estadístico*).
7. **Reglas interesantes** → `lift > 1.5` **y** soporte absoluto ≥ 30 **y** que no sean
   dos presentaciones del mismo producto.
8. **Comportamiento del consumidor** → lectura de tienda de conveniencia: misión de
   compra inmediata, combo snack+bebida, impulso en caja, ticket bajo.

### F9 — Entregables
- **`notebooks/lab03.ipynb`**: importa de `src/`, narrativa mínima en Markdown entre
  celdas, salidas guardadas. `make notebook` lo re-ejecuta con `nbconvert --execute`
  para garantizar que corre limpio desde cero.
- **`report/report.pdf`**: `\documentclass[12pt,twocolumn]{article}` + `\usepackage{mathptmx}`
  (Times New Roman) + `geometry`. Secciones: Introducción · Datos y método · Resultados
  (EDA + tabla comparativa + 2 figuras) · Respuestas · Conclusiones · Limitaciones.
  Compilado **dentro del contenedor** (`make report`) → el PDF no depende del Mac.
- **Control de longitud**: 2 páginas con 8 preguntas es duro. Las respuestas van en una
  tabla compacta y las figuras son **2**, no 6: tabla comparativa apriori/fp-growth +
  Pareto del TOP 10.

## Extras incluidos (el "aunque sea opcional")

- Día de la semana (fin de semana vs semana): variable temporal real, no proxy.
- Test chi²/Fisher por regla → separa asociación de ruido.
- Tercera implementación de contraste (`fpmax`/`hmine` de mlxtend) → muestra que la
  diferencia apriori↔fp-growth es de familia algorítmica (nivel vs árbol).
- Anexo de sensibilidad "solo FA" vs todos los `CODTRA`.
- Validación de integridad `Σ subtotales ≠ TOTAL`.
- `Makefile` + imagen fijada por digest → el docente reproduce en un comando.
- Anexo de datos faltantes: `fc05042024.DBF` ausente.

## Orden de ejecución

1. **F1** Docker (bloquea todo) — arrancar Docker Desktop primero.
2. **F2 + F3** parser, modelo y caché parquet.
3. **F4** hora-proxy; en paralelo pedir el dato real al docente.
4. **F5** EDA → **F6** hipótesis (en ese orden, no al revés).
5. **F7** benchmark — lo más largo; se deja corriendo con `make bench` fuera de Jupyter.
6. **F8** respuestas → **F9** notebook + PDF.

## Verificación

- `make up` → `python -c "import mlxtend, pandas; print(mlxtend.__version__)"` y
  `pdflatex -v` responden dentro del contenedor.
- `make parse` → imprime 147,413 line-items y 68,809 facturas (coincide con lo medido
  aquí); `outputs/facturas.parquet` existe; tasa de descuadre `Σ subtotales` vs `TOTAL`
  por debajo del umbral y reportada.
- Parser: test que compara `src/dbf_reader.py` contra `dbfread` en 100 registros y
  verifica que `DESCRI` no trae el byte de longitud (p. ej. `'LUCKY DAIQUIRI X 10'`
  sin la `B` final).
- `make bench` → `outputs/tables/bench.csv` con **120 filas** (20 combinaciones × 2
  algoritmos × 3 repeticiones) y el assert de equivalencia de itemsets en verde para
  las 20 combinaciones.
- `make notebook` → re-ejecución limpia de punta a punta, exit 0.
- `make report` → `report.pdf` de **≤ 2 páginas**; verificar con
  `pdfinfo report.pdf | grep Pages`, y que la fuente embebida sea Times
  (`pdffonts report.pdf`).
- `make all` desde cero en un contenedor recién construido reproduce todos los
  artefactos sin intervención manual.
