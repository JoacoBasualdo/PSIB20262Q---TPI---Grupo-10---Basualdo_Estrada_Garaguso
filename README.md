# PSIB2026Q2-TPI-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Imágenes Biomédicas (TPI) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

**Integrantes:** Joaquín Basualdo, Bautista Estrada y Facundo Garaguso.

## Índice

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Datos: qué descargar y dónde ponerlo](#datos-qué-descargar-y-dónde-ponerlo)
5. [Ejecución](#ejecución)
6. [Cómo funciona el procesamiento](#cómo-funciona-el-procesamiento)
7. [Estructura del código](#estructura-del-código)
8. [Resultados](#resultados)
9. [Problemas frecuentes](#problemas-frecuentes)
10. [Referencias](#referencias)

---

## Descripción del proyecto

El proyecto cuantifica la atrofia cerebral asociada a la enfermedad de Alzheimer a partir de resonancias magnéticas T1 longitudinales del dataset OASIS-2. Para cada paciente se mide, en cada visita:

- el **volumen de tejido cerebral** (sustancia gris + blanca);
- el **volumen de los ventrículos laterales** (izquierdo y derecho);
- la **fracción ventricular**: el volumen ventricular dividido por el volumen de la zona intracraneal, en %;
- la **textura del tejido periventricular**: características de la matriz de co-ocurrencia de grises (GLCM) calculadas en una banda de 4 mm de tejido alrededor de los ventrículos, para evaluar si el tejido se vuelve más heterogéneo.

Después se analiza cómo cambian esas medidas entre visitas y se relacionan con la evolución clínica (CDR y MMSE). El método se valida de tres formas: contra los valores publicados por OASIS, repitiendo la medición sobre dos adquisiciones de la misma sesión (test-retest) y contra segmentaciones manuales de MRBrainS18.

| Paciente | Grupo | Visitas | Edad | Evolución clínica |
|---|---|---|---|---|
| OAS2_0070 | Sano (*Nondemented*) | 5 en 6,5 años | 80 → 86 | CDR 0 en todas las visitas |
| OAS2_0048 | Demencia (*Demented*) | 5 en 3,4 años | 66 → 69 | CDR 1; el MMSE cae de 19 a 4 |
| OAS2_0087 | Demencia (*Demented*) | 2 en 2,1 años | 96 → 98 | CDR 1 → 2 |
| OAS2_0127 | Convertido (*Converted*) | 5 en 7,2 años | 79 → 86 | CDR 0 → 0,5 desde la segunda visita |

**CDR** (*Clinical Dementia Rating*): 0 = sin demencia, 0,5 = muy leve, 1 = leve, 2 = moderada.
**MMSE** (*Mini-Mental State Examination*): puntaje de 0 a 30; cuanto más bajo, mayor deterioro cognitivo.

---

## Requisitos

- **Python 3.10 o superior.**
- Las bibliotecas de `requirements.txt`: `nibabel`, `numpy`, `matplotlib`, `SimpleITK`, `pandas`, `openpyxl`, `scipy`, `nilearn`, `scikit-image` y `PySide6` (esta última solo para la interfaz gráfica).
- **Espacio en disco:** ~19 GB para descargar los dos `.tar.gz` de OASIS-2. Una vez extraídos los 4 pacientes, los `.tar.gz` se pueden borrar: los datos que usa el código ocupan mucho menos.
- **Internet:** para descargar los datos y, la primera vez que se corre `main.py`, el atlas Harvard-Oxford.
- **[aria2](https://aria2.github.io/)** (recomendado): descarga los archivos grandes en paralelo y retoma la descarga si se corta.

## Instalación

Desde la carpeta raíz del repositorio:

```bash
# (Opcional) entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

---

## Datos: qué descargar y dónde ponerlo

Los datos **no se suben al repositorio** (`Dataset/` está en el `.gitignore`). Cada integrante los descarga en su computadora.

### Resumen de descargas

| Qué | Para qué | Link | Tamaño | ¿Obligatorio? |
|---|---|---|---|---|
| OASIS-2, parte 1 | Imágenes de los pacientes | [OAS2_RAW_PART1.tar.gz](https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz) | ~10 GB | Sí |
| OASIS-2, parte 2 | Imágenes de los pacientes | [OAS2_RAW_PART2.tar.gz](https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz) | ~8 GB | Sí |
| Excel clínico de OASIS-2 | Edad, CDR, MMSE, eTIV, nWBV de cada visita | [oasis_longitudinal_demographics.xlsx](https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx) | < 1 MB | Sí (lo baja solo `descargar_dataset.py`) |
| Atlas Harvard-Oxford | Ubicar los ventrículos | Automático (nilearn), se guarda en `Dataset/Atlas/` | Chico | Sí (lo baja solo `main.py`) |
| Plantilla MNI152 | Extraer el cerebro | Viene incluida en nilearn | — | Sí (no hay que descargar nada) |
| MRBrainS18 (`training.zip`) | Validar los ventrículos | [DataverseNL](https://doi.org/10.34894/E0U32Q) | 471 MB | No (solo para `VALIDAR_MRBRAINS`) |

Página oficial de OASIS-2: https://sites.wustl.edu/oasisbrains/home/oasis-2/

### Paso 1: descargar OASIS-2

1. Instalar aria2 (una sola vez) y **abrir una terminal nueva** para que reconozca el comando:
   ```powershell
   winget install aria2.aria2
   ```
2. Desde la carpeta raíz del repo, descargar las dos partes en la carpeta `Dataset/`. Se pueden correr en dos terminales a la vez:
   ```powershell
   aria2c -c -x 16 -s 16 -k 1M -d Dataset https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz
   aria2c -c -x 16 -s 16 -k 1M -d Dataset https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz
   ```
   `-c` permite retomar la descarga si se corta: alcanza con volver a correr el mismo comando. Mientras al lado del `.tar.gz` exista un archivo `.aria2`, la descarga no terminó.

   Sin aria2, también se pueden bajar desde el navegador con los links de la tabla y guardarlos en `Dataset/`.

### Paso 2: extraer los pacientes usados

```bash
python descargar_dataset.py
```

El script:
1. descarga el Excel clínico a `Dataset/oasis_longitudinal_demographics.xlsx` (si no está);
2. recorre los `.tar.gz` y extrae **solo** los 4 pacientes del trabajo;
3. verifica que estén todas las visitas.

Tiene que terminar con **`Dataset listo.`**. Si dice que faltan visitas, poner `CORTAR_AL_TERMINAR = False` en `Funciones/Descarga.py` y volver a correrlo.

> Importante: el script **no** descarga los `.tar.gz`; los lee desde `Dataset/`. Si no los encuentra, da `FileNotFoundError` (ver [Problemas frecuentes](#problemas-frecuentes)).

### Paso 3 (opcional): MRBrainS18, para validar los ventrículos

1. Entrar a https://mrbrains18.isi.uu.nl/data y seguir el enlace al dataset en DataverseNL (https://doi.org/10.34894/E0U32Q).
2. Leer la pestaña **Terms**: condiciones de uso y cita obligatoria.
3. Descargar **`training.zip`** (471 MB, 7 sujetos con segmentación manual). `test.zip` no sirve porque no trae segmentaciones.
4. Descomprimirlo en `Dataset/MRBrainS18/`, con una carpeta por sujeto, cada una con su `segm.nii.gz` y su T1.

Si esta carpeta no existe, `main.py` saltea esta validación y sigue con el resto.

### Estructura final esperada

```
Dataset/
├── oasis_longitudinal_demographics.xlsx
├── OAS2_RAW_PART1.tar.gz                 (se puede borrar después del paso 2)
├── OAS2_RAW_PART2.tar.gz                 (se puede borrar después del paso 2)
├── OAS2_RAW_PART1/
│   ├── OAS2_0048_MR1/RAW/mpr-1.nifti.img (+ .hdr, mpr-2, mpr-3, ...)
│   ├── OAS2_0048_MR2/RAW/...
│   └── ...
├── OAS2_RAW_PART2/
│   └── OAS2_0127_MR1/RAW/...
├── Atlas/                                (lo crea main.py: atlas Harvard-Oxford)
└── MRBrainS18/                           (opcional)
    ├── 1/
    │   ├── segm.nii.gz
    │   └── orig/T1.nii.gz ...
    ├── 4/
    └── ...
```

`main.py` busca cada visita (`OAS2_XXXX_MRn`) en todas las carpetas `OAS2_RAW_PART*`, así que no importa en qué parte quedó cada una.

---

## Ejecución

Desde la carpeta raíz del repositorio:

```bash
python descargar_dataset.py   # una sola vez, para extraer los datos
python main.py                # los 4 pacientes: procesamiento, interpretación y validación
python Interfaz/App.py        # interfaz gráfica: un paciente por vez
```

Hay dos formas de correr el procesamiento:

| | `main.py` | Interfaz gráfica (`Interfaz/App.py`) |
|---|---|---|
| Pacientes | Los 4 juntos (lista `PACIENTES`) | Uno por vez, elegido de una lista |
| Opciones | Se cambian editando las constantes al principio de `main.py` | Se eligen en la ventana |
| Resultados | `Resultados/` (CSV, gráficos y un PDF por paciente) | `Resultados/Interfaz/<paciente>/`, sin pisar los de `main.py`; el PDF se guarda con un botón |
| Para qué | Generar los resultados completos del trabajo | Explorar un paciente y ver cada etapa |

### Interfaz gráfica (NeuroVol)

```bash
python Interfaz/App.py
```

Se puede correr desde cualquier carpeta: el programa se ubica solo en la raíz del repo. Necesita que el dataset ya esté extraído (paso 2 de [Datos](#datos-qué-descargar-y-dónde-ponerlo)).

1. **Paciente**: se elige de la lista. Solo aparecen los pacientes que tienen visitas extraídas en `Dataset/`. Al elegir uno se muestran su grupo, sus edades y la evolución del CDR y el MMSE.
2. **Visitas a procesar**: cuántas visitas del paciente se procesan, empezando por la primera.
3. **Validación**: casillas para la concordancia con OASIS (necesita al menos 2 visitas), el test-retest y MRBrainS18 (se deshabilita si no está descargado). En **opciones avanzadas** se elige cuántas visitas usar en el test-retest.
4. **Ejecutar**: el procesamiento corre en segundo plano y una barra muestra el avance por visita y por etapa.
5. **Resultados**: aparecen en pestañas, una por etapa (Lectura, Procesamiento, Features, Interpretación y Validación), con los textos y las figuras de cada visita. La pestaña **Consola** muestra todo lo que se imprimió. En la pestaña **Features** se ven, para cada visita, la banda periventricular donde se mide la textura y, en la primera visita, la matriz de co-ocurrencia (GLCM). Cada pestaña tiene controles de zoom para las figuras (**−**, **+** y **Ajustar al ancho**, o Ctrl + rueda del mouse).
6. **Guardar PDF** genera el reporte con todo lo que se muestra en las pestañas, y **Abrir carpeta de resultados** abre `Resultados/Interfaz/<paciente>/`.

Procesar un paciente con 5 visitas tarda unos 8-10 minutos (más si se activan el test-retest o MRBrainS18).

### Opciones de `main.py`

Al principio de `main.py` se elige qué partes correr:

| Opción | Qué hace | Tiempo aproximado |
|---|---|---|
| `PROCESAR_IMAGENES` | Procesa las resonancias de los 4 pacientes. Con `False` usa el `Resultados/volumenes.csv` ya generado y solo rehace la interpretación y la validación | ~30-40 min |
| `VALIDAR_CONCORDANCIA_OASIS` | Compara nuestra tasa de atrofia con la del nWBV publicado por OASIS | segundos |
| `VALIDAR_TEST_RETEST` | Mide el error del método (volúmenes y textura) comparando dos adquisiciones de la misma sesión (`VISITAS_TEST_RETEST` visitas por paciente) | ~2-3 min por visita |
| `VALIDAR_MRBRAINS` | Compara nuestros ventrículos con la segmentación manual de MRBrainS18 (se saltea si no está `Dataset/MRBrainS18/`) | ~1-2 min por sujeto |
| `VISITAS_TEST_RETEST` | Cantidad de visitas por paciente que se usan en el test-retest | — |
| `PACIENTES` | Lista de pacientes a procesar | — |
| `CANTIDAD_VISITAS` | Máximo de visitas por paciente | — |

> El registro de SimpleITK usa varios hilos, así que dos corridas pueden diferir en ~1% en algunos volúmenes. Es normal y está por debajo del error del método medido en el test-retest.

---

## Cómo funciona el procesamiento

Para cada paciente y cada visita (`procesar_paciente` en `main.py`):

1. **Lectura**: se cargan las 3-4 adquisiciones T1 (MPRAGE) de la sesión, formato Analyze, vóxel de 1 × 1 × 1,25 mm.
2. **Promediado**: las adquisiciones se alinean con `mpr-1` (registro rígido por información mutua) y se promedian para reducir el ruido.
3. **Máscara de la cabeza**: umbral de Otsu, cierre morfológico, relleno de huecos y componente conectada más grande.
4. **Corrección de *bias field***: algoritmo N4 sobre la máscara de la cabeza, para que un mismo tejido tenga el mismo brillo en toda la imagen.
5. **Normalización de intensidad**: percentiles 1 y 99 de la cabeza → 0 y 1.
6. **Extracción del cerebro**:
   - **Primera visita**: se registra la plantilla MNI152 sobre el paciente (registro afín, con búsqueda de la inclinación inicial) para obtener la **zona intracraneal**. Adentro de esa zona, el tejido se separa del líquido con un umbral calculado por k-medias de 4 clases.
   - **Visitas siguientes**: se alinean con la primera visita (registro rígido) y se usa **la misma zona intracraneal**, así todas las visitas se miden sobre la misma región.
7. **Ventrículos laterales**: las etiquetas del atlas Harvard-Oxford se llevan al paciente con las mismas transformaciones. Alrededor de cada etiqueta (8 mm) se buscan los vóxeles oscuros (líquido) y se conservan las regiones conectadas a la etiqueta. Cada visita recibe un **estado de control** (`REVISAR_MASCARA`, `REVISAR_VOLUMEN` o `REVISAR_REGISTRO_O_SEGMENTACION`).
8. **Textura periventricular**: se arma una banda de 4 mm de tejido alrededor de los ventrículos (dilatación de la máscara ventricular, sin los ventrículos y solo donde hay tejido). Sobre la imagen normalizada, cuantizada a 32 niveles de gris, se calcula la matriz de co-ocurrencia (GLCM) en cada corte axial, a distancia 1 y en 4 direcciones, y se suman todas en una sola matriz. De ella se obtienen contraste, homogeneidad, energía, correlación y entropía, además de la media y el desvío del brillo de la banda.
9. **Medidas**: volumen de tejido, volumen intracraneal, nWBV propio, ventrículos (total, izquierdo y derecho), fracción ventricular y características de textura.
10. **Salidas**: PDF del paciente con las figuras de cada etapa, máscara ventricular en NIfTI y comparación anatómica de la primera y la última visita.

Después de procesar a todos los pacientes:

- **Interpretación** (`interpretar`): cambios desde la primera visita y entre visitas, tasas por año, evolución de la textura periventricular y relación con CDR y MMSE.
- **Validación** (`validar`): concordancia con OASIS, test-retest (volúmenes y textura) y MRBrainS18.

---

## Estructura del código

```
├── main.py                 Flujo completo: procesamiento, interpretación y validación
├── descargar_dataset.py    Extrae los pacientes de OASIS-2 (llama a Funciones/Descarga.py)
├── requirements.txt
├── Funciones/              Un archivo por etapa
├── Interfaz/               Interfaz gráfica (PySide6)
├── Resultados/             CSV, gráficos, PDFs y máscaras generados
└── Informe/                Informe en LaTeX y PDF
```

| Archivo de `Interfaz/` | Qué hace |
|---|---|
| `App.py` | Ventana principal: encabezado, panel de parámetros, barra de progreso, pestañas de resultados y botones para guardar el PDF y abrir la carpeta. Se ejecuta con `python Interfaz/App.py` |
| `Panel_parametros.py` | Panel izquierdo: paciente, visitas, validaciones y opciones avanzadas. Controla que los parámetros sean válidos antes de ejecutar |
| `Trabajador.py` | Corre `ejecutar_pipeline` de `main.py` en otro hilo, para que la ventana no se congele, y le manda a la ventana el texto, el avance y los resultados |
| `Vista_resultados.py` | Arma cada pestaña de resultados con sus subtítulos, textos e imágenes, y los controles de zoom de las figuras |
| `Estilo.py` | Colores, tipografías y nombre de la aplicación |
| `recursos/` | Logo del ITBA e íconos |

| Módulo | Qué hace |
|---|---|
| `Lectura.py` | Carga de las resonancias, tamaño de vóxel, carpetas de cada visita, Excel clínico y datos de MRBrainS18 |
| `Procesamiento.py` | Promediado de las adquisiciones, máscara de la cabeza, corrección de *bias field* (N4) y normalización de intensidad |
| `Features.py` | Extracción del cerebro (plantilla MNI152 y registro entre visitas), ventrículos laterales (atlas Harvard-Oxford), fracción ventricular y textura periventricular (GLCM) |
| `Interpretacion.py` | Evolución del cerebro, los ventrículos y la textura entre visitas, y relación con CDR y MMSE |
| `Validacion.py` | Concordancia con OASIS, test-retest y comparación con MRBrainS18 |
| `Visualizacion.py` | Todas las figuras |
| `Reporte.py` | PDF por paciente y máscaras ventriculares en NIfTI |
| `Descarga.py` | Extracción del dataset OASIS-2 desde los `.tar.gz` |

<details>
<summary><b>Detalle de las funciones de cada módulo</b> (click para desplegar)</summary>

#### `main.py`
| Función | Qué hace |
|---|---|
| `procesar_paciente(paciente_id, cantidad_visitas, guardar_pdf)` | Procesa las visitas de un paciente, genera su PDF (si `guardar_pdf`) y devuelve una fila por visita |
| `procesar_todos()` | Procesa los pacientes de `PACIENTES` y guarda `volumenes.csv` después de cada uno |
| `interpretar(datos, carpeta_resultados, ...)` | Evolución del cerebro, de los ventrículos, de la textura y relación con la clínica (CSV y gráficos) |
| `validar(datos, carpeta_resultados, ...)` | Corre las validaciones elegidas (por defecto, las activadas en las opciones) |
| `ejecutar_pipeline(paciente_id, ...)` | Procesa, interpreta y valida UN paciente y guarda en `Resultados/Interfaz/<paciente>/`. Es lo que corre la interfaz |
| `main()` | Punto de entrada: procesa (o lee el CSV), interpreta y valida |

`ETAPAS` define las cinco etapas (Lectura, Procesamiento, Features, Interpretación y Validación). Cada texto y figura queda marcado con su etapa, y la interfaz muestra una pestaña por cada una.

#### `Lectura.py`
| Función | Qué hace |
|---|---|
| `cargar_volumen_nifti(ruta)` | Lee una resonancia (Analyze/NIfTI) como matriz 3D |
| `leer_tamano_voxel(ruta)` | Tamaño del vóxel en mm, leído del encabezado |
| `listar_adquisiciones(carpeta_raw)` | Lista las adquisiciones de una visita, con `mpr-1` primero |
| `cargar_datos_clinicos(ruta_xlsx, paciente_id)` | Filas del Excel de OASIS de un paciente |
| `buscar_carpeta_visita(ruta_dataset, mri_id)` | Busca la carpeta `RAW` de una visita en `OAS2_RAW_PART1` y `PART2` |
| `listar_pacientes(ruta_dataset, ruta_datos_clinicos)` | Pacientes con visitas extraídas en `Dataset/`, con sus datos clínicos (arma la lista de la interfaz) |
| `buscar_archivos_mrbrains(carpeta_sujeto)` | Segmentación manual y T1 de 1 mm de un sujeto de MRBrainS18 |
| `cargar_t1_como_oasis(ruta)` | Carga la T1 de MRBrainS18 en la orientación que usa el código |

#### `Procesamiento.py`
| Función | Qué hace |
|---|---|
| `numpy_a_sitk(volumen, tamano_voxel)` | Convierte una matriz de NumPy en imagen de SimpleITK con su tamaño físico |
| `registrar_rigido(fija, movil)` | Registro rígido (6 parámetros) por información mutua de Mattes, en 3 resoluciones |
| `movimiento_de_transformada(transformada)` | Resume la rotación y el desplazamiento de un registro |
| `promediar_adquisiciones(volumenes, tamano_voxel)` | Alinea las adquisiciones con `mpr-1` y las promedia |
| `mascara_cabeza(volumen, tamano_voxel)` | Máscara de la cabeza: Otsu, cierre, relleno y componente más grande |
| `rellenar_huecos_por_corte(mascara)` | Rellena los huecos de una máscara corte a corte |
| `corregir_bias_field(volumen, tamano_voxel, ...)` | Corrección N4 de la inhomogeneidad de campo |
| `normalizar_intensidad(volumen, mascara, ...)` | Lleva los percentiles 1 y 99 a 0 y 1 |
| `filtro_mediana(volumen, tamano=3)` | Filtro de mediana 3×3×3 (solo para una figura comparativa) |
| `filtro_gaussiano(volumen, tamano_voxel, sigma_mm)` | Filtro gaussiano (no se usa en el flujo principal) |

#### `Features.py`
| Función | Qué hace |
|---|---|
| `distancia_al_borde`, `erosionar`, `dilatar` | Morfología en milímetros (independiente del tamaño del vóxel) |
| `componente_mas_grande(mascara)` | Se queda con la región conectada más grande |
| `volumen_ml(mascara, tamano_voxel)` | Cantidad de vóxeles → mililitros |
| `limites_kmedias(valores, clases=4)` | Umbral de tejido por k-medias de 4 clases |
| `cargar_atlas()` | Carga la plantilla MNI152 2009a y su máscara de cerebro |
| `busqueda_inclinacion(...)` | Prueba varias inclinaciones iniciales y elige la que mejor alinea |
| `registrar_atlas(...)` / `ubicar_atlas(...)` | Registro afín de la plantilla MNI152 sobre el paciente |
| `tejido_dentro_de_zona(...)` | Tejido dentro de la zona intracraneal, sin meninges ni senos venosos |
| `mascara_cerebro(...)` | Extracción del cerebro en la primera visita |
| `registrar_visitas(...)` | Registro rígido entre visitas, con dos puntos de partida y control por correlación |
| `rigida_desde_atlas(...)` | Punto de partida del registro a partir del atlas (cuando la cabeza se movió mucho) |
| `mascara_cerebro_seguimiento(...)` | Extracción del cerebro en las visitas siguientes, con la zona de la primera |
| `segmentar_ventriculos(...)` | Ventrículos laterales con el atlas Harvard-Oxford y umbral de líquido |
| `fraccion_ventricular(vent_ml, intracraneal_ml)` | 100 · ventrículos / zona intracraneal |
| `banda_periventricular(mascara_ventricular, mascara_tejido, tamano_voxel, grosor_mm)` | Banda de tejido de 4 mm alrededor de los ventrículos, donde se mide la textura |
| `cuantizar(normalizada, niveles)` | Pasa la imagen normalizada [0, 1] a 32 niveles de gris enteros |
| `glcm_corte(corte_cuantizado, mascara_corte)` | Matriz de co-ocurrencia de un corte, contando solo pares de vóxeles dentro de la banda |
| `caracteristicas_textura(normalizada, mascara_banda)` | Suma las GLCM de todos los cortes y calcula contraste, homogeneidad, energía, correlación, entropía, media y desvío |

#### `Interpretacion.py`
| Función | Qué hace |
|---|---|
| `calcular_evolucion(datos)` | Cambio del volumen cerebral y del nWBV desde la primera visita |
| `resumir_pacientes(evolucion)` | Resumen por paciente: cambio total y tasa por año |
| `calcular_cambios_ventriculares(datos)` | Cambio de los ventrículos (ml, % y fracción ventricular) |
| `analizar_evolucion_clinica(datos)` | Cambios morfológicos junto con CDR y MMSE; devuelve el detalle por visita y el resumen |
| `calcular_cambios_textura(datos)` | Cambio de cada característica de textura desde la primera visita (%) y tasa por año de cada paciente |

#### `Validacion.py`
| Función | Qué hace |
|---|---|
| `concordancia_oasis(datos)` | Compara la pendiente de nuestro nWBV con la del de OASIS |
| `test_retest(...)` | Mide cerebro, ventrículos y textura (contraste, homogeneidad y entropía) en `mpr-1` y `mpr-2` de una misma visita |
| `validacion_mrbrains(...)` | Dice y diferencia de volumen contra la segmentación manual de MRBrainS18 |
| `pendiente`, `dice`, `a_grilla_manual`, ... | Funciones auxiliares |
| `imprimir_*` | Muestran cada validación en la terminal |

#### `Visualizacion.py`
| Función | Qué hace |
|---|---|
| `visualizar_planos_ortogonales` | Cortes sagital, coronal y axial |
| `graficar_histogramas_visitas` | Histogramas de intensidad por visita |
| `graficar_histograma_bias` | Histograma de la cabeza antes y después de la corrección N4 |
| `visualizar_mascaras` | Contornos de las máscaras sobre la imagen |
| `graficar_histograma_umbral` | Histograma con el umbral de k-medias |
| `visualizar_ventriculos` | Contorno de los ventrículos en tres planos |
| `visualizar_comparacion_ventricular` | Primera y última visita sobre los mismos cortes |
| `visualizar_banda_textura` | Banda periventricular (magenta) y ventrículos (amarillo) en tres planos |
| `graficar_glcm` | Matriz de co-ocurrencia como imagen |
| `guardar_graficos_evolucion` | Evolución del volumen cerebral y del nWBV |
| `graficar_evolucion_ventricular` | Evolución del tejido y de los ventrículos |
| `graficar_fraccion_ventricular` | Fracción ventricular de todos los pacientes |
| `graficar_evolucion_textura` | Cambio de las características de textura desde la primera visita, por paciente |
| `graficar_tasas_textura` | Tasa de cambio por año de la textura, para todos los pacientes |
| `graficar_analisis_clinico` | Morfología junto con CDR y MMSE |
| `graficar_concordancia_oasis` | nWBV propio contra el de OASIS |
| `graficar_validacion_mrbrains` | Contorno manual (verde) contra el nuestro (rojo) |

#### `Reporte.py`
| Función | Qué hace |
|---|---|
| `iniciar_registro`, `agregar_subtitulo`, `agregar_figura` | Van juntando el texto de la terminal y las figuras de un paciente |
| `agregar_imagen(ruta)` | Agrega al reporte un gráfico ya guardado como imagen |
| `marcar_etapa(nombre)` | Marca con esa etapa todo lo que se imprima o agregue desde ese momento |
| `al_cambiar_etapa(funcion)` | Avisa a la interfaz cada vez que cambia la etapa (para la barra de progreso) |
| `cambiar_consola(nueva)` | Manda lo que se imprime a otro destino (la consola de la interfaz) |
| `elementos_por_etapa(etapas)` | Agrupa textos y figuras por etapa, para las pestañas de la interfaz |
| `guardar_pdf(ruta_pdf, titulo)` | Arma el PDF del paciente |
| `imprimir_datos_clinicos(datos)` | Muestra la tabla clínica del paciente |
| `guardar_mascara_ventricular(...)` | Guarda la máscara ventricular en NIfTI, con la ubicación espacial de `mpr-1` |

#### `Descarga.py`
| Función | Qué hace |
|---|---|
| `main()` | Excel clínico → extracción de los pacientes → verificación |
| `diagnostico()` | Muestra dónde busca la carpeta `Dataset` y si encuentra los `.tar.gz` |
| `extraer_pacientes(url, pacientes, destino)` | Recorre un `.tar.gz` y extrae solo los pacientes pedidos |
| `verificar(pacientes)` | Compara las visitas del Excel con las carpetas extraídas |

</details>

---

## Resultados

Todo queda en `Resultados/`:

| Archivo | Contenido |
|---|---|
| `OAS2_XXXX_visitas.pdf` | Reporte de cada paciente: salida de la terminal y figuras de cada etapa |
| `volumenes.csv` | Una fila por visita: datos clínicos, volumen de tejido, zona intracraneal, nWBV propio, ventrículos (ml), fracción ventricular, estado de control, volumen de la banda periventricular y características de textura (`textura_...`) |
| `comparacion_volumenes.csv`, `resumen_longitudinal.csv` | Evolución del volumen cerebral |
| `comparacion_ventriculos.csv` | Evolución de los ventrículos (ml, % y fracción ventricular) |
| `comparacion_textura.csv` | Cambio de cada característica de textura desde la primera visita, por visita |
| `resumen_textura.csv` | Tasa de cambio por año de cada característica de textura, por paciente |
| `analisis_clinico_integrado.csv`, `resumen_clinico_integrado.csv` | Cambios morfológicos junto con CDR y MMSE |
| `concordancia_oasis.csv` | Pendiente de nuestro nWBV contra la del de OASIS |
| `test_retest.csv` | Diferencia entre `mpr-1` y `mpr-2` de una misma visita (volúmenes y textura) |
| `validacion_mrbrains.csv` | Dice y error de volumen contra MRBrainS18 |
| `Graficos/` | Evolución por paciente, fracción ventricular, textura periventricular (evolución y tasas), análisis clínico y concordancia con OASIS |
| `Graficos/ComparacionAnatomica/` | Primera y última visita de cada paciente sobre los mismos cortes |
| `Graficos/ValidacionMRBrainS/` | Contorno manual contra el nuestro para cada sujeto de MRBrainS18 |
| `MascarasVentriculares/OAS2_XXXX/` | Máscara 3D de los ventrículos de cada visita (`.nii.gz`); se puede abrir en ITK-SNAP, 3D Slicer o FSLeyes junto con `mpr-1` |
| `Interfaz/OAS2_XXXX/` | Resultados de la interfaz gráfica para ese paciente: las mismas tablas y gráficos, pero de un solo paciente |

> Los resultados de textura de cada paciente están en `Resultados/Interfaz/<paciente>/`. Los CSV generales de `Resultados/` se generaron antes de agregar la textura; para incluirla, correr `main.py` con `PROCESAR_IMAGENES = True`.

**Fracción ventricular:** volumen ventricular dividido por el volumen de la zona intracraneal, en %. La zona intracraneal es la misma región en todas las visitas de un paciente: dentro de un paciente, la fracción cambia solo si cambian los ventrículos; entre pacientes, quita el efecto del tamaño de la cabeza.

**Zona intracraneal:** es la máscara de cerebro de la plantilla MNI152 llevada al paciente. No es el volumen intracraneal real (mide entre 1,02 y 1,19 veces el eTIV de OASIS), pero sirve como referencia fija para relativizar.

**Textura periventricular:** se mide sobre la imagen normalizada, dentro de una banda de 4 mm de tejido alrededor de los ventrículos. Si el tejido se vuelve más heterogéneo, el contraste y la entropía suben, y la homogeneidad y la energía bajan. Como la banda depende de la segmentación ventricular, si esta falla en una visita la textura queda sin calcular (NaN).

### Resumen de la validación

| Validación | Resultado |
|---|---|
| Test-retest (error del método) | Tejido: 6,9 ml (0,55%). Ventrículos: 0,22 ml (0,48%) |
| Test-retest (textura) | Error típico (mediana de las diferencias, primera visita de los 4 pacientes): contraste 7,11%, homogeneidad 3,90%, entropía 1,13% |
| MRBrainS18 (7 sujetos) | Dice medio 0,84 (0,74-0,91); correlación de volumen r = 0,98; el método mide en promedio un 19,5% menos que la segmentación manual (diferencia sistemática, no afecta la comparación entre visitas) |
| Concordancia con OASIS | Cociente entre nuestra pendiente de nWBV y la de OASIS entre 1,02 y 1,56 (mismo sentido en los 4 pacientes) |

---

## Problemas frecuentes

| Problema | Solución |
|---|---|
| `FileNotFoundError: No se encontró el archivo: ...OAS2_RAW_PART1.tar.gz` | Los `.tar.gz` no están en `Dataset/`. Revisar con la salida de `diagnostico()` dónde se buscan, y descargarlos (paso 1) |
| `descargar_dataset.py` dice que faltan visitas | Poner `CORTAR_AL_TERMINAR = False` en `Funciones/Descarga.py` y volver a correrlo |
| La descarga con aria2 se cortó | Volver a correr el mismo comando `aria2c -c ...`: sigue desde donde quedó |
| `aria2c` no se reconoce como comando | Cerrar y abrir la terminal después de instalarlo con `winget` |
| `Aviso: No se encontró la carpeta de OAS2_XXXX_MRn` | Esa visita no se extrajo; volver a correr `python descargar_dataset.py` |
| Error al descargar el atlas Harvard-Oxford | Hace falta internet la primera vez que se corre `main.py`. Si queda una descarga a medias, borrar `Dataset/Atlas/` y volver a correr |
| `MRBrainS18: no se encontraron sujetos` | Es solo un aviso: falta `Dataset/MRBrainS18/` (paso 3). El resto se ejecuta igual |
| `PROCESAR_IMAGENES = False pero no existe Resultados/volumenes.csv` | Correr una vez con `PROCESAR_IMAGENES = True` |
| `ModuleNotFoundError: No module named 'PySide6'` | Instalar las dependencias de nuevo: `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'skimage'` | Falta scikit-image: `pip install -r requirements.txt` |
| La interfaz dice que no se encontraron pacientes | Falta extraer el dataset: correr `python descargar_dataset.py` (paso 2) |
| Al guardar el PDF desde la interfaz dice que no se pudo escribir | El PDF está abierto en otro programa: cerrarlo y volver a guardar |
| La textura de una visita sale NaN | La segmentación ventricular de esa visita falló y la banda quedó vacía; revisar la figura de los ventrículos de esa visita |
| Las columnas de textura no aparecen en `volumenes.csv` | El CSV es anterior a la textura: correr una vez con `PROCESAR_IMAGENES = True` |
| Los volúmenes cambian levemente entre corridas | Es el registro multihilo de SimpleITK (~1%); ver [Ejecución](#ejecución) |

---

## Referencias

- **OASIS-2:** Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684. https://doi.org/10.1162/jocn.2009.21407
- **Plantilla MNI152 2009a:** Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C., Collins, D. L., & the Brain Development Cooperative Group (2011). Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage*, 54(1), 313-327.
- **Corrección N4:** Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A., Yushkevich, P. A., & Gee, J. C. (2010). N4ITK: Improved N3 bias correction. *IEEE Transactions on Medical Imaging*, 29(6), 1310-1320.
- **Textura (GLCM):** Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610-621.
- **Atlas Harvard-Oxford:** distribuido con FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases), descargado mediante nilearn.
- **MRBrainS18:** Kuijf, H. J., et al. (2024). MR Brain Segmentation Challenge 2018 Data. DataverseNL. https://doi.org/10.34894/E0U32Q
