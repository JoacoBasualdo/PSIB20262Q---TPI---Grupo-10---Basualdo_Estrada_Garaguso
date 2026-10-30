# PSIB2026Q2-TPI-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Imágenes Biomédicas (TPI) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

## Descripción del Proyecto
Cuantificación de la atrofia cerebral asociada a la Enfermedad de Alzheimer a partir de resonancias magnéticas T1 longitudinales del dataset OASIS-2. Para cada paciente se mide, en cada visita, el volumen de tejido cerebral y el volumen de los ventrículos laterales, y se analiza su evolución en el tiempo.

Pacientes analizados: OAS2_0070 (sano), OAS2_0048 y OAS2_0087 (demencia) y OAS2_0127 (converted, comienza sano y desarrolla demencia).

## Requisitos y Dependencias
Para ejecutar el código de este proyecto, es necesario contar con Python instalado. Todas las librerías y dependencias necesarias se encuentran listadas en el archivo `requirements.txt`.

## Instrucciones de Instalación
```bash
pip install -r requirements.txt
```

La primera vez que se corre `main.py` se descarga automáticamente el atlas Harvard-Oxford (usado para los ventrículos) en `Dataset/Atlas/`. Hace falta conexión a internet solo esa vez.

## Datos

Los datos no se suben al repositorio (`Dataset/` está en el `.gitignore`).

### OASIS-2 (datos principales)

Fuente: https://sites.wustl.edu/oasisbrains/home/oasis-2/

1. Instalar aria2: `winget install aria2.aria2` y abrir una terminal nueva.
2. Desde la carpeta del repo, descargar las dos partes (se pueden lanzar en paralelo):
   ```powershell
   aria2c -c -x 16 -s 16 -k 1M -d Dataset https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz
   aria2c -c -x 16 -s 16 -k 1M -d Dataset https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz
   ```
3. Extraer los pacientes usados: `python descargar_dataset.py` (debe terminar con "Dataset listo.").

### MRBrainS18 (solo para validar los ventrículos)

1. Entrar a https://mrbrains18.isi.uu.nl/data y seguir el enlace al dataset en DataverseNL.
2. Leer la pestaña **Terms** (condiciones de uso y cita requerida).
3. Descargar **`training.zip`** (471 MB, 7 sujetos con segmentación manual). `test.zip` no sirve.
4. Descomprimirlo en `Dataset/MRBrainS18/` (una carpeta por sujeto, cada una con su `segm.nii.gz`).

## Estructura del código

`main.py` organiza todo el flujo y llama a las funciones de la carpeta `Funciones/`, un archivo por etapa:

| Módulo | Qué hace |
|---|---|
| `Lectura.py` | Carga de las resonancias, tamaño de vóxel, carpetas de cada visita y Excel clínico (y los datos de MRBrainS18) |
| `Procesamiento.py` | Promediado de las adquisiciones de cada sesión, corrección de bias field (N4) y normalización de intensidad |
| `Features.py` | Extracción del cerebro (atlas MNI152 + registro entre visitas), ventrículos laterales (atlas Harvard-Oxford) y fracción ventricular |
| `Interpretacion.py` | Evolución del cerebro y los ventrículos entre visitas, y relación con CDR y MMSE |
| `Validacion.py` | Concordancia con OASIS, test-retest y comparación con MRBrainS18 |
| `Visualizacion.py` | Todas las figuras |
| `Reporte.py` | PDF por paciente y máscaras ventriculares en NIfTI |
| `Descarga.py` | Descarga y extracción del dataset OASIS-2 |

## Ejecución

Desde la carpeta raíz del repositorio:

```bash
python descargar_dataset.py   # una sola vez, para extraer los datos (ver "Datos")
python main.py                # procesamiento, interpretación y validación
```

Al principio de `main.py` se elige qué partes correr:

| Opción | Qué hace | Tiempo aproximado |
|---|---|---|
| `PROCESAR_IMAGENES` | Procesa las resonancias de los 4 pacientes. Con `False` usa el `Resultados/volumenes.csv` ya generado y solo rehace interpretación y validación | ~30-40 min |
| `VALIDAR_CONCORDANCIA_OASIS` | Compara nuestra tasa de atrofia con la del nWBV publicado por OASIS | segundos |
| `VALIDAR_TEST_RETEST` | Mide el error del método comparando dos adquisiciones de la misma sesión (`VISITAS_TEST_RETEST` visitas por paciente) | ~2-3 min por visita |
| `VALIDAR_MRBRAINS` | Compara nuestros ventrículos con la segmentación manual de MRBrainS18 (se saltea si no está `Dataset/MRBrainS18/`) | ~1-2 min por sujeto |

## Resultados

Todo queda en `Resultados/`:

| Archivo | Contenido |
|---|---|
| `OAS2_XXXX_visitas.pdf` | Reporte de cada paciente: salida de la terminal y figuras de cada etapa |
| `volumenes.csv` | Una fila por visita: datos clínicos, volumen de tejido, zona intracraneal, nWBV propio, ventrículos (ml) y fracción ventricular |
| `comparacion_volumenes.csv`, `resumen_longitudinal.csv` | Evolución del volumen cerebral |
| `comparacion_ventriculos.csv` | Evolución de los ventrículos (ml, % y fracción ventricular) |
| `analisis_clinico_integrado.csv`, `resumen_clinico_integrado.csv` | Cambios morfológicos junto con CDR y MMSE |
| `concordancia_oasis.csv`, `test_retest.csv`, `validacion_mrbrains.csv` | Validación del método |
| `Graficos/` | Gráficos de evolución, fracción ventricular, comparación anatómica MR1/última visita y validación |
| `MascarasVentriculares/` | Máscara 3D de los ventrículos de cada visita (`.nii.gz`) |

**Fracción ventricular:** volumen ventricular dividido por el volumen de la zona intracraneal (en %). La zona intracraneal es la misma región en todas las visitas de un paciente, así que dentro de un paciente la fracción cambia solo si cambian los ventrículos; entre pacientes, quita el efecto del tamaño de la cabeza.