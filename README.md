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

## Ejecución

Desde la carpeta raíz del repositorio:

| Script | Qué hace | Resultado |
|---|---|---|
| `python main.py` | Procesa los 4 pacientes: acondicionamiento, extracción del cerebro, ventrículos y análisis longitudinal/clínico. | PDFs por paciente, `Resultados/volumenes.csv`, tablas y gráficos |
| `python test_retest.py` | Mide el error del método comparando dos adquisiciones de la misma sesión. Correr antes `main.py`. | `Resultados/test_retest.csv` |
| `python validar_mrbrains.py` | Compara nuestros ventrículos con la segmentación manual de MRBrainS18 (volumen y Dice). | `Resultados/validacion_mrbrains.csv` y figuras |
| `python validar_oasis.py` | Compara nuestra tasa de atrofia con la del nWBV publicado por OASIS. Correr antes `main.py`. | `Resultados/concordancia_oasis.csv` y `Resultados/Graficos/concordancia_oasis.png` |