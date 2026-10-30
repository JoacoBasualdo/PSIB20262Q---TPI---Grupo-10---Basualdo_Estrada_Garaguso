# =============================================================================
# LECTURA
# =============================================================================
# Carga de datos desde el disco:
#   - OASIS-2: volúmenes Analyze/NIfTI de cada adquisición, tamaño de vóxel, carpetas de
#     cada visita y el Excel clínico.
#   - MRBrainS18 (validación): T1 de 1 mm y segmentación manual de cada sujeto.

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

# Esta función lo que hace es agarrar el erchivo de la resonancia y armar una matriz matemática 3D donde cada número representa un nivel de intensidad.
def cargar_volumen_nifti(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    
    volumen = nib.load(ruta)
    matriz_3d = volumen.get_fdata()
    
    # Esto elimina cualquier dimensión que sea "1" (pasa de 4D a 3D puro)
    matriz_3d = np.squeeze(matriz_3d)
    
    return matriz_3d

# Devuelve el tamaño del voxel en milimetros. El voxel indica cuánto espacio real del cerebro representa cada celda lo cual es fundamental para la cuantificación. Esa data se encuentra dentro del header.
def leer_tamano_voxel(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
 
    volumen = nib.load(ruta)
    return tuple(float(z) for z in volumen.header.get_zooms()[:3])
 
# Se fija cuantos archivos hay y los ordena para que mpr-1 quede siempre primera.
def listar_adquisiciones(carpeta_raw):
    return sorted(glob.glob(os.path.join(carpeta_raw, "mpr-*.nifti.img")))
 
# Se lee el Excel de OASIS y se devuelve solo las filas del paciente pedido. Compara cada fila contra el ID del paciente y genera una columna de true/false
def cargar_datos_clinicos(ruta_xlsx, paciente_id):
    datos = pd.read_excel(ruta_xlsx)
    datos_paciente = datos[datos["Subject ID"] == paciente_id]
    return datos_paciente.sort_values("Visit").reset_index(drop=True)
 
# Busca las carpetas en el las carpetas del data set. Tanto en la parte 1 como en la parte 2
def buscar_carpeta_visita(ruta_dataset, mri_id):
    coincidencias = glob.glob(os.path.join(ruta_dataset, "OAS2_RAW_PART*", mri_id, "RAW"))
    if not coincidencias:
        return None
    return coincidencias[0]


# Busca, dentro de la carpeta de un sujeto, la segmentación manual y la T1 3D de 1 mm.
# Se prefiere la T1 original (carpeta "orig"), sin corregir, porque nuestro procesamiento
# hace su propia corrección de bias (N4), igual que con OASIS.
def buscar_archivos_mrbrains(carpeta_sujeto):
    segmentacion = glob.glob(os.path.join(carpeta_sujeto, "**", "segm.nii.gz"), recursive=True)
    candidatas = []
    for ruta in glob.glob(os.path.join(carpeta_sujeto, "**", "T1.nii.gz"), recursive=True):
        zooms = nib.load(ruta).header.get_zooms()[:3]
        if max(zooms) < 1.5:  # la T1 3D de 1 mm (las otras imágenes tienen cortes de 3 mm)
            candidatas.append(ruta)
    if not segmentacion or not candidatas:
        return None, None
    candidatas.sort(key=lambda r: 0 if "orig" in r.split(os.sep) else 1)
    return segmentacion[0], candidatas[0]


# Carga la T1 y la lleva a la orientación que usa nuestro código (la de OASIS):
# ejes (posterior->anterior, inferior->superior, izquierda-derecha).
# Devuelve también la "affine" (posición en el espacio físico) de ese arreglo, para después
# poder llevar nuestras máscaras a la grilla de la segmentación manual.
def cargar_t1_como_oasis(ruta):
    imagen = nib.as_closest_canonical(nib.load(ruta))          # ejes (izq->der, post->ant, inf->sup)
    datos = np.asarray(imagen.dataobj, dtype=np.float32)
    datos = np.transpose(datos, (1, 2, 0))                      # -> (post->ant, inf->sup, izq->der)
    zooms = imagen.header.get_zooms()[:3]
    tamano_voxel = (float(zooms[1]), float(zooms[2]), float(zooms[0]))
    affine = imagen.affine[:, [1, 2, 0, 3]]                     # mismo reordenamiento de ejes
    return datos, tamano_voxel, affine
