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
    datos = pd.read_excel(ruta_xlsx)
    datos_paciente = datos[datos["Subject ID"] == paciente_id]
    return datos_paciente.sort_values("Visit").reset_index(drop=True)