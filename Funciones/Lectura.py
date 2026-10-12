import os
import nibabel as nib
import numpy as np


# Esta función lo que hace es agarrar el erchivo de la resonancia y armar una matriz matemática 3D donde cada número representa un nivel de intensidad.
def cargar_volumen_nifti(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    
    volumen = nib.load(ruta)
    matriz_3d = volumen.get_fdata()
    
    # Esto elimina cualquier dimensión que sea "1" (pasa de 4D a 3D puro)
    matriz_3d = np.squeeze(matriz_3d)
    
    return matriz_3d