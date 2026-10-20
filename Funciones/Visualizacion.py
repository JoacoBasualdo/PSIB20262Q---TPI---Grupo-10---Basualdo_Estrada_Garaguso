import matplotlib.pyplot as plt
import numpy as np


# Realiza un corte a la mitad en cada planto y luego plotea eso pasando las intensidades a escala de grises
def visualizar_planos_ortogonales(matriz_3d, titulo="Cortes Centrales", tamano_voxel=(1.0, 1.0, 1.25)):
    dim_ap, dim_si, dim_lr = matriz_3d.shape
    paso_ap, paso_si, paso_lr = tamano_voxel
 
    corte_sagital = np.rot90(matriz_3d[:, :, dim_lr // 2])
    corte_coronal = np.flipud(matriz_3d[dim_ap // 2, :, :])
    corte_axial = np.flipud(matriz_3d[:, dim_si // 2, :])
 
    cortes = [
        ("Plano Sagital", corte_sagital, paso_si / paso_ap),
        ("Plano Coronal", corte_coronal, paso_si / paso_lr),
        ("Plano Axial", corte_axial, paso_ap / paso_lr),
    ]
 
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(titulo, fontsize=16)
 
    for eje, (nombre, corte, aspecto) in zip(ejes, cortes):
        eje.imshow(corte, cmap='gray', aspect=aspecto)
        eje.set_title(nombre)
        eje.axis('off')
 
    plt.tight_layout()
    return fig
 
 
 
# Muestra cuántos píxeles hay de cada brillo antes y después de normalizar.   
def graficar_histogramas_visitas(valores_por_visita, nombres, titulo="Histogramas por visita", bins=200):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo, fontsize=16)
 
    for (valores_antes, valores_despues), nombre in zip(valores_por_visita, nombres):
        ejes[0].hist(valores_antes, bins=bins, histtype='step', label=nombre)
        ejes[1].hist(valores_despues, bins=bins, histtype='step', label=nombre)
 
    ejes[0].set_title("Antes de normalizar")
    ejes[1].set_title("Después de normalizar")
    for eje in ejes:
        eje.set_xlabel("Intensidad")
        eje.set_ylabel("Cantidad de vóxeles")
        eje.legend()
 
    plt.tight_layout()
    return fig