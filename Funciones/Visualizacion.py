import matplotlib.pyplot as plt
import numpy as np


# Realiza un corte a la mitad en cada planto y luego plotea eso pasando las intensidades a escala de grises
def visualizar_planos_ortogonales(matriz_3d, titulo="Cortes Centrales"):
    dim_x, dim_y, dim_z = matriz_3d.shape
    
    corte_sagital = matriz_3d[dim_x // 2, :, :]
    corte_coronal = matriz_3d[:, dim_y // 2, :]
    corte_axial = matriz_3d[:, :, dim_z // 2]
    
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(titulo, fontsize=16)
    
    ejes[0].imshow(np.rot90(corte_sagital), cmap='gray')
    ejes[0].set_title("Plano Sagital")
    ejes[0].axis('off')
    
    ejes[1].imshow(np.rot90(corte_coronal), cmap='gray')
    ejes[1].set_title("Plano Coronal")
    ejes[1].axis('off')
    
    ejes[2].imshow(np.rot90(corte_axial), cmap='gray')
    ejes[2].set_title("Plano Axial")
    ejes[2].axis('off')
    
    plt.tight_layout()
    plt.show()