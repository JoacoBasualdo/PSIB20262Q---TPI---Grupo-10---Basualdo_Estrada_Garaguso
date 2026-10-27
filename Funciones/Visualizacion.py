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


# Igual que visualizar_planos_ortogonales, pero dibuja encima el contorno de una o más máscaras.
# Sirve para controlar a ojo si la segmentación sigue el borde real del cerebro.
def visualizar_mascaras(matriz_3d, mascaras, titulo="Segmentación", tamano_voxel=(1.0, 1.0, 1.25),
                        colores=("red", "cyan", "yellow"), nombres=None):
    dim_ap, dim_si, dim_lr = matriz_3d.shape
    paso_ap, paso_si, paso_lr = tamano_voxel

    def cortar(volumen):
        return [
            np.rot90(volumen[:, :, dim_lr // 2]),
            np.flipud(volumen[dim_ap // 2, :, :]),
            np.flipud(volumen[:, dim_si // 2, :]),
        ]

    nombres_planos = ["Plano Sagital", "Plano Coronal", "Plano Axial"]
    aspectos = [paso_si / paso_ap, paso_si / paso_lr, paso_ap / paso_lr]
    cortes_imagen = cortar(matriz_3d)
    cortes_mascaras = [cortar(m.astype(float)) for m in mascaras]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(titulo, fontsize=16)

    for i, eje in enumerate(ejes):
        eje.imshow(cortes_imagen[i], cmap='gray', aspect=aspectos[i])
        for cortes_m, color in zip(cortes_mascaras, colores):
            eje.contour(cortes_m[i], levels=[0.5], colors=color, linewidths=1)
        eje.set_title(nombres_planos[i])
        eje.axis('off')

    if nombres:
        from matplotlib.lines import Line2D
        lineas = [Line2D([0], [0], color=c, lw=2) for c in colores[:len(nombres)]]
        fig.legend(lineas, nombres, loc="lower center", ncol=len(nombres))

    plt.tight_layout()
    return fig



# Histograma de la cabeza con los centros de cada clase y el umbral elegido.
# Sirve para ver si el umbral cae en el valle entre lo oscuro y la sustancia gris.
def graficar_histograma_umbral(valores, umbral, centros, titulo="Umbral de tejido", bins=200):
    fig, eje = plt.subplots(figsize=(10, 5))
    eje.hist(valores, bins=bins, histtype='step', color='black')
    nombres = ["Oscuro", "Sust. gris", "Sust. blanca", "Grasa"]
    for centro, nombre in zip(centros, nombres):
        eje.axvline(centro, color='gray', linestyle=':')
        eje.text(centro, eje.get_ylim()[1] * 0.95, nombre, ha='center', fontsize=9)
    eje.axvline(umbral, color='red', linewidth=2, label=f"Umbral = {umbral:.3f}")
    eje.set_title(titulo)
    eje.set_xlabel("Intensidad normalizada")
    eje.set_ylabel("Cantidad de vóxeles")
    eje.legend()
    plt.tight_layout()
    return fig
