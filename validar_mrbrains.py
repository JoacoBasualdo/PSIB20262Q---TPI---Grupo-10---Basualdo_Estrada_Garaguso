# =============================================================================
# VALIDACIÓN DE LOS VENTRÍCULOS CONTRA SEGMENTACIONES MANUALES (MRBrainS18)
# =============================================================================
#
# OASIS no publica volúmenes de ventrículos, así que no hay contra qué comparar si
# nuestras máscaras están bien ubicadas. MRBrainS18 sí: son resonancias de pacientes
# mayores de 50 años (con demencia, Alzheimer, diabetes y controles), con atrofia, en las
# que expertos segmentaron A MANO cada estructura, incluidos los ventrículos.
#
# Para cada sujeto se corre exactamente nuestro procesamiento (el mismo de main.py) sobre
# su T1 de 1 mm y se comparan los VENTRÍCULOS contra la segmentación manual.
# (El volumen de tejido cerebral se valida aparte, contra el nWBV de OASIS: en estas
# imágenes de 3 T el atlas MNI puede quedar agrandado y el borde exterior del cerebro
# no es confiable; los ventrículos, al estar en el centro, casi no se ven afectados.)
#
#   - Volumen (ml): nuestro vs. manual, y el error en %. Se mide en la grilla de la
#     segmentación manual (48 cortes de 3 mm), que NO cubre toda la cabeza: así se compara
#     exactamente la misma región.
#   - Coeficiente Dice: cuánto se superponen las dos máscaras.
#       Dice = 2 x |A ∩ B| / (|A| + |B|)      1 = idénticas, 0 = no se tocan.
#       Como referencia, en la literatura un Dice > 0.8 en ventrículos se considera bueno.
#
# Estructura comparada (etiqueta de la segmentación manual, ver readme.txt):
#   - Ventrículos: etiqueta 6. OJO: la etiqueta manual incluye TODOS los ventrículos
#     (laterales con sus astas temporales, tercero y cuarto), mientras que
#     SegmentacionVentricular.py mide solo los ventrículos LATERALES. Parte de la diferencia
#     de volumen es por esta diferencia de definición (ver la figura de control).
#
# Datos: MRBrainS18, archivo "training.zip" en DataverseNL (https://mrbrains18.isi.uu.nl/data).
#   Antes de usarlos leer la pestaña "Terms" del dataset y citar el trabajo que indica.
#
# Uso (desde la carpeta del repo):
#   1. Descomprimir training.zip en Dataset/MRBrainS18/ (una carpeta por sujeto).
#   2. python validar_mrbrains.py
# Resultado: Resultados/validacion_mrbrains.csv y figuras en Resultados/Graficos/ValidacionMRBrainS/

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to

from Funciones.Acondicionamiento import corregir_bias_field, normalizar_intensidad
from Funciones.Features import mascara_cerebro, volumen_ml
from Funciones.SegmentacionVentricular import segmentar_ventriculos

RUTA_MRBRAINS = os.path.join("Dataset", "MRBrainS18")
RUTA_RESULTADOS = "Resultados"
RUTA_CSV = os.path.join(RUTA_RESULTADOS, "validacion_mrbrains.csv")
CARPETA_FIGURAS = os.path.join(RUTA_RESULTADOS, "Graficos", "ValidacionMRBrainS")

ETIQUETA_VENTRICULOS = 6


# Busca, dentro de la carpeta de un sujeto, la segmentación manual y la T1 3D de 1 mm.
# Se prefiere la T1 original (carpeta "orig"), sin corregir, porque nuestro procesamiento
# hace su propia corrección de bias (N4), igual que con OASIS.
def buscar_archivos(carpeta_sujeto):
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


# Lleva una máscara nuestra (grilla de la T1 de 1 mm) a la grilla de la segmentación manual
def a_grilla_manual(mascara, affine_mascara, imagen_manual):
    origen = nib.Nifti1Image(mascara.astype(np.float32), affine_mascara)
    remuestreada = resample_from_to(origen, imagen_manual, order=1)
    return np.asarray(remuestreada.dataobj) > 0.5


def dice(a, b):
    total = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / total if total > 0 else np.nan


# Figura de control: 3 cortes axiales con el contorno manual (verde) y el nuestro (rojo)
def figura_control(t1_manual, ventriculos_manual, ventriculos_propios, sujeto, ruta):
    cortes_con_ventriculos = np.nonzero(ventriculos_manual.any(axis=(0, 1)))[0]
    if len(cortes_con_ventriculos) == 0:
        return
    cortes = np.linspace(cortes_con_ventriculos[0], cortes_con_ventriculos[-1], 5)[1:4].astype(int)
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MRBrainS18 sujeto {sujeto} - ventrículos: manual (verde) vs. nuestro (rojo)", fontsize=14)
    for eje, z in zip(ejes, cortes):
        eje.imshow(np.rot90(t1_manual[:, :, z]), cmap="gray")
        eje.contour(np.rot90(ventriculos_manual[:, :, z].astype(float)), [0.5], colors="lime", linewidths=1.2)
        eje.contour(np.rot90(ventriculos_propios[:, :, z].astype(float)), [0.5], colors="red", linewidths=1.2)
        eje.set_title(f"Corte {z}")
        eje.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)


def validar_sujeto(carpeta_sujeto):
    sujeto = os.path.basename(os.path.normpath(carpeta_sujeto))
    ruta_segm, ruta_t1 = buscar_archivos(carpeta_sujeto)
    if ruta_segm is None:
        print(f"  {sujeto}: no se encontraron segm.nii.gz y una T1 de 1 mm, se saltea")
        return None
    print(f"\n### Sujeto {sujeto}  (T1: {ruta_t1})")

    # ---- Nuestro procesamiento (igual que main.py, con una sola adquisición) ----
    t1, tamano_voxel, affine = cargar_t1_como_oasis(ruta_t1)
    corregida, _, mascara_cabeza = corregir_bias_field(t1, tamano_voxel)
    normalizada, _, _ = normalizar_intensidad(corregida, mascara_cabeza)
    tejido, zona, umbral, _, transformada_atlas = mascara_cerebro(normalizada, mascara_cabeza, tamano_voxel)
    ventriculos, info = segmentar_ventriculos(
        normalizada, zona, tejido, umbral, tamano_voxel, transformada_atlas=transformada_atlas
    )

    # ---- Referencia manual ----
    imagen_manual = nib.load(ruta_segm)
    etiquetas = np.asarray(imagen_manual.dataobj).astype(int)
    voxel_manual = imagen_manual.header.get_zooms()[:3]
    ventriculos_manual = etiquetas == ETIQUETA_VENTRICULOS

    # ---- Comparación en la grilla de la segmentación manual ----
    ventriculos_en_manual = a_grilla_manual(ventriculos, affine, imagen_manual)
    zona_en_manual = a_grilla_manual(zona, affine, imagen_manual)
    intracraneal_manual = etiquetas > 0   # todo lo etiquetado: tejido + LCR + ventrículos

    fila = {
        "sujeto": sujeto,
        "ventriculos_manual_ml": round(volumen_ml(ventriculos_manual, voxel_manual), 2),
        "ventriculos_propio_ml": round(volumen_ml(ventriculos_en_manual, voxel_manual), 2),
        "dice_ventriculos": round(dice(ventriculos_en_manual, ventriculos_manual), 3),
        "estado_ventricular": info.get("estado", ""),
        # Control del registro del atlas: la zona del cerebro no debería ser mucho más grande
        # que el interior del cráneo marcado a mano. Si da > 1.3, el atlas quedó agrandado
        # (sobre el cráneo/cuero cabelludo); conviene mirar la figura de ese sujeto.
        "zona_atlas_vs_intracraneal": round(zona_en_manual.sum() / max(intracraneal_manual.sum(), 1), 2),
    }
    fila["error_ventriculos_pct"] = round(100 * (fila["ventriculos_propio_ml"] / fila["ventriculos_manual_ml"] - 1), 1)
    print(f"  Ventrículos: manual {fila['ventriculos_manual_ml']} ml | nuestro {fila['ventriculos_propio_ml']} ml "
          f"| error {fila['error_ventriculos_pct']:+.1f}% | Dice {fila['dice_ventriculos']}")
    if fila["zona_atlas_vs_intracraneal"] > 1.3:
        print(f"  Nota: la zona del atlas mide {fila['zona_atlas_vs_intracraneal']} veces el interior del cráneo "
              "marcado a mano (atlas agrandado). Revisar la figura de control de este sujeto.")

    # Figura de control sobre la T1 registrada a la grilla manual (si está), si no la propia segmentación
    t1_reg = glob.glob(os.path.join(carpeta_sujeto, "**", "reg_T1.nii.gz"), recursive=True)
    fondo = np.asarray(nib.load(t1_reg[0]).dataobj) if t1_reg else etiquetas.astype(float)
    figura_control(fondo, ventriculos_manual, ventriculos_en_manual, sujeto,
                   os.path.join(CARPETA_FIGURAS, f"ventriculos_sujeto_{sujeto}.png"))
    return fila


def main():
    carpetas = sorted(c for c in glob.glob(os.path.join(RUTA_MRBRAINS, "**", ""), recursive=True)
                      if os.path.isfile(os.path.join(c, "segm.nii.gz")))
    if not carpetas:
        print(f"No se encontraron sujetos en {RUTA_MRBRAINS} (cada uno debe tener su segm.nii.gz).")
        return

    filas = []
    for carpeta in carpetas:
        fila = validar_sujeto(carpeta)
        if fila is not None:
            filas.append(fila)
            os.makedirs(RUTA_RESULTADOS, exist_ok=True)
            pd.DataFrame(filas).to_csv(RUTA_CSV, index=False)

    tabla = pd.DataFrame(filas)
    print("\n========== VALIDACIÓN DE VENTRÍCULOS CONTRA SEGMENTACIÓN MANUAL (MRBrainS18) ==========")
    print(tabla.to_string(index=False))
    print(f"\nVentrículos: Dice medio {tabla['dice_ventriculos'].mean():.3f} | "
          f"error de volumen medio {tabla['error_ventriculos_pct'].mean():+.1f}% "
          f"(en valor absoluto {tabla['error_ventriculos_pct'].abs().mean():.1f}%)")
    print(f"\nTabla guardada en: {RUTA_CSV}")
    print(f"Figuras de control en: {CARPETA_FIGURAS}")


if __name__ == "__main__":
    main()
