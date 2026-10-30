# =============================================================================
# VALIDACIÓN
# =============================================================================
# Tres formas de comprobar que el método mide bien:
#
#   1. Concordancia con OASIS (volumen cerebral): nuestro nWBV contra el nWBV publicado por
#      OASIS, calculado por los autores del dataset con su propio método automático.
#      Se compara la tasa de atrofia (pendiente por año) de cada paciente.
#
#   2. Test-retest (precisión, cerebro y ventrículos): en cada visita OASIS adquirió 3-4
#      resonancias seguidas. En minutos el cerebro no cambia, así que la diferencia entre
#      medir mpr-1 y mpr-2 por separado es el error del método. Se usa el mismo
#      procedimiento que entre visitas (mpr-1 como "primera visita", mpr-2 como "siguiente").
#      Se comparan adquisiciones sueltas (más ruido que el promedio), así que el error que
#      da es una cota superior del error real.
#
#   3. MRBrainS18 (exactitud de los ventrículos): OASIS no publica ventrículos. MRBrainS18 son
#      resonancias de pacientes mayores de 50 años con segmentación MANUAL de expertos. Se
#      corre nuestro procesamiento sobre su T1 de 1 mm y se compara contra la etiqueta manual
#      de ventrículos (volumen y coeficiente Dice = 2|A∩B| / (|A|+|B|), 1 = idénticas).
#      OJO: la etiqueta manual incluye TODOS los ventrículos (laterales con astas temporales,
#      tercero y cuarto); nuestro método mide solo los laterales.

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from nibabel.processing import resample_from_to

from Funciones.Lectura import (cargar_datos_clinicos, buscar_carpeta_visita, listar_adquisiciones,
                               leer_tamano_voxel, cargar_volumen_nifti, buscar_archivos_mrbrains,
                               cargar_t1_como_oasis)
from Funciones.Procesamiento import corregir_bias_field, normalizar_intensidad
from Funciones.Features import mascara_cerebro, mascara_cerebro_seguimiento, segmentar_ventriculos, volumen_ml
from Funciones.Visualizacion import graficar_validacion_mrbrains

ETIQUETA_VENTRICULOS_MRBRAINS = 6


# Pendiente de la recta que mejor ajusta los puntos (cambio por año)
def pendiente(anios, valores):
    return np.polyfit(anios, valores, 1)[0]


# =============================================================================
# 1. CONCORDANCIA CON OASIS
# =============================================================================

# Compara nuestro nWBV con el de OASIS en un paciente
def comparar_paciente_oasis(filas):
    filas = filas.sort_values("anios_desde_primera")
    anios = filas["anios_desde_primera"].values
    propio = filas["nwbv_propio"].values
    oasis = filas["nwbv_oasis"].values

    pendiente_propia = pendiente(anios, propio)
    pendiente_oasis = pendiente(anios, oasis)

    return {
        "paciente": filas["paciente"].iloc[0],
        "grupo": filas["grupo"].iloc[0],
        "visitas": len(filas),
        "anios_seguimiento": round(anios.max(), 2),
        # Tasa de atrofia: cambio de nWBV por año (negativo = pierde cerebro)
        "pendiente_nwbv_propio_anio": round(pendiente_propia, 4),
        "pendiente_nwbv_oasis_anio": round(pendiente_oasis, 4),
        # Cuántas veces la nuestra respecto de la de OASIS (1 = iguales)
        "relacion_pendientes": round(pendiente_propia / pendiente_oasis, 2) if pendiente_oasis != 0 else np.nan,
        # Lo mismo en % del valor inicial por año, para comparar pacientes entre sí
        "tasa_propia_pct_anio": round(100 * pendiente_propia / propio[0], 2),
        "tasa_oasis_pct_anio": round(100 * pendiente_oasis / oasis[0], 2),
        # Corrimiento promedio de nivel (propio - OASIS)
        "diferencia_media_nwbv": round(float(np.mean(propio - oasis)), 4),
        # Con 2 visitas la correlación siempre da ±1, así que no se informa
        "correlacion": round(float(np.corrcoef(propio, oasis)[0, 1]), 3) if len(filas) >= 3 else np.nan,
    }


# Tabla de concordancia de todos los pacientes, a partir de la tabla de volúmenes por visita.
# Se saltean los pacientes con menos de 2 visitas (no se puede calcular una pendiente).
def concordancia_oasis(datos):
    tabla = pd.DataFrame([comparar_paciente_oasis(filas) for _, filas in datos.groupby("paciente", sort=False)
                          if len(filas) >= 2])
    correlacion_global = float(np.corrcoef(datos["nwbv_propio"], datos["nwbv_oasis"])[0, 1])
    return tabla, correlacion_global


def imprimir_concordancia_oasis(tabla, correlacion_global):
    print("\n================ CONCORDANCIA CON OASIS (nWBV) ================")
    print(tabla.to_string(index=False))
    print(f"\nCorrelación con todas las visitas juntas: r = {correlacion_global:.3f}")
    print("(Incluye diferencias entre pacientes; para la tasa de atrofia mirar las pendientes por paciente.)")


# =============================================================================
# 2. TEST-RETEST
# =============================================================================

# Corrección de bias y normalización de UNA adquisición (igual que en main.py)
def preparar_adquisicion(ruta, tamano_voxel):
    volumen = cargar_volumen_nifti(ruta)
    corregida, _, mascara_cabeza = corregir_bias_field(volumen, tamano_voxel)
    normalizada, _, _ = normalizar_intensidad(corregida, mascara_cabeza)
    return normalizada, mascara_cabeza


# Mide cerebro y ventrículos en mpr-1 y en mpr-2 de una misma visita
def medir_test_retest_visita(carpeta_raw, mri_id):
    rutas = listar_adquisiciones(carpeta_raw)
    if len(rutas) < 2:
        print(f"  {mri_id}: tiene una sola adquisición, se saltea")
        return None

    tamano_voxel = leer_tamano_voxel(rutas[0])

    # mpr-1: se trata como si fuera la primera visita (atlas MNI152)
    normalizada_1, cabeza_1 = preparar_adquisicion(rutas[0], tamano_voxel)
    tejido_1, zona_1, umbral_1, _, transformada_atlas_1 = mascara_cerebro(normalizada_1, cabeza_1, tamano_voxel)
    ventriculos_1, info_1 = segmentar_ventriculos(
        normalizada_1, zona_1, tejido_1, umbral_1, tamano_voxel,
        transformada_atlas=transformada_atlas_1
    )

    # mpr-2: se trata como una visita posterior (registro rígido contra mpr-1)
    referencia = {
        "mri_id": f"{mri_id} mpr-1",
        "normalizada": normalizada_1,
        "intracraneal": zona_1,
        "tamano_voxel": tamano_voxel,
        "transformada_atlas": transformada_atlas_1,
    }
    normalizada_2, cabeza_2 = preparar_adquisicion(rutas[1], tamano_voxel)
    tejido_2, zona_2, umbral_2, _, (rot, tras, corr), transformada_2_a_1 = mascara_cerebro_seguimiento(
        normalizada_2, cabeza_2, tamano_voxel, referencia, devolver_transformada=True
    )
    transformada_atlas_2 = sitk.CompositeTransform([transformada_atlas_1, transformada_2_a_1])
    ventriculos_2, info_2 = segmentar_ventriculos(
        normalizada_2, zona_2, tejido_2, umbral_2, tamano_voxel,
        transformada_atlas=transformada_atlas_2
    )

    return {
        "mri_id": mri_id,
        "tejido_mpr1_ml": round(volumen_ml(tejido_1, tamano_voxel), 1),
        "tejido_mpr2_ml": round(volumen_ml(tejido_2, tamano_voxel), 1),
        "ventriculos_mpr1_ml": round(info_1["volumen_ml"], 2),
        "ventriculos_mpr2_ml": round(info_2["volumen_ml"], 2),
        "correlacion_registro": round(corr, 4),
    }


# Corre el test-retest sobre las primeras visitas de cada paciente.
# ruta_csv: si se da, guarda la tabla después de cada visita (por si algo falla a mitad de camino).
def test_retest(pacientes, ruta_dataset, ruta_datos_clinicos, visitas_por_paciente=2, ruta_csv=None):
    filas = []
    for paciente_id in pacientes:
        datos = cargar_datos_clinicos(ruta_datos_clinicos, paciente_id)
        for mri_id in datos["MRI ID"].head(visitas_por_paciente):
            carpeta_raw = buscar_carpeta_visita(ruta_dataset, mri_id)
            if carpeta_raw is None:
                print(f"  {mri_id}: no se encontró la carpeta, se saltea")
                continue
            print(f"\n### Test-retest {mri_id}")
            fila = medir_test_retest_visita(carpeta_raw, mri_id)
            if fila is None:
                continue
            fila["paciente"] = paciente_id
            filas.append(fila)
            if ruta_csv:
                pd.DataFrame(filas).to_csv(ruta_csv, index=False)

    if not filas:
        return None

    tabla = pd.DataFrame(filas)
    tabla["dif_tejido_ml"] = (tabla["tejido_mpr2_ml"] - tabla["tejido_mpr1_ml"]).round(1)
    tabla["dif_tejido_pct"] = (100 * tabla["dif_tejido_ml"] / tabla["tejido_mpr1_ml"]).round(2)
    tabla["dif_ventriculos_ml"] = (tabla["ventriculos_mpr2_ml"] - tabla["ventriculos_mpr1_ml"]).round(2)
    tabla["dif_ventriculos_pct"] = (100 * tabla["dif_ventriculos_ml"] / tabla["ventriculos_mpr1_ml"]).round(2)
    columnas = ["paciente", "mri_id", "tejido_mpr1_ml", "tejido_mpr2_ml", "dif_tejido_ml", "dif_tejido_pct",
                "ventriculos_mpr1_ml", "ventriculos_mpr2_ml", "dif_ventriculos_ml", "dif_ventriculos_pct",
                "correlacion_registro"]
    tabla = tabla[columnas]
    if ruta_csv:
        tabla.to_csv(ruta_csv, index=False)
    return tabla


# Resumen del test-retest: error típico y comparación con el cambio de cada paciente
# (datos_visitas = tabla de volúmenes por visita que arma main.py)
def imprimir_test_retest(tabla, datos_visitas):
    print("\n================ TEST-RETEST (mpr-1 vs mpr-2, misma sesión) ================")
    print(tabla.to_string(index=False))

    # Error típico: mediana de |diferencia| (no se deja arrastrar por una adquisición mala);
    # también se muestra el promedio
    error_tejido = tabla["dif_tejido_ml"].abs().median()
    error_ventriculos = tabla["dif_ventriculos_ml"].abs().median()
    print("\nError típico (mediana de |diferencia|; entre paréntesis, el promedio):")
    print(f"  Tejido cerebral: {error_tejido:.1f} ml = {tabla['dif_tejido_pct'].abs().median():.2f}% "
          f"(promedio {tabla['dif_tejido_ml'].abs().mean():.1f} ml)")
    print(f"  Ventrículos:     {error_ventriculos:.2f} ml = {tabla['dif_ventriculos_pct'].abs().median():.2f}% "
          f"(promedio {tabla['dif_ventriculos_ml'].abs().mean():.2f} ml)")

    print("\nCambio acumulado en todo el seguimiento (pendiente x años) comparado con el error:")
    print("  'veces el error' > 2-3 -> el cambio no se explica por ruido de medición.")
    for paciente, filas in datos_visitas.groupby("paciente", sort=False):
        if len(filas) < 2:
            continue
        anios = filas["anios_desde_primera"].values
        cambio_tejido = pendiente(anios, filas["volumen_tejido_ml"].values) * anios.max()
        cambio_ventriculos = pendiente(anios, filas["volumen_ventricular_ml"].values) * anios.max()
        print(f"  {paciente} ({anios.max():.1f} años): "
              f"tejido {cambio_tejido:+.1f} ml = {abs(cambio_tejido) / error_tejido:.1f} veces el error | "
              f"ventrículos {cambio_ventriculos:+.1f} ml = {abs(cambio_ventriculos) / error_ventriculos:.1f} veces el error")


# =============================================================================
# 3. MRBrainS18
# =============================================================================

# Lleva una máscara nuestra (grilla de la T1 de 1 mm) a la grilla de la segmentación manual
def a_grilla_manual(mascara, affine_mascara, imagen_manual):
    origen = nib.Nifti1Image(mascara.astype(np.float32), affine_mascara)
    remuestreada = resample_from_to(origen, imagen_manual, order=1)
    return np.asarray(remuestreada.dataobj) > 0.5


def dice(a, b):
    total = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / total if total > 0 else np.nan


def validar_sujeto_mrbrains(carpeta_sujeto, carpeta_figuras):
    sujeto = os.path.basename(os.path.normpath(carpeta_sujeto))
    ruta_segm, ruta_t1 = buscar_archivos_mrbrains(carpeta_sujeto)
    if ruta_segm is None:
        print(f"  {sujeto}: no se encontraron segm.nii.gz y una T1 de 1 mm, se saltea")
        return None
    print(f"\n### MRBrainS18 sujeto {sujeto}  (T1: {ruta_t1})")

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
    ventriculos_manual = etiquetas == ETIQUETA_VENTRICULOS_MRBRAINS

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
    graficar_validacion_mrbrains(fondo, ventriculos_manual, ventriculos_en_manual, sujeto,
                                 os.path.join(carpeta_figuras, f"ventriculos_sujeto_{sujeto}.png"))
    return fila


# Valida todos los sujetos que haya en ruta_mrbrains (cada uno con su segm.nii.gz).
# Devuelve None si no hay datos de MRBrainS18.
def validacion_mrbrains(ruta_mrbrains, carpeta_figuras, ruta_csv=None):
    carpetas = sorted(c for c in glob.glob(os.path.join(ruta_mrbrains, "**", ""), recursive=True)
                      if os.path.isfile(os.path.join(c, "segm.nii.gz")))
    if not carpetas:
        return None

    filas = []
    for carpeta in carpetas:
        fila = validar_sujeto_mrbrains(carpeta, carpeta_figuras)
        if fila is not None:
            filas.append(fila)
            if ruta_csv:
                pd.DataFrame(filas).to_csv(ruta_csv, index=False)
    return pd.DataFrame(filas) if filas else None


def imprimir_validacion_mrbrains(tabla):
    print("\n========== VALIDACIÓN DE VENTRÍCULOS CONTRA SEGMENTACIÓN MANUAL (MRBrainS18) ==========")
    print(tabla.to_string(index=False))
    print(f"\nVentrículos: Dice medio {tabla['dice_ventriculos'].mean():.3f} | "
          f"error de volumen medio {tabla['error_ventriculos_pct'].mean():+.1f}% "
          f"(en valor absoluto {tabla['error_ventriculos_pct'].abs().mean():.1f}%)")
    if len(tabla) >= 3:
        r = np.corrcoef(tabla["ventriculos_manual_ml"], tabla["ventriculos_propio_ml"])[0, 1]
        print(f"Correlación entre volumen manual y nuestro: r = {r:.3f}")
