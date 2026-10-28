# =============================================================================
# TEST-RETEST: ¿CUÁNTO ERROR TIENE NUESTRA MEDICIÓN?
# =============================================================================
#
# En cada visita, OASIS adquirió 3 o 4 resonancias T1 seguidas (mpr-1, mpr-2, ...),
# con minutos de diferencia. En minutos el cerebro no cambia, así que si medimos el
# volumen cerebral y ventricular en mpr-1 y en mpr-2 POR SEPARADO, cualquier diferencia
# entre las dos mediciones es error del método (ruido, registro, segmentación).
#
# Ese error se compara con el cambio que medimos entre visitas: si el cambio a lo largo
# de los años es mucho mayor que el error, el cambio es real y no ruido de medición.
#
# Se usa exactamente el mismo procedimiento que main.py usa entre visitas:
#   - mpr-1: extracción del cerebro con el atlas MNI152 y ventrículos (como una "MR1").
#   - mpr-2: se registra contra mpr-1 y se mide dentro de la misma zona (como una "MR2").
#
# Nota: se comparan adquisiciones SUELTAS (sin promediar), que tienen más ruido que el
# promedio que usa main.py. Por eso el error que da este script es una cota superior:
# el error real de main.py debería ser igual o menor.
#
# Uso (desde la carpeta del repo, con el dataset descargado):
#   python test_retest.py
# Resultado: Resultados/test_retest.csv y un resumen en la terminal.

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from Funciones.Lectura import cargar_datos_clinicos, buscar_carpeta_visita, listar_adquisiciones, leer_tamano_voxel, cargar_volumen_nifti
from Funciones.Acondicionamiento import corregir_bias_field, normalizar_intensidad
from Funciones.Features import mascara_cerebro, mascara_cerebro_seguimiento, volumen_ml
from Funciones.SegmentacionVentricular import segmentar_ventriculos

RUTA_DATASET = "Dataset"
RUTA_DATOS_CLINICOS = os.path.join(RUTA_DATASET, "oasis_longitudinal_demographics.xlsx")
RUTA_RESULTADOS = "Resultados"
RUTA_CSV_VISITAS = os.path.join(RUTA_RESULTADOS, "volumenes.csv")      # lo genera main.py
RUTA_CSV_SALIDA = os.path.join(RUTA_RESULTADOS, "test_retest.csv")

PACIENTES = ["OAS2_0070", "OAS2_0048", "OAS2_0087", "OAS2_0127"]

# Cuántas visitas por paciente se prueban. Cada visita tarda ~2-3 minutos (dos
# segmentaciones completas). Con 2 visitas por paciente son 8 visitas en total.
VISITAS_POR_PACIENTE = 2


# Corrección de bias y normalización de UNA adquisición (igual que en main.py)
def preparar(ruta, tamano_voxel):
    volumen = cargar_volumen_nifti(ruta)
    corregida, _, mascara_cabeza = corregir_bias_field(volumen, tamano_voxel)
    normalizada, _, _ = normalizar_intensidad(corregida, mascara_cabeza)
    return normalizada, mascara_cabeza


# Mide cerebro y ventrículos en mpr-1 y en mpr-2 de una misma visita
def medir_visita(carpeta_raw, mri_id):
    rutas = listar_adquisiciones(carpeta_raw)
    if len(rutas) < 2:
        print(f"  {mri_id}: tiene una sola adquisición, se saltea")
        return None

    tamano_voxel = leer_tamano_voxel(rutas[0])

    # mpr-1: se trata como si fuera la primera visita (atlas MNI152)
    normalizada_1, cabeza_1 = preparar(rutas[0], tamano_voxel)
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
    normalizada_2, cabeza_2 = preparar(rutas[1], tamano_voxel)
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


# Tasa de cambio por año de cada paciente (pendiente de la recta sobre todas las visitas),
# tomada del CSV que genera main.py
def tasas_por_paciente():
    if not os.path.exists(RUTA_CSV_VISITAS):
        return None
    datos = pd.read_csv(RUTA_CSV_VISITAS)
    tasas = {}
    for paciente, filas in datos.groupby("paciente"):
        anios = filas["anios_desde_primera"].values
        if len(filas) < 2:
            continue
        tasas[paciente] = {
            "tejido": np.polyfit(anios, filas["volumen_tejido_ml"].values, 1)[0],
            "ventriculos": np.polyfit(anios, filas["volumen_ventricular_ml"].values, 1)[0],
            "anios": anios.max(),
        }
    return tasas


def main():
    filas = []
    for paciente_id in PACIENTES:
        datos = cargar_datos_clinicos(RUTA_DATOS_CLINICOS, paciente_id)
        for mri_id in datos["MRI ID"].head(VISITAS_POR_PACIENTE):
            carpeta_raw = buscar_carpeta_visita(RUTA_DATASET, mri_id)
            if carpeta_raw is None:
                print(f"  {mri_id}: no se encontró la carpeta, se saltea")
                continue
            print(f"\n### {mri_id}")
            fila = medir_visita(carpeta_raw, mri_id)
            if fila is None:
                continue
            fila["paciente"] = paciente_id
            filas.append(fila)
            # Se guarda después de cada visita, por si algo falla a mitad de camino
            pd.DataFrame(filas).to_csv(RUTA_CSV_SALIDA, index=False)

    if not filas:
        print("No se pudo medir ninguna visita.")
        return

    tabla = pd.DataFrame(filas)
    tabla["dif_tejido_ml"] = (tabla["tejido_mpr2_ml"] - tabla["tejido_mpr1_ml"]).round(1)
    tabla["dif_tejido_pct"] = (100 * tabla["dif_tejido_ml"] / tabla["tejido_mpr1_ml"]).round(2)
    tabla["dif_ventriculos_ml"] = (tabla["ventriculos_mpr2_ml"] - tabla["ventriculos_mpr1_ml"]).round(2)
    tabla["dif_ventriculos_pct"] = (100 * tabla["dif_ventriculos_ml"] / tabla["ventriculos_mpr1_ml"]).round(2)
    columnas = ["paciente", "mri_id", "tejido_mpr1_ml", "tejido_mpr2_ml", "dif_tejido_ml", "dif_tejido_pct",
                "ventriculos_mpr1_ml", "ventriculos_mpr2_ml", "dif_ventriculos_ml", "dif_ventriculos_pct",
                "correlacion_registro"]
    tabla = tabla[columnas]
    tabla.to_csv(RUTA_CSV_SALIDA, index=False)

    print("\n================ TEST-RETEST (mpr-1 vs mpr-2, misma sesión) ================")
    print(tabla.to_string(index=False))

    # Error típico: promedio de las diferencias en valor absoluto
    error_tejido = tabla["dif_tejido_ml"].abs().mean()
    error_ventriculos = tabla["dif_ventriculos_ml"].abs().mean()
    print(f"\nError típico (promedio de |diferencia|):")
    print(f"  Tejido cerebral: {error_tejido:.1f} ml ({tabla['dif_tejido_pct'].abs().mean():.2f}%)")
    print(f"  Ventrículos:     {error_ventriculos:.2f} ml ({tabla['dif_ventriculos_pct'].abs().mean():.2f}%)")

    # Comparación con el cambio por año medido entre visitas
    tasas = tasas_por_paciente()
    if tasas:
        print("\nCambio acumulado en todo el seguimiento (pendiente x años) comparado con el error:")
        print("  'veces el error' > 2-3 -> el cambio no se explica por ruido de medición.")
        for paciente, tasa in tasas.items():
            cambio_tejido = tasa["tejido"] * tasa["anios"]
            cambio_ventriculos = tasa["ventriculos"] * tasa["anios"]
            print(f"  {paciente} ({tasa['anios']:.1f} años): "
                  f"tejido {cambio_tejido:+.1f} ml = {abs(cambio_tejido) / error_tejido:.1f} veces el error | "
                  f"ventrículos {cambio_ventriculos:+.1f} ml = {abs(cambio_ventriculos) / error_ventriculos:.1f} veces el error")
    else:
        print(f"\n(No se encontró {RUTA_CSV_VISITAS}: correr main.py antes para comparar con el cambio entre visitas)")

    print(f"\nTabla guardada en: {RUTA_CSV_SALIDA}")


if __name__ == "__main__":
    main()
