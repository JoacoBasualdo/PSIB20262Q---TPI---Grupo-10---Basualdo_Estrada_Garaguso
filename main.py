# ==========================================
# DEL DATASET: EVALUACIÓN CLÍNICA Y MÉTRICAS VOLUMÉTRICAS
# ==========================================
#
# Group: Etiqueta general del paciente durante el estudio.
#   - 'Nondemented' = Sano
#   - 'Demented' = Con Alzheimer/Demencia
#   - 'Converted' = Empezó sano y desarrolló demencia durante el estudio.
#
# MMSE: Puntuación del Mini-Mental State Examination. 
#   - Test estandarizado de evaluación cognitiva (0 a 30). 
#   - Un puntaje menor a 24 suele indicar deterioro cognitivo.
#
# CDR: Calificación Clínica de Demencia (Clinical Dementia Rating). 
#   - Etiqueta médica del nivel de gravedad:
#       0.0 = Sano / Normal
#       0.5 = Demencia muy leve
#       1.0 = Demencia leve
#       2.0 = Demencia moderada
#
# eTIV: Estimated Total Intracranial Volume (Volumen intracraneal total estimado).
#   - Representa el tamaño total del cráneo (medido en mm^3). 
#   - Permanece constante en el tiempo en adultos y sirve para normalizar 
#     las medidas del cerebro de distintas personas.
#
# nWBV: Normalized Whole Brain Volume (Volumen cerebral total normalizado).
#   - Es la proporción del cráneo que efectivamente está ocupada por tejido cerebral. 
#   - Se calcula dividiendo el volumen del cerebro por el eTIV. 
#   - Variable fundamental: una caída en este valor indica atrofia y pérdida de masa encefálica.
#
# ASF: Atlas Scaling Factor (Factor de escalamiento del atlas).
#   - Coeficiente matemático que indica cuánto hay que comprimir o expandir el cráneo 
#     del paciente para que coincida con un molde de cerebro estándar (atlas de Talairach). 
#   - Es inversamente proporcional al eTIV.



# La idea es seleccionar un paciente sano, uno enfermo y uno converted. 
# El paciente 70 tuvo 5 visitias entre los 80 y 86 años y siempre estuvo sano.
# El paciente 48 tuvo 5 visitas entre los 66 y 69. Ingreso y se mantuvo con un nivel de demencia leve (1.0)
# El paciente 127 tuvo 5 visitas a lo largo de 7 años. Ingreso con sano y a partir de la seguna sesion tenia demencia muy leve
# El paciente 87 entro con 1.0 de demencia y aumento a 2.0 en solo 2 años.

import os
import numpy as np
import pandas as pd
from Funciones.Lectura import cargar_datos_clinicos, buscar_carpeta_visita, listar_adquisiciones, leer_tamano_voxel, cargar_volumen_nifti
from Funciones.Acondicionamiento import promediar_adquisiciones, filtro_mediana, corregir_bias_field, normalizar_intensidad
from Funciones.Visualizacion import visualizar_planos_ortogonales, graficar_histogramas_visitas, visualizar_mascaras, graficar_histograma_umbral
from Funciones.Features import mascara_cerebro, mascara_cerebro_seguimiento, volumen_ml
from Funciones import Reporte

RUTA_DATASET = "Dataset"
RUTA_DATOS_CLINICOS = os.path.join(RUTA_DATASET, "oasis_longitudinal_demographics.xlsx")
RUTA_RESULTADOS = "Resultados"
CANTIDAD_VISITAS = 5

# Pacientes a procesar (ver la descripción de cada uno en los comentarios de arriba)
PACIENTES = ["OAS2_0070", "OAS2_0048", "OAS2_0087", "OAS2_0127"]

# Tabla con una fila por visita: es lo que se usa después para el análisis (AnalisisFinal.py)
RUTA_CSV = os.path.join(RUTA_RESULTADOS, "volumenes.csv")


# Procesa todas las visitas de un paciente: acondicionamiento, extracción del cerebro y
# volúmenes. Genera un PDF con las figuras y devuelve una lista con una fila por visita.
def procesar_paciente(paciente_id):
    Reporte.iniciar_registro()
    filas = []

    datos = cargar_datos_clinicos(RUTA_DATOS_CLINICOS, paciente_id)
    Reporte.agregar_subtitulo(f"Datos clínicos - {paciente_id}")
    Reporte.imprimir_datos_clinicos(datos)

    falta_mostrar_original = True
    referencia = None  # datos de la primera visita (ver más abajo), contra la que se registran las demás
    valores_histograma = []
    nombres_histograma = []

    for mri_id in datos["MRI ID"].head(CANTIDAD_VISITAS):
        Reporte.agregar_subtitulo(f"Visita {mri_id}")

        carpeta_raw = buscar_carpeta_visita(RUTA_DATASET, mri_id)
        if carpeta_raw is None:
            print(f"Aviso: No se encontró la carpeta de {mri_id} en {RUTA_DATASET}")
            continue

        rutas = listar_adquisiciones(carpeta_raw)
        if not rutas:
            print(f"Aviso: No hay adquisiciones mpr en {carpeta_raw}")
            continue

        tamano_voxel = leer_tamano_voxel(rutas[0])
        volumenes = [cargar_volumen_nifti(ruta) for ruta in rutas]

        print(f"Carpeta: {carpeta_raw}")
        print(f"Tamaño de vóxel (mm): {tamano_voxel}")
        print(f"Adquisiciones encontradas: {len(rutas)}")
        for ruta, volumen in zip(rutas, volumenes):
            print(f"  {os.path.basename(ruta)}: {volumen.shape}")

        if falta_mostrar_original:
            fig_original = visualizar_planos_ortogonales(
                volumenes[0],
                titulo=f"{mri_id} - {os.path.basename(rutas[0])} sin promediar",
                tamano_voxel=tamano_voxel
            )
            Reporte.agregar_figura(fig_original)

        promedio, movimientos = promediar_adquisiciones(volumenes, tamano_voxel)

        print()
        print(f"Registro rígido contra {os.path.basename(rutas[0])}:")
        for ruta, (rotacion, traslacion) in zip(rutas[1:], movimientos):
            print(f"  {os.path.basename(ruta)}: rotación máx = {rotacion:.2f}°, traslación = {traslacion:.2f} mm")

        fig = visualizar_planos_ortogonales(
            promedio,
            titulo=f"{mri_id} - Promedio de {len(rutas)} adquisiciones",
            tamano_voxel=tamano_voxel
        )
        Reporte.agregar_figura(fig)

        corregida, bias, mascara = corregir_bias_field(promedio, tamano_voxel)
        bias_min, bias_max = np.percentile(bias[mascara], [1, 99])

        print()
        print("Corrección de bias field (N4):")
        print(f"  El bias varía entre {bias_min:.2f} y {bias_max:.2f} dentro de la cabeza")

        fig_corregida = visualizar_planos_ortogonales(
            corregida,
            titulo=f"{mri_id} - Promedio corregido por bias field (N4)",
            tamano_voxel=tamano_voxel
        )
        Reporte.agregar_figura(fig_corregida)

        normalizada, valor_bajo, valor_alto = normalizar_intensidad(corregida, mascara)

        print()
        print("Normalización de intensidad (percentiles 1 y 99):")
        print(f"  Valor llevado a 0: {valor_bajo:.1f}")
        print(f"  Valor llevado a 1: {valor_alto:.1f}")

        fig_normalizada = visualizar_planos_ortogonales(
            normalizada,
            titulo=f"{mri_id} - Corregida y normalizada",
            tamano_voxel=tamano_voxel
        )
        Reporte.agregar_figura(fig_normalizada)

        # ---- Extracción del cerebro (skull stripping) ----
        # El atlas se ubica sobre UNA sola adquisición (mpr-1, corregida y normalizada): con el
        # promedio, que tiene menos ruido, la máscara aproximada puede pegarse a la cara y el
        # cuello y el atlas queda mal ubicado. El promedio está en la misma grilla que mpr-1,
        # así que la zona obtenida sirve directamente para medir sobre el promedio.
        corregida_1, _, mascara_1 = corregir_bias_field(volumenes[0], tamano_voxel)
        normalizada_1, _, _ = normalizar_intensidad(corregida_1, mascara_1)

        # Primera visita: se registra el atlas MNI152.
        # Visitas siguientes: se lleva la zona del cerebro de la primera con un registro rígido.
        if referencia is None:
            mascara_tejido, mascara_intracraneal, umbral, centros, transformada_atlas = mascara_cerebro(
                normalizada, mascara, tamano_voxel, normalizada_1, mascara_1
            )
            referencia = {
                "mri_id": mri_id,
                "normalizada": normalizada,
                "intracraneal": mascara_intracraneal,
                "tamano_voxel": tamano_voxel,
                "transformada_atlas": transformada_atlas,
            }
            origen_zona = "atlas MNI152 (registro afín)"
            corr = 1.0  # la primera visita es la referencia: no se registra contra nada
        else:
            mascara_tejido, mascara_intracraneal, umbral, centros, (rot, tras, corr) = mascara_cerebro_seguimiento(
                normalizada, mascara, tamano_voxel, referencia, normalizada_1, mascara_1
            )
            origen_zona = (f"{referencia['mri_id']} (registro rígido: rotación máx = {rot:.2f}°, "
                           f"traslación = {tras:.2f} mm, correlación = {corr:.3f})")

        volumen_tejido = volumen_ml(mascara_tejido, tamano_voxel)
        volumen_intracraneal = volumen_ml(mascara_intracraneal, tamano_voxel)
        fila = datos[datos["MRI ID"] == mri_id].iloc[0]
        etiv_ml = fila["eTIV"]  # en el Excel está en cm^3 = ml
        nwbv_propio = volumen_tejido / etiv_ml

        print()
        print("Extracción del cerebro:")
        print(f"  Zona del cerebro tomada de: {origen_zona}")
        print(f"  Centros de las clases (k-medias): {', '.join(f'{c:.3f}' for c in centros)}")
        print(f"  Umbral tejido / oscuro: {umbral:.3f}")
        print(f"  Volumen de tejido cerebral (gris + blanca): {volumen_tejido:.1f} ml")
        print(f"  Volumen intracraneal segmentado: {volumen_intracraneal:.1f} ml")
        print(f"  nWBV propio (tejido / eTIV de OASIS): {nwbv_propio:.3f}    nWBV de OASIS: {fila['nWBV']:.3f}")

        # Control: la zona del atlas no puede ser mucho más grande que el cráneo. En los casos
        # buenos da 1,05-1,2 veces el eTIV de OASIS; si el atlas quedó sobre la cara o el
        # cuello da 1,5 o más.
        relacion_zona = volumen_intracraneal / etiv_ml
        if relacion_zona > 1.3:
            print(f"  AVISO: la zona del cerebro mide {relacion_zona:.2f} veces el eTIV. "
                  "El atlas pudo haber quedado mal ubicado: revisar la figura.")

        # Fila de la tabla de resultados. Las columnas clínicas se copian del Excel de OASIS
        # para que el análisis no necesite volver a leerlo.
        filas.append({
            "paciente": paciente_id,
            "grupo": fila["Group"],
            "visita": int(fila["Visit"]),
            "mri_id": mri_id,
            "anios_desde_primera": round(fila["MR Delay"] / 365.25, 3),
            "edad": fila["Age"],
            "cdr": fila["CDR"],
            "mmse": fila["MMSE"],
            "etiv_oasis_ml": round(etiv_ml, 1),
            "nwbv_oasis": round(fila["nWBV"], 4),
            "volumen_tejido_ml": round(volumen_tejido, 1),
            "volumen_intracraneal_ml": round(volumen_intracraneal, 1),
            "nwbv_propio": round(nwbv_propio, 4),
            "umbral_tejido": round(float(umbral), 4),
            "correlacion_registro": round(corr, 4),
        })

        fig_cerebro = visualizar_mascaras(
            normalizada,
            [mascara_tejido, mascara_intracraneal],
            titulo=f"{mri_id} - Extracción del cerebro",
            tamano_voxel=tamano_voxel,
            nombres=["Tejido cerebral", "Intracraneal"]
        )
        Reporte.agregar_figura(fig_cerebro)

        fig_umbral = graficar_histograma_umbral(
            normalizada[mascara], umbral, centros,
            titulo=f"{mri_id} - Histograma de la cabeza y umbral de tejido"
        )
        Reporte.agregar_figura(fig_umbral)

        valores_histograma.append((corregida[mascara], normalizada[mascara]))
        nombres_histograma.append(mri_id)

        if falta_mostrar_original:
            print(f"  Vóxeles dentro de la máscara de la cabeza: {mascara.mean() * 100:.1f}% del volumen")
            fig_mascara = visualizar_planos_ortogonales(
                mascara.astype(float),
                titulo=f"{mri_id} - Máscara de la cabeza",
                tamano_voxel=tamano_voxel
            )
            Reporte.agregar_figura(fig_mascara)

            fig_bias = visualizar_planos_ortogonales(
                bias,
                titulo=f"{mri_id} - Mapa del bias field estimado",
                tamano_voxel=tamano_voxel
            )
            Reporte.agregar_figura(fig_bias)

            promedio_mediana = filtro_mediana(promedio, tamano=3)
            fig_mediana = visualizar_planos_ortogonales(
                promedio_mediana,
                titulo=f"{mri_id} - Promedio + filtro de mediana (3x3x3)",
                tamano_voxel=tamano_voxel
            )
            Reporte.agregar_figura(fig_mediana)

            falta_mostrar_original = False

    if valores_histograma:
        Reporte.agregar_subtitulo("Comparación de intensidades entre visitas")
        fig_histogramas = graficar_histogramas_visitas(
            valores_histograma, nombres_histograma,
            titulo=f"{paciente_id} - Histogramas antes y después de normalizar"
        )
        Reporte.agregar_figura(fig_histogramas)

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    ruta_pdf = os.path.join(RUTA_RESULTADOS, f"{paciente_id}_visitas.pdf")
    Reporte.guardar_pdf(ruta_pdf, titulo=f"Paciente {paciente_id}")
    return filas


# Procesa todos los pacientes. El CSV se guarda después de cada paciente, así si algo
# falla a mitad de camino no se pierde lo que ya se calculó.
def main():
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    todas_las_filas = []
    for paciente_id in PACIENTES:
        print(f"\n########## {paciente_id} ##########")
        todas_las_filas += procesar_paciente(paciente_id)
        pd.DataFrame(todas_las_filas).to_csv(RUTA_CSV, index=False)
        print(f"Tabla actualizada: {RUTA_CSV}")


if __name__ == "__main__":
    main()
#Que cada píxel tenga el brillo real de su tejido, sin ruido encima	Píxeles sueltos de materia blanca parecen gris por el ruido, y la segmentación queda “salpicada”	Promediado
#ue el mismo tejido tenga el mismo brillo en toda la cabeza	La materia blanca de abajo, más oscura por el bias, se clasifica como gris	Corrección de bias field
#ue el mismo tejido tenga el mismo brillo en todas las visitas	Un umbral que sirve para MR1 no sirve para MR2, y los volúmenes no son comparables en el tiempo	Normalización
#ue un píxel mida lo mismo en todas las direcciones