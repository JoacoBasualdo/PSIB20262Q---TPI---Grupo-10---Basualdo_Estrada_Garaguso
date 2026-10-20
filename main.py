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
from Funciones.Lectura import cargar_datos_clinicos, buscar_carpeta_visita, listar_adquisiciones, leer_tamano_voxel, cargar_volumen_nifti
from Funciones.Acondicionamiento import promediar_adquisiciones, filtro_mediana, corregir_bias_field, normalizar_intensidad
from Funciones.Visualizacion import visualizar_planos_ortogonales, graficar_histogramas_visitas
from Funciones import Reporte

RUTA_DATASET = "Dataset"
RUTA_DATOS_CLINICOS = os.path.join(RUTA_DATASET, "oasis_longitudinal_demographics.xlsx")
RUTA_RESULTADOS = "Resultados"
CANTIDAD_VISITAS = 1


def main():
    paciente_id = "OAS2_0127"

    Reporte.iniciar_registro()

    datos = cargar_datos_clinicos(RUTA_DATOS_CLINICOS, paciente_id)
    Reporte.agregar_subtitulo(f"Datos clínicos - {paciente_id}")
    Reporte.imprimir_datos_clinicos(datos)

    falta_mostrar_original = True
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

        fig_normalizada = visualizar_planos_ortogonales(
            normalizada,
            titulo=f"{mri_id} - Imagen normalizada [0, 1]",
            tamano_voxel=tamano_voxel
        )
        Reporte.agregar_figura(fig_normalizada)

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


if __name__ == "__main__":
    main()
#Que cada píxel tenga el brillo real de su tejido, sin ruido encima	Píxeles sueltos de materia blanca parecen gris por el ruido, y la segmentación queda “salpicada”	Promediado
#ue el mismo tejido tenga el mismo brillo en toda la cabeza	La materia blanca de abajo, más oscura por el bias, se clasifica como gris	Corrección de bias field
#ue el mismo tejido tenga el mismo brillo en todas las visitas	Un umbral que sirve para MR1 no sirve para MR2, y los volúmenes no son comparables en el tiempo	Normalización
#ue un píxel mida lo mismo en todas las direcciones