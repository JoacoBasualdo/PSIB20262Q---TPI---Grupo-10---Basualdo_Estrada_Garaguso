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

# =============================================================================
# ESTRUCTURA DEL CÓDIGO (carpeta Funciones/)
# =============================================================================
#   Lectura.py         carga de imágenes, tamaño de vóxel, Excel clínico (y MRBrainS18)
#   Procesamiento.py   promediado de adquisiciones, corrección de bias (N4), normalización
#   Features.py        extracción del cerebro, registro entre visitas, ventrículos, fracción ventricular
#   Interpretacion.py  evolución en el tiempo de cerebro y ventrículos, relación con CDR/MMSE
#   Validacion.py      concordancia con OASIS, test-retest y comparación con MRBrainS18
#   Visualizacion.py   todas las figuras
#   Reporte.py         PDF por paciente y máscaras ventriculares en NIfTI
#   Descarga.py        descarga/extracción del dataset (se corre aparte: python descargar_dataset.py)

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from Funciones.Lectura import cargar_datos_clinicos, buscar_carpeta_visita, listar_adquisiciones, leer_tamano_voxel, cargar_volumen_nifti
from Funciones.Procesamiento import promediar_adquisiciones, filtro_mediana, corregir_bias_field, normalizar_intensidad
from Funciones.Features import mascara_cerebro, mascara_cerebro_seguimiento, volumen_ml, segmentar_ventriculos, fraccion_ventricular
from Funciones.Interpretacion import calcular_evolucion, resumir_pacientes, calcular_cambios_ventriculares, analizar_evolucion_clinica
from Funciones.Validacion import (concordancia_oasis, imprimir_concordancia_oasis, test_retest, imprimir_test_retest,
                                  validacion_mrbrains, imprimir_validacion_mrbrains)
from Funciones.Visualizacion import (visualizar_planos_ortogonales, graficar_histogramas_visitas, visualizar_mascaras,
                                     graficar_histograma_umbral, visualizar_ventriculos, visualizar_comparacion_ventricular,
                                     guardar_graficos_evolucion, graficar_evolucion_ventricular, graficar_fraccion_ventricular,
                                     graficar_analisis_clinico, graficar_concordancia_oasis)
from Funciones import Reporte

RUTA_DATASET = "Dataset"
RUTA_DATOS_CLINICOS = os.path.join(RUTA_DATASET, "oasis_longitudinal_demographics.xlsx")
RUTA_RESULTADOS = "Resultados"
CANTIDAD_VISITAS = 5

# Pacientes a procesar (ver la descripción de cada uno en los comentarios de arriba)
PACIENTES = ["OAS2_0070", "OAS2_0048", "OAS2_0087", "OAS2_0127"]

# Tabla con una fila por visita: es lo que se usa después para la interpretación y la validación
RUTA_CSV = os.path.join(RUTA_RESULTADOS, "volumenes.csv")
RUTA_GRAFICOS = os.path.join(RUTA_RESULTADOS, "Graficos")

# ---- Qué partes correr ----
# PROCESAR_IMAGENES: si es False, no procesa las resonancias y usa el volumenes.csv ya
# generado (sirve para rehacer solo la interpretación y la validación, en segundos).
PROCESAR_IMAGENES = True
# Validaciones (ver Funciones/Validacion.py):
VALIDAR_CONCORDANCIA_OASIS = True   # segundos
VALIDAR_TEST_RETEST = True          # ~2-3 min por visita (VISITAS_TEST_RETEST por paciente)
VALIDAR_MRBRAINS = True             # ~1-2 min por sujeto; se saltea si no está Dataset/MRBrainS18
VISITAS_TEST_RETEST = 2
RUTA_MRBRAINS = os.path.join(RUTA_DATASET, "MRBrainS18")


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

    # Para comparar MR1 con la última visita sobre los mismos cortes anatómicos.
    # Solo se conservan dos imágenes en memoria; el resto del procesamiento no cambia.
    primera_visita_ventricular = None
    ultima_visita_ventricular = None

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
            mascara_tejido, mascara_intracraneal, umbral, centros, (rot, tras, corr), transformada_visitas = mascara_cerebro_seguimiento(
                normalizada, mascara, tamano_voxel, referencia, normalizada_1, mascara_1,
                devolver_transformada=True
            )
            transformada_atlas = sitk.CompositeTransform([
                referencia["transformada_atlas"], transformada_visitas
            ])
            origen_zona = (f"{referencia['mri_id']} (registro rígido: rotación máx = {rot:.2f}°, "
                           f"traslación = {tras:.2f} mm, correlación = {corr:.3f})")

        volumen_tejido = volumen_ml(mascara_tejido, tamano_voxel)
        volumen_intracraneal = volumen_ml(mascara_intracraneal, tamano_voxel)
        fila = datos[datos["MRI ID"] == mri_id].iloc[0]
        etiv_ml = fila["eTIV"]  # en el Excel está en cm^3 = ml
        nwbv_propio = volumen_tejido / etiv_ml

        # ---- Segmentación preliminar de los ventrículos laterales ----
        mascara_ventricular, info_ventricular = segmentar_ventriculos(
            normalizada, mascara_intracraneal, mascara_tejido, umbral, tamano_voxel,
            transformada_atlas=transformada_atlas
        )
        volumen_ventricular = info_ventricular["volumen_ml"]
        # Ventrículos relativizados a la zona intracraneal (misma región en todas las visitas)
        fraccion_vent = fraccion_ventricular(volumen_ventricular, volumen_intracraneal)
        print(f"  Volumen ventricular estimado: {volumen_ventricular:.1f} ml "
              f"({info_ventricular['estado']})")
        print(f"  Fracción ventricular (ventrículos / zona intracraneal): {fraccion_vent:.3f}%")

        # Guardar la máscara binaria de cada visita en formato NIfTI 3D.
        # Se usa la affine de mpr-1: la segmentación tiene la misma grilla.
        ruta_mascara_ventricular = os.path.join(
            RUTA_RESULTADOS, "MascarasVentriculares", paciente_id,
            f"{mri_id}_ventriculos.nii.gz"
        )
        Reporte.guardar_mascara_ventricular(
            mascara_ventricular, rutas[0], ruta_mascara_ventricular
        )
        print(f"  Máscara 3D guardada en: {ruta_mascara_ventricular}")

        # Para la figura comparativa se guardan MR1 y la última visita procesada.
        # 'transformada_visitas' viene de Features.py y lleva la visita actual a MR1.
        if primera_visita_ventricular is None:
            primera_visita_ventricular = {
                "id": mri_id,
                "imagen": normalizada,
                "mascara": mascara_ventricular,
                "tamano_voxel": tamano_voxel,
            }
        ultima_visita_ventricular = {
            "id": mri_id,
            "imagen": normalizada,
            "mascara": mascara_ventricular,
            "tamano_voxel": tamano_voxel,
            "transformada_visitas": (
                None if mri_id == primera_visita_ventricular["id"]
                else transformada_visitas
            ),
        }

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
            "volumen_ventricular_ml": round(volumen_ventricular, 2),
            "volumen_ventriculo_izquierdo_ml": round(info_ventricular.get("volumen_izquierdo_ml", np.nan), 2),
            "volumen_ventriculo_derecho_ml": round(info_ventricular.get("volumen_derecho_ml", np.nan), 2),
            "fraccion_ventricular_pct": round(fraccion_vent, 3),
            "estado_ventricular": info_ventricular["estado"],
            "ruta_mascara_ventricular": ruta_mascara_ventricular,
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

        fig_ventriculos = visualizar_ventriculos(
            normalizada, mascara_ventricular, mri_id, tamano_voxel
        )
        Reporte.agregar_figura(fig_ventriculos)

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

    # Comparación visual sobre exactamente los mismos cortes anatómicos.
    # La última visita se remuestrea en la grilla de MR1 con el registro ya calculado.
    if (primera_visita_ventricular is not None
            and ultima_visita_ventricular is not None
            and ultima_visita_ventricular["id"] != primera_visita_ventricular["id"]):
        fig_comparacion = visualizar_comparacion_ventricular(
            primera_visita_ventricular["imagen"],
            primera_visita_ventricular["mascara"],
            primera_visita_ventricular["tamano_voxel"],
            ultima_visita_ventricular["imagen"],
            ultima_visita_ventricular["mascara"],
            ultima_visita_ventricular["tamano_voxel"],
            ultima_visita_ventricular["transformada_visitas"],
            paciente_id,
            primera_visita_ventricular["id"],
            ultima_visita_ventricular["id"],
        )
        carpeta_comparacion = os.path.join(
            RUTA_RESULTADOS, "Graficos", "ComparacionAnatomica"
        )
        os.makedirs(carpeta_comparacion, exist_ok=True)
        ruta_comparacion_anatomica = os.path.join(
            carpeta_comparacion, f"comparacion_ventricular_{paciente_id}.png"
        )
        fig_comparacion.savefig(
            ruta_comparacion_anatomica, dpi=170, bbox_inches="tight"
        )
        Reporte.agregar_subtitulo("Comparación anatómica MR1 vs última visita")
        Reporte.agregar_figura(fig_comparacion)
        print(f"Comparación anatómica guardada en: {ruta_comparacion_anatomica}")

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    ruta_pdf = os.path.join(RUTA_RESULTADOS, f"{paciente_id}_visitas.pdf")
    Reporte.guardar_pdf(ruta_pdf, titulo=f"Paciente {paciente_id}")
    return filas


# Procesa todos los pacientes. El CSV se guarda después de cada paciente, así si algo
# falla a mitad de camino no se pierde lo que ya se calculó.
def procesar_todos():
    todas_las_filas = []
    for paciente_id in PACIENTES:
        print(f"\n########## {paciente_id} ##########")
        todas_las_filas += procesar_paciente(paciente_id)
        pd.DataFrame(todas_las_filas).to_csv(RUTA_CSV, index=False)
        print(f"Tabla actualizada: {RUTA_CSV}")
    return pd.DataFrame(todas_las_filas)


# Interpretación: evolución del cerebro, de los ventrículos y relación con la clínica
def interpretar(datos):
    # Volumen cerebral entre visitas
    evolucion = calcular_evolucion(datos)
    resumen = resumir_pacientes(evolucion)

    ruta_comparacion = os.path.join(RUTA_RESULTADOS, "comparacion_volumenes.csv")
    ruta_resumen = os.path.join(RUTA_RESULTADOS, "resumen_longitudinal.csv")
    evolucion.to_csv(ruta_comparacion, index=False)
    resumen.to_csv(ruta_resumen, index=False)
    print(f"Comparación longitudinal guardada en: {ruta_comparacion}")
    print(f"Resumen longitudinal guardado en: {ruta_resumen}")

    for ruta in guardar_graficos_evolucion(evolucion, RUTA_GRAFICOS):
        print(f"Gráfico guardado en: {ruta}")

    # Ventrículos entre visitas (en ml, % y fracción ventricular)
    ventriculos = calcular_cambios_ventriculares(datos)
    ruta_ventriculos = os.path.join(RUTA_RESULTADOS, "comparacion_ventriculos.csv")
    ventriculos.to_csv(ruta_ventriculos, index=False)
    print(f"Comparación ventricular guardada en: {ruta_ventriculos}")
    for ruta in graficar_evolucion_ventricular(ventriculos, RUTA_GRAFICOS):
        print(f"Gráfico ventricular guardado en: {ruta}")
    if "fraccion_ventricular_pct" in datos.columns:
        ruta = graficar_fraccion_ventricular(datos, os.path.join(RUTA_GRAFICOS, "fraccion_ventricular_pacientes.png"))
        print(f"Gráfico de fracción ventricular guardado en: {ruta}")

    # Análisis morfológico-clínico: compara volúmenes, nWBV, CDR y MMSE.
    detalle_clinico, resumen_clinico = analizar_evolucion_clinica(datos)
    detalle_clinico.to_csv(os.path.join(RUTA_RESULTADOS, "analisis_clinico_integrado.csv"), index=False)
    resumen_clinico.to_csv(os.path.join(RUTA_RESULTADOS, "resumen_clinico_integrado.csv"), index=False)
    print("\n========== ANÁLISIS MORFOLÓGICO-CLÍNICO ==========")
    columnas = ["paciente", "cambio_cerebral_pct", "cambio_ventricular_pct",
                "fraccion_ventricular_inicial_pct", "fraccion_ventricular_final_pct",
                "cambio_cdr", "cambio_mmse_puntos"]
    print(resumen_clinico[[c for c in columnas if c in resumen_clinico.columns]].to_string(index=False))
    print("Tablas guardadas en Resultados/ (analisis_clinico_integrado.csv y resumen_clinico_integrado.csv)")
    for ruta in graficar_analisis_clinico(detalle_clinico, RUTA_GRAFICOS):
        print(f"Gráfico clínico guardado en: {ruta}")


# Validación del método (ver Funciones/Validacion.py)
def validar(datos):
    if VALIDAR_CONCORDANCIA_OASIS:
        tabla, correlacion_global = concordancia_oasis(datos)
        ruta = os.path.join(RUTA_RESULTADOS, "concordancia_oasis.csv")
        tabla.to_csv(ruta, index=False)
        imprimir_concordancia_oasis(tabla, correlacion_global)
        graficar_concordancia_oasis(datos, os.path.join(RUTA_GRAFICOS, "concordancia_oasis.png"))
        print(f"Tabla guardada en: {ruta}")

    if VALIDAR_TEST_RETEST:
        ruta = os.path.join(RUTA_RESULTADOS, "test_retest.csv")
        tabla = test_retest(PACIENTES, RUTA_DATASET, RUTA_DATOS_CLINICOS, VISITAS_TEST_RETEST, ruta_csv=ruta)
        if tabla is None:
            print("\nTest-retest: no se encontraron visitas con 2 o más adquisiciones.")
        else:
            imprimir_test_retest(tabla, datos)
            print(f"Tabla guardada en: {ruta}")

    if VALIDAR_MRBRAINS:
        ruta = os.path.join(RUTA_RESULTADOS, "validacion_mrbrains.csv")
        carpeta_figuras = os.path.join(RUTA_GRAFICOS, "ValidacionMRBrainS")
        tabla = validacion_mrbrains(RUTA_MRBRAINS, carpeta_figuras, ruta_csv=ruta)
        if tabla is None:
            print(f"\nMRBrainS18: no se encontraron sujetos en {RUTA_MRBRAINS} (ver README). Se saltea.")
        else:
            imprimir_validacion_mrbrains(tabla)
            print(f"Tabla guardada en: {ruta}")
            print(f"Figuras de control en: {carpeta_figuras}")


def main():
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)

    if PROCESAR_IMAGENES:
        datos = procesar_todos()
    elif os.path.exists(RUTA_CSV):
        print(f"PROCESAR_IMAGENES = False: se usa la tabla existente {RUTA_CSV}")
        datos = pd.read_csv(RUTA_CSV)
    else:
        print(f"PROCESAR_IMAGENES = False pero no existe {RUTA_CSV}. Correr primero con True.")
        return

    if datos.empty:
        print("No se procesó ninguna visita; se omiten la interpretación y la validación.")
        return

    interpretar(datos)
    validar(datos)


if __name__ == "__main__":
    main()
#Que cada píxel tenga el brillo real de su tejido, sin ruido encima	Píxeles sueltos de materia blanca parecen gris por el ruido, y la segmentación queda “salpicada”	Promediado
#ue el mismo tejido tenga el mismo brillo en toda la cabeza	La materia blanca de abajo, más oscura por el bias, se clasifica como gris	Corrección de bias field
#ue el mismo tejido tenga el mismo brillo en todas las visitas	Un umbral que sirve para MR1 no sirve para MR2, y los volúmenes no son comparables en el tiempo	Normalización
#ue un píxel mida lo mismo en todas las direcciones