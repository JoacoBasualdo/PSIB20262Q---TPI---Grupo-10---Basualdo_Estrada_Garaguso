# =============================================================================
# EXTRACCIÓN DE FEATURES
# =============================================================================
# Todo lo que se MIDE en cada visita:
#   1. Cerebro: extracción del cerebro con el atlas MNI152 (primera visita) y registro
#      rígido entre visitas (visitas siguientes) -> tejido cerebral y zona intracraneal.
#   2. Ventrículos laterales: atlas Harvard-Oxford + umbral de LCR + conectividad 3D.
#   3. Fracción ventricular: ventrículos relativizados a la zona intracraneal.

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from functools import lru_cache
from Funciones.Procesamiento import numpy_a_sitk, rellenar_huecos_por_corte


# =============================================================================
# HERRAMIENTAS GENERALES
# =============================================================================

# Distancia (en mm) de cada vóxel de la máscara al borde más cercano.
# Con esto se hacen erosiones y dilataciones de radio exacto en mm, aunque los
# vóxeles no sean cúbicos (OASIS: 1 x 1 x 1.25 mm):
#   erosión de r mm  -> distancia_al_borde(mascara) > r
#   dilatación de r mm -> distancia_al_borde(~mascara) <= r
def distancia_al_borde(mascara, tamano_voxel):
    return ndimage.distance_transform_edt(mascara, sampling=tamano_voxel)


# Erosión: "achica" la máscara r mm desde el borde. Se queda solo con los vóxeles que
# están a más de r mm del fondo. Sirve para cortar uniones finas entre dos regiones.
def erosionar(mascara, radio_mm, tamano_voxel):
    return distancia_al_borde(mascara, tamano_voxel) > radio_mm


# Dilatación: "agranda" la máscara r mm. Agrega todos los vóxeles que están a menos
# de r mm de la máscara. Es la operación inversa de la erosión.
def dilatar(mascara, radio_mm, tamano_voxel):
    return distancia_al_borde(~mascara, tamano_voxel) <= radio_mm


# Numera cada "isla" de vóxeles conectados entre sí (ndimage.label) y se queda solo con
# la más grande. Sirve para descartar pedazos sueltos que no son cerebro.
def componente_mas_grande(mascara):
    etiquetas, cantidad = ndimage.label(mascara)
    if cantidad <= 1:
        return mascara
    tamanos = ndimage.sum(mascara, etiquetas, range(1, cantidad + 1))
    return etiquetas == (np.argmax(tamanos) + 1)


# Pasa de cantidad de vóxeles a mililitros (1 ml = 1000 mm^3)
def volumen_ml(mascara, tamano_voxel):
    return mascara.sum() * float(np.prod(tamano_voxel)) / 1000


# =============================================================================
# UMBRAL DE TEJIDO
# =============================================================================

# K-medias en 1D: agrupa los brillos de la cabeza en 4 clases
#   oscuro (hueso, LCR, aire) | sustancia gris | sustancia blanca | grasa (muy brillante)
# Arranca con 4 centros repartidos en el histograma y repite:
#   1. Cada vóxel va a la clase cuyo centro tiene el brillo más parecido.
#   2. Cada centro pasa a ser el promedio de los vóxeles de su clase.
# hasta que los centros dejan de moverse. Los límites entre clases quedan a mitad de
# camino entre centros vecinos.
# Se usan 4 clases para que la grasa del cuero cabelludo tenga su propia clase; si no,
# "empuja" los límites y el primero cae en medio de la sustancia gris.
def limites_kmedias(valores, clases=4, iteraciones=100):
    centros = np.quantile(valores, np.linspace(0.1, 0.9, clases))
    for _ in range(iteraciones):
        limites = (centros[:-1] + centros[1:]) / 2
        etiquetas = np.digitize(valores, limites)
        nuevos = np.array([valores[etiquetas == k].mean() if np.any(etiquetas == k) else centros[k]
                           for k in range(clases)])
        if np.allclose(nuevos, centros, atol=1e-5):
            break
        centros = nuevos
    return (centros[:-1] + centros[1:]) / 2, centros


# =============================================================================
# ATLAS MNI152
# =============================================================================
# El atlas MNI152 (ICBM 2009a) es un "cerebro promedio" armado con resonancias T1 de
# 152 adultos. Viene sin cráneo y con mapas de probabilidad de sustancia gris y blanca.
# Se usa la copia que trae la librería nilearn (no hace falta descargar nada).
#
# El atlas está orientado (izq-der, post-ant, inf-sup). Las imágenes de OASIS, tal como
# las carga nibabel, están orientadas (post-ant, inf-sup, izq-der) — es lo que asume
# visualizar_planos_ortogonales. Por eso se reordenan los ejes del atlas.

def cargar_atlas():
    import nilearn
    carpeta = os.path.join(os.path.dirname(nilearn.__file__), "datasets", "data")

    def leer(nombre):
        ruta = os.path.join(carpeta, f"mni_icbm152_{nombre}_tal_nlin_sym_09a_converted.nii.gz")
        volumen = nib.load(ruta).get_fdata().astype(np.float32)
        return np.transpose(volumen, (1, 2, 0))  # (izq-der, post-ant, inf-sup) -> (post-ant, inf-sup, izq-der)

    t1 = leer("t1")
    probabilidad_cerebro = leer("gm") + leer("wm")

    # Máscara del cerebro del atlas: gris + blanca, cerrando surcos y rellenando ventrículos
    cerebro = ndimage.binary_closing(probabilidad_cerebro > 0.5, iterations=3)
    cerebro = rellenar_huecos_por_corte(cerebro)

    return t1 / t1.max(), cerebro, (1.0, 1.0, 1.0)


# Búsqueda inicial de la inclinación de la cabeza. El registro por descenso de gradiente
# solo corrige giros chicos (~10°): si la cabeza está inclinada distinto (mentón más arriba
# o más abajo), se queda trabado en una solución mala. Por eso, antes de registrar, se
# prueban inclinaciones de -45° a +45° cada 5° alrededor del eje izquierda-derecha (el
# movimiento de "asentir"), sobre imágenes reducidas para que sea rápido, y se devuelve
# la que mejor encaja. Se usa tanto para el atlas como entre visitas.
#   inicial: transformada rígida (Euler3D) de partida, ya centrada
#   mascara_fija: si se da, solo se compara esa zona de la imagen fija
def busqueda_inclinacion(fija_sitk, movil_sitk, inicial, mascara_fija=None, paso_grados=5, pasos=9):
    busqueda = sitk.ImageRegistrationMethod()
    busqueda.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    if mascara_fija is not None:
        busqueda.SetMetricFixedMask(mascara_fija)
    busqueda.SetMetricSamplingStrategy(busqueda.RANDOM)
    busqueda.SetMetricSamplingPercentage(0.1, seed=42)
    busqueda.SetInterpolator(sitk.sitkLinear)
    # Ejes de SimpleITK: x = izq-der, y = inf-sup, z = post-ant. Se gira solo alrededor de x.
    busqueda.SetOptimizerAsExhaustive(numberOfSteps=[pasos, 0, 0, 0, 0, 0],
                                      stepLength=np.radians(paso_grados))
    busqueda.SetOptimizerScales([1, 1, 1, 1, 1, 1])
    busqueda.SetShrinkFactorsPerLevel([4])
    busqueda.SetSmoothingSigmasPerLevel([2])
    busqueda.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    busqueda.SetInitialTransform(inicial, inPlace=True)
    busqueda.Execute(fija_sitk, movil_sitk)
    return inicial


# Execute() de SimpleITK puede devolver la transformada "envuelta" en una compuesta.
# Esto la desenvuelve y la devuelve con su tipo (afín o rígida) para poder usarla.
def desenvolver(transformada, tipo):
    if transformada.GetName() == "CompositeTransform":
        transformada = sitk.CompositeTransform(transformada).GetNthTransform(0)
    return tipo(transformada)


# Registro afín: busca la transformación (rotación, traslación, escala y cizallamiento,
# 12 parámetros) que mejor superpone el atlas sobre el cerebro del paciente.
# Es la misma idea que el registro rígido de Procesamiento.py (información mutua,
# descenso por gradiente, 3 resoluciones), pero:
#   - Afín en vez de rígido, porque cada cerebro tiene distinto tamaño y forma.
#   - Solo se mira la zona del cerebro del paciente (máscara aproximada + 20 mm),
#     porque el atlas no tiene cráneo y el cuero cabelludo confundiría al registro.
# Así lo hizo también OASIS: registraron cada cerebro contra un atlas para estimar el eTIV.
# Devuelve la zona del cerebro en el paciente y la transformada (paciente -> atlas).
def registrar_atlas(normalizada, mascara_aproximada, tamano_voxel, atlas_t1, atlas_cerebro, voxel_atlas):
    fija = numpy_a_sitk(normalizada, tamano_voxel)
    movil = numpy_a_sitk(atlas_t1, voxel_atlas)

    zona_cerebro = dilatar(mascara_aproximada, 20.0, tamano_voxel)
    zona_sitk = sitk.Cast(numpy_a_sitk(zona_cerebro.astype(np.float32), tamano_voxel), sitk.sitkUInt8)

    # Punto de partida: hacer coincidir el centro del cerebro del paciente con el del atlas
    # y buscar la inclinación de la cabeza que mejor encaja (ver busqueda_inclinacion)
    centro_fijo = numpy_a_sitk(mascara_aproximada.astype(np.float32), tamano_voxel)
    centro_movil = numpy_a_sitk(atlas_cerebro.astype(np.float32), voxel_atlas)
    rigida = sitk.CenteredTransformInitializer(
        centro_fijo, centro_movil, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    rigida = busqueda_inclinacion(fija, movil, rigida, zona_sitk)

    # Se pasa a afín (que además puede escalar) para el registro completo
    inicial = sitk.AffineTransform(3)
    inicial.SetCenter(rigida.GetCenter())
    inicial.SetMatrix(rigida.GetMatrix())
    inicial.SetTranslation(rigida.GetTranslation())

    registro = sitk.ImageRegistrationMethod()
    registro.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registro.SetMetricFixedMask(zona_sitk)
    registro.SetMetricSamplingStrategy(registro.RANDOM)
    registro.SetMetricSamplingPercentage(0.05, seed=42)
    registro.SetInterpolator(sitk.sitkLinear)
    registro.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300)
    registro.SetOptimizerScalesFromPhysicalShift()
    registro.SetShrinkFactorsPerLevel([4, 2, 1])
    registro.SetSmoothingSigmasPerLevel([3, 2, 1])
    registro.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registro.SetInitialTransform(inicial, inPlace=False)

    transformada = desenvolver(registro.Execute(fija, movil), sitk.AffineTransform)

    # Se lleva la máscara de cerebro del atlas al espacio del paciente
    cerebro_atlas_sitk = numpy_a_sitk(atlas_cerebro.astype(np.float32), voxel_atlas)
    cerebro_en_paciente = sitk.Resample(cerebro_atlas_sitk, fija, transformada, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(cerebro_en_paciente) > 0.5, transformada


# Registra el atlas en una imagen: arma la máscara aproximada (erosión de 4 mm, componente
# más grande) para saber dónde está el cerebro y después llama a registrar_atlas.
def ubicar_atlas(normalizada, mascara_cabeza, tamano_voxel, radio_aproximado_mm=4.0):
    limites, _ = limites_kmedias(normalizada[mascara_cabeza])
    tejido = (normalizada > limites[0]) & mascara_cabeza
    mascara_aproximada = componente_mas_grande(erosionar(tejido, radio_aproximado_mm, tamano_voxel))

    atlas_t1, atlas_cerebro, voxel_atlas = cargar_atlas()
    zona, transformada = registrar_atlas(normalizada, mascara_aproximada, tamano_voxel,
                                         atlas_t1, atlas_cerebro, voxel_atlas)

    # Control: si el registro falló, el atlas casi no se superpone con la máscara aproximada
    superposicion = (zona & mascara_aproximada).sum() / max(mascara_aproximada.sum(), 1)
    if superposicion < 0.8:
        print(f"  AVISO: el atlas cubre solo el {superposicion * 100:.0f}% de la máscara aproximada. "
              "Revisar la figura: el registro pudo haber fallado.")
    return zona, transformada


# Pasos 4 y 5 de la extracción: tejido dentro de la zona del cerebro (+ margen),
# apertura para cortar bordes finos (meninges, senos venosos) y componente más grande.
def tejido_dentro_de_zona(tejido, zona, tamano_voxel, margen_mm=2.0, apertura_mm=1.5):
    mascara = tejido & dilatar(zona, margen_mm, tamano_voxel)
    nucleo = erosionar(mascara, apertura_mm, tamano_voxel)
    mascara &= dilatar(nucleo, apertura_mm + 0.01, tamano_voxel)
    return componente_mas_grande(mascara)


# =============================================================================
# EXTRACCIÓN DEL CEREBRO (SKULL STRIPPING) - PRIMERA VISITA
# =============================================================================
#   1. Umbral (k-medias, 4 clases): separa lo oscuro (hueso, LCR, aire) del tejido
#      (sustancia gris, blanca y también grasa/músculo del cuero cabelludo).
#   2. Máscara aproximada: se erosiona el tejido 4 mm y se toma la componente más grande.
#      No es precisa (pierde corteza), solo sirve para saber DÓNDE está el cerebro.
#   3. Registro afín del atlas MNI152 sobre el paciente y se trae su máscara de cerebro
#      -> mascara_intracraneal: la zona donde "debería" estar el cerebro (incluye ventrículos).
#   4. Tejido dentro de esa zona (+ margen de 2 mm por si el registro no es perfecto).
#      Como el umbral deja afuera lo oscuro, los ventrículos y surcos agrandados por
#      la atrofia quedan excluidos solos, aunque el atlas sea de un cerebro sano.
#   5. Apertura de 1.5 mm: corta bordes finos que no son cerebro (meninges, senos venosos).
#      -> mascara_tejido: sustancia gris + blanca.
#
# imagen_atlas: imagen sobre la que se ubica el atlas (pasos 2 y 3). Conviene pasar UNA sola
# adquisición (mpr-1) y no el promedio: en el promedio hay menos ruido y la máscara aproximada
# puede quedar "pegada" a la cara y el cuello, y el atlas termina ubicado ahí. Como el promedio
# se arma sobre la grilla de mpr-1, la zona sirve igual para el promedio.
# Si no se pasa, se usa la misma imagen normalizada.
def mascara_cerebro(normalizada, mascara_cabeza, tamano_voxel,
                    imagen_atlas=None, mascara_cabeza_atlas=None,
                    margen_mm=2.0, apertura_mm=1.5):
    if imagen_atlas is None:
        imagen_atlas, mascara_cabeza_atlas = normalizada, mascara_cabeza

    limites, centros = limites_kmedias(normalizada[mascara_cabeza])
    umbral = limites[0]
    tejido = (normalizada > umbral) & mascara_cabeza

    mascara_intracraneal, transformada_atlas = ubicar_atlas(imagen_atlas, mascara_cabeza_atlas, tamano_voxel)
    mascara_tejido = tejido_dentro_de_zona(tejido, mascara_intracraneal, tamano_voxel, margen_mm, apertura_mm)

    return mascara_tejido, mascara_intracraneal, umbral, centros, transformada_atlas


# =============================================================================
# SEGUIMIENTO: VISITAS POSTERIORES
# =============================================================================
# Registro rígido (6 parámetros: 3 rotaciones + 3 traslaciones) entre dos visitas del
# MISMO paciente. Como es la misma cabeza, no hace falta escalar: solo cambia cómo se
# acostó en el resonador. Arranca alineando los centros de masa de las dos cabezas
# (MOMENTS), por si en una visita la cabeza quedó corrida dentro de la imagen.
#
# Opcionalmente se puede comparar solo una zona (mascara_fija / mascara_movil) y arrancar
# desde una transformada ya calculada (inicial). Ver mascara_cerebro_seguimiento.
def registrar_visitas(fija, tamano_voxel_fija, movil, tamano_voxel_movil,
                      mascara_fija=None, mascara_movil=None, inicial=None):
    fija_sitk = numpy_a_sitk(fija, tamano_voxel_fija)
    movil_sitk = numpy_a_sitk(movil, tamano_voxel_movil)

    if inicial is None:
        inicial = sitk.CenteredTransformInitializer(
            fija_sitk, movil_sitk, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    registro = sitk.ImageRegistrationMethod()
    registro.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if mascara_fija is not None:
        registro.SetMetricFixedMask(sitk.Cast(numpy_a_sitk(mascara_fija.astype(np.float32), tamano_voxel_fija), sitk.sitkUInt8))
    if mascara_movil is not None:
        registro.SetMetricMovingMask(sitk.Cast(numpy_a_sitk(mascara_movil.astype(np.float32), tamano_voxel_movil), sitk.sitkUInt8))
    registro.SetMetricSamplingStrategy(registro.RANDOM)
    registro.SetMetricSamplingPercentage(0.1, seed=42)
    registro.SetInterpolator(sitk.sitkLinear)
    registro.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300)
    registro.SetOptimizerScalesFromPhysicalShift()
    registro.SetShrinkFactorsPerLevel([4, 2, 1])
    registro.SetSmoothingSigmasPerLevel([2, 1, 0])
    registro.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registro.SetInitialTransform(inicial, inPlace=False)

    return desenvolver(registro.Execute(fija_sitk, movil_sitk), sitk.Euler3DTransform)


# Usa el atlas como "intermediario" para estimar cuánto se movió la cabeza entre dos visitas.
# Si conocemos cómo encaja el atlas en cada visita (dos transformadas afines visita -> atlas),
# combinándolas se obtiene visita actual -> atlas -> visita de referencia. Esa combinación
# es casi rígida (misma cabeza), así que se le saca la parte de rotación y traslación.
# Sirve como punto de partida cuando la cabeza se movió mucho (ej: 0070 MR2: 14° y 53 mm).
def rigida_desde_atlas(afin_ref, afin_actual, centro):
    combinada = sitk.CompositeTransform([afin_ref.GetInverse(), afin_actual])  # se aplica de derecha a izquierda
    centro = np.array(centro)
    origen = np.array(combinada.TransformPoint(tuple(centro)))
    # Matriz lineal aproximada: cómo se mueven 3 puntos a 20 mm del centro
    matriz = np.column_stack([
        (np.array(combinada.TransformPoint(tuple(centro + paso))) - origen) / 20.0
        for paso in np.eye(3) * 20.0
    ])
    # Rotación más parecida a esa matriz (descomposición SVD: se descarta la escala)
    u, _, vt = np.linalg.svd(matriz)
    rotacion = u @ vt
    if np.linalg.det(rotacion) < 0:
        u[:, -1] *= -1
        rotacion = u @ vt
    rigida = sitk.Euler3DTransform()
    rigida.SetCenter(tuple(centro))
    rigida.SetMatrix(tuple(rotacion.ravel()))
    rigida.SetTranslation(tuple(origen - centro))
    return rigida


# Extracción del cerebro para las visitas 2, 3, ...: en vez de usar el atlas de cada visita,
# se toma la zona del cerebro de la PRIMERA visita y se la lleva a la visita actual con un
# registro rígido. Ventajas:
#   - El cráneo no cambia con la edad, así que la zona intracraneal debe ser la misma
#     en todas las visitas: usar la misma evita que el error del atlas cambie entre visitas.
#   - Los volúmenes de todas las visitas quedan medidos dentro de la misma región,
#     que es lo que hace falta para compararlos en el tiempo.
#
# El registro rígido puede trabarse si la cabeza se movió mucho entre visitas, así que se
# prueban dos puntos de partida y se queda con el que mejor superpone los cerebros:
#   A. Búsqueda de inclinación (busqueda_inclinacion).
#   B. Atlas como intermediario (rigida_desde_atlas).
# Desde cada uno se hace el registro con la cabeza entera y un ajuste fino mirando solo
# cráneo/cerebro (el cuello y la mandíbula cambian de ángulo entre visitas).
#
# referencia: diccionario con la primera visita (ver main.py):
#   "normalizada", "intracraneal", "tamano_voxel", "transformada_atlas"
def mascara_cerebro_seguimiento(normalizada, mascara_cabeza, tamano_voxel, referencia,
                                imagen_atlas=None, mascara_cabeza_atlas=None,
                                margen_mm=2.0, apertura_mm=1.5,
                                devolver_transformada=False):
    if imagen_atlas is None:
        imagen_atlas, mascara_cabeza_atlas = normalizada, mascara_cabeza
    normalizada_ref = referencia["normalizada"]
    intracraneal_ref = referencia["intracraneal"]
    tamano_voxel_ref = referencia["tamano_voxel"]

    limites, centros = limites_kmedias(normalizada[mascara_cabeza])
    umbral = limites[0]
    tejido = (normalizada > umbral) & mascara_cabeza

    fija = numpy_a_sitk(normalizada, tamano_voxel)
    zona_ref = numpy_a_sitk(intracraneal_ref.astype(np.float32), tamano_voxel_ref)
    imagen_ref = numpy_a_sitk(normalizada_ref, tamano_voxel_ref)

    # Lleva la zona de la referencia a la visita actual con una transformada y mide qué tan
    # bien quedó: dentro del cerebro, las dos visitas tienen que parecerse mucho
    # (correlación cercana a 1). Si el registro falló, la correlación cae.
    def aplicar(transformada):
        zona = sitk.GetArrayFromImage(sitk.Resample(zona_ref, fija, transformada, sitk.sitkLinear, 0.0)) > 0.5
        ref_en_actual = sitk.GetArrayFromImage(sitk.Resample(imagen_ref, fija, transformada, sitk.sitkLinear, 0.0))
        correlacion = float(np.corrcoef(normalizada[zona], ref_en_actual[zona])[0, 1]) if zona.sum() > 1000 else 0.0
        return zona, correlacion

    # Zonas de cráneo/cerebro para el ajuste fino
    aproximada = componente_mas_grande(erosionar(tejido, 4.0, tamano_voxel))
    zona_fija = dilatar(aproximada, 15.0, tamano_voxel) & mascara_cabeza
    zona_movil = dilatar(intracraneal_ref, 10.0, tamano_voxel_ref)

    # Puntos de partida
    inicio_a = busqueda_inclinacion(
        fija, imagen_ref,
        sitk.CenteredTransformInitializer(fija, imagen_ref, sitk.Euler3DTransform(),
                                          sitk.CenteredTransformInitializerFilter.MOMENTS)
    )
    _, transformada_atlas = ubicar_atlas(imagen_atlas, mascara_cabeza_atlas, tamano_voxel)
    centro = fija.TransformContinuousIndexToPhysicalPoint([n / 2 for n in fija.GetSize()])
    inicio_b = rigida_desde_atlas(referencia["transformada_atlas"], transformada_atlas, centro)

    # Desde cada punto de partida: cabeza entera y después ajuste fino. Se guarda el mejor.
    candidatos = []
    for nombre, inicio in (("inclinación", inicio_a), ("atlas", inicio_b)):
        cabeza = registrar_visitas(normalizada, tamano_voxel, normalizada_ref, tamano_voxel_ref, inicial=inicio)
        fina = registrar_visitas(normalizada, tamano_voxel, normalizada_ref, tamano_voxel_ref,
                                 zona_fija, zona_movil, inicial=sitk.Euler3DTransform(cabeza))
        for etapa, transformada in (("cabeza entera", cabeza), ("ajuste fino", fina)):
            zona, correlacion = aplicar(transformada)
            candidatos.append((correlacion, f"{nombre} + {etapa}", transformada, zona))

    correlacion, elegido, transformada, mascara_intracraneal = max(candidatos, key=lambda c: c[0])
    print("  Registro entre visitas: " + " | ".join(f"{c[1]} r = {c[0]:.3f}" for c in candidatos))
    print(f"  -> se usa {elegido}")
    if correlacion < 0.85:
        print(f"  AVISO: correlación con la visita de referencia = {correlacion:.3f}. "
              "El registro entre visitas pudo haber fallado.")

    mascara_tejido = tejido_dentro_de_zona(tejido, mascara_intracraneal, tamano_voxel, margen_mm, apertura_mm)

    # Cuánto se movió la cabeza respecto de la primera visita (para reportar)
    parametros = transformada.GetParameters()
    rotacion = float(np.degrees(np.abs(parametros[:3])).max())
    traslacion = float(np.linalg.norm(parametros[3:6]))

    if devolver_transformada:
        return mascara_tejido, mascara_intracraneal, umbral, centros, (rotacion, traslacion, correlacion), transformada
    return mascara_tejido, mascara_intracraneal, umbral, centros, (rotacion, traslacion, correlacion)


# =============================================================================
# VENTRÍCULOS LATERALES
# =============================================================================
# Segmentación exploratoria guiada por atlas anatómico (Harvard-Oxford subcortical,
# izquierda/derecha), proyectado al espacio de cada visita con las transformaciones del
# registro del cerebro; umbral de LCR T1 local; conectividad 3D con semilla por hemisferio.

# Margen (mm) alrededor de la etiqueta del atlas: permite expansión ventricular
# pero evita buscar LCR en toda la cabeza.
MARGEN_ROI_MM = 8.0

# Pequeña tolerancia por volumen parcial en los límites del LCR T1.
MARGEN_UMBRAL = 0.025


@lru_cache(maxsize=1)
def _atlas_ventricular_en_mni():
    """Etiquetas laterales de Harvard-Oxford en la grilla MNI de Features.py.

    El atlas se descarga/cachea una sola vez en Dataset/Atlas si falta.
    Se remuestrea al template exacto que usa cargar_atlas() en Features.py.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    import nilearn
    from nilearn.datasets import fetch_atlas_harvard_oxford

    carpeta_cache = os.path.join("Dataset", "Atlas")

    atlas = fetch_atlas_harvard_oxford(
        "sub-maxprob-thr0-1mm",
        data_dir=carpeta_cache,
        verbose=0
    )

    imagen_atlas = atlas.maps

    if isinstance(imagen_atlas, (str, os.PathLike)):
        imagen_atlas = nib.load(str(imagen_atlas))

    nombres = [
        str(nombre).strip().lower()
        for nombre in atlas.labels
    ]

    # El fichero original puede escribir "ventrical" en lugar de "ventricle".
    ids = [
        i for i, nombre in enumerate(nombres)
        if "lateral ventri" in nombre
    ]

    if len(ids) != 2:
        raise RuntimeError(
            "Harvard-Oxford: no se encontraron exactamente las dos etiquetas "
            f"ventriculares. Etiquetas disponibles: {atlas.labels}"
        )

    carpeta_nilearn = os.path.join(
        os.path.dirname(nilearn.__file__),
        "datasets",
        "data"
    )

    archivo_mni = os.path.join(
        carpeta_nilearn,
        "mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz"
    )

    if not os.path.isfile(archivo_mni):
        raise FileNotFoundError(
            f"No se encontró el MNI que utiliza Features.py: {archivo_mni}"
        )

    plantilla = nib.load(archivo_mni)

    # El atlas HO y el template MNI pueden diferir en grilla y affine.
    # Interpolación vecino más cercano para preservar IDs de etiquetas.
    regrillado = resample_from_to(
        imagen_atlas,
        (plantilla.shape, plantilla.affine),
        order=0
    )

    etiquetas = np.asarray(regrillado.dataobj)

    # Mismo orden de ejes que Features.cargar_atlas: (AP, SI, LR).
    return tuple(
        np.transpose(etiquetas == idx, (1, 2, 0)).copy()
        for idx in ids
    )


def _mascaras_atlas_en_paciente(
    volumen,
    tamano_voxel,
    transformada_atlas
):
    """Lleva las dos etiquetas MNI al espacio de la visita actual."""

    if transformada_atlas is None:
        raise ValueError(
            "Falta transformada_atlas: se necesita el registro de Features.py"
        )


    referencia = numpy_a_sitk(
        volumen,
        tamano_voxel
    )

    proyectadas = []

    for etiqueta in _atlas_ventricular_en_mni():

        etiqueta_sitk = numpy_a_sitk(
            etiqueta.astype(np.float32),
            (1., 1., 1.)
        )

        resultado = sitk.Resample(
            etiqueta_sitk,
            referencia,
            transformada_atlas,
            sitk.sitkNearestNeighbor,
            0.0,
            sitk.sitkFloat32
        )

        proyectadas.append(
            sitk.GetArrayFromImage(resultado) > 0.5
        )

    return proyectadas


def _roi_alrededor_de(
    mascara_atlas,
    tamano_voxel,
    margen_mm
):
    """Dilata solo una caja pequeña alrededor de la estructura: ahorra memoria."""

    roi = np.zeros(
        mascara_atlas.shape,
        dtype=bool
    )

    coords = np.where(mascara_atlas)

    if len(coords[0]) == 0:
        return roi

    slices = []

    for eje, indices in enumerate(coords):

        padding = int(
            np.ceil(margen_mm / float(tamano_voxel[eje]))
        ) + 1

        a = max(
            0,
            int(indices.min()) - padding
        )

        b = min(
            roi.shape[eje],
            int(indices.max()) + padding + 1
        )

        slices.append(slice(a, b))

    slices = tuple(slices)

    recorte = mascara_atlas[slices]

    distancias = ndimage.distance_transform_edt(
        ~recorte,
        sampling=tuple(float(v) for v in tamano_voxel)
    )

    roi[slices] = distancias <= margen_mm

    return roi


def _seleccionar_regiones(
    candidatos,
    prior,
    tamano_voxel
):
    """Conserva los componentes oscuros que contienen la semilla del atlas."""

    conectividad = ndimage.generate_binary_structure(
        3, 1
    )  # 6 vecinos

    etiquetas, n = ndimage.label(
        candidatos,
        structure=conectividad
    )

    mascara = np.zeros(
        candidatos.shape,
        dtype=bool
    )

    if n == 0:
        return mascara, 0

    solapamiento = np.bincount(
        etiquetas[prior],
        minlength=n + 1
    )

    solapamiento[0] = 0

    principal = int(np.argmax(solapamiento))

    if solapamiento[principal] < 12:
        return mascara, 0

    mascara = etiquetas == principal
    numero = 1

    # Si el LCR aparece en dos fragmentos relevantes dentro de la etiqueta
    # del atlas, también recuperamos el segundo. No se eligen por tamaño global.
    tamano_min_vox = max(
        12,
        int(np.ceil(0.2 * 1000 / np.prod(tamano_voxel)))
    )

    candidatos_secundarios = np.flatnonzero(
        solapamiento >= max(
            12,
            int(0.25 * solapamiento[principal])
        )
    )

    for idx in candidatos_secundarios:

        if idx == principal or idx == 0:
            continue

        region = etiquetas == idx

        if region.sum() >= tamano_min_vox:
            mascara |= region
            numero += 1

    return mascara, numero


# =============================================================================
# SEGMENTACIÓN VENTRICULAR 3D
# =============================================================================

def segmentar_ventriculos(
    normalizada,
    mascara_intracraneal,
    mascara_tejido,
    umbral,
    tamano_voxel,
    transformada_atlas=None
):
    """Segmenta los ventrículos laterales con atlas + umbral + conectividad.

    transformada_atlas debe mapear coordenadas de la visita actual al MNI de
    Features.py. En MR1 es la transformada afín devuelta por mascara_cerebro;
    en visitas posteriores, la composición actual -> MR1 -> MNI.

    Devuelve (máscara bool 3D, diccionario de resultados). La máscara de tejido
    se recibe por compatibilidad pero NO se usa como complemento directo:
    las medidas cerebral y ventricular no deben definirse una a partir de otra.
    """

    imagen = np.asarray(
        normalizada,
        dtype=np.float32
    )

    intracraneal = np.asarray(
        mascara_intracraneal,
        dtype=bool
    )

    tejido = np.asarray(
        mascara_tejido,
        dtype=bool
    )

    spacing = tuple(
        float(v) for v in tamano_voxel
    )

    if (
        imagen.ndim != 3
        or imagen.shape != intracraneal.shape
        or imagen.shape != tejido.shape
    ):
        raise ValueError(
            "La imagen y las máscaras deben ser 3D y tener idéntica forma"
        )

    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError(
            "tamano_voxel debe ser una tupla de tres medidas positivas"
        )

    prior_izq, prior_der = _mascaras_atlas_en_paciente(
        imagen,
        spacing,
        transformada_atlas
    )

    umbral_lcr = min(
        0.30,
        max(
            0.08,
            float(umbral) + MARGEN_UMBRAL
        )
    )

    sigma = tuple(
        0.6 / v for v in spacing
    )  # suavizado ligero de T1

    suave = ndimage.gaussian_filter(
        imagen,
        sigma=sigma
    )

    mascara_final = np.zeros(
        imagen.shape,
        dtype=bool
    )

    datos = {}
    componentes = 0
    calidad = []

    volumen_voxel = np.prod(spacing) / 1000.0

    # Planos medio-sagitales aproximados: las etiquetas del atlas se mantienen
    # separadas para que un mismo vóxel no sea sumado a ambos ventrículos.
    centro_izq = (
        float(np.mean(np.where(prior_izq)[2]))
        if prior_izq.any()
        else np.nan
    )

    centro_der = (
        float(np.mean(np.where(prior_der)[2]))
        if prior_der.any()
        else np.nan
    )

    corte_medio = (
        centro_izq + centro_der
    ) / 2.0

    posiciones_lr = np.arange(
        imagen.shape[2]
    )[None, None, :]

    for nombre, prior in (
        ("izquierdo", prior_izq),
        ("derecho", prior_der)
    ):

        if not np.any(prior):
            datos[f"volumen_{nombre}_ml"] = np.nan
            calidad.append("ATLAS_SIN_SOLAPAMIENTO")
            continue

        region = _roi_alrededor_de(
            prior,
            spacing,
            MARGEN_ROI_MM
        )

        # Se selecciona LCR dentro del entorno del atlas y del cerebro.
        candidatos = (
            (suave <= umbral_lcr)
            & region
            & intracraneal
        )

        if np.isfinite(corte_medio):

            if nombre == "izquierdo":

                lado = (
                    (posiciones_lr < corte_medio)
                    if centro_izq < centro_der
                    else (posiciones_lr > corte_medio)
                )

            else:

                lado = (
                    (posiciones_lr > corte_medio)
                    if centro_izq < centro_der
                    else (posiciones_lr < corte_medio)
                )

            candidatos &= lado

        seleccion, cantidad = _seleccionar_regiones(
            candidatos,
            prior,
            spacing
        )

        mascara_final |= seleccion
        componentes += cantidad

        datos[f"volumen_{nombre}_ml"] = float(
            seleccion.sum() * volumen_voxel
        )

        if cantidad == 0:
            calidad.append("SIN_SEMILLAS")

        else:
            cobertura = float(
                (seleccion & prior).sum()
                / max(prior.sum(), 1)
            )

            if cobertura < 0.12:
                calidad.append("COBERTURA_BAJA")

    volumen = float(
        mascara_final.sum() * volumen_voxel
    )

    if calidad:
        estado = "REVISAR_REGISTRO_O_SEGMENTACION"

    elif volumen < 2.0 or volumen > 120.0:
        estado = "REVISAR_VOLUMEN"

    else:
        estado = "REVISAR_MASCARA"

    # No reportar como volumen válido si no localizó alguna de las estructuras.
    if any(
        q in ("ATLAS_SIN_SOLAPAMIENTO", "SIN_SEMILLAS")
        for q in calidad
    ):
        volumen = np.nan

    info = {
        "volumen_ml": volumen,
        "estado": estado,
        "componentes": componentes,
        "umbral_utilizado": umbral_lcr,
        "metodo": "Harvard-Oxford + umbral T1 + componentes 3D",
        **datos,
    }

    return mascara_final, info


# =============================================================================
# FRACCIÓN VENTRICULAR (VENTRÍCULOS RELATIVIZADOS AL CRÁNEO)
# =============================================================================
# Volumen ventricular dividido por el volumen de la zona intracraneal, en %.
# La zona intracraneal es la misma región física en todas las visitas de un paciente
# (se lleva desde la primera visita con el registro rígido), así que el denominador es
# constante: dentro de un paciente, la fracción cambia solo si cambian los ventrículos.
# Entre pacientes, quita el efecto del tamaño de la cabeza (una cabeza más grande tiene
# ventrículos más grandes aunque no haya atrofia), igual que el nWBV para el cerebro.
# Al ser un cociente de dos volúmenes, no tiene unidades ni depende del tamaño del vóxel.
def fraccion_ventricular(volumen_ventricular_ml, volumen_intracraneal_ml):
    if not np.isfinite(volumen_ventricular_ml) or volumen_intracraneal_ml <= 0:
        return np.nan
    return 100.0 * volumen_ventricular_ml / volumen_intracraneal_ml
