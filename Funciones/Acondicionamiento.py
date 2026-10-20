import numpy as np
# Es una libreria en C++ que desarrollaron la Biblioteca Nacional de Medicina de EE.UU. y varias universidades para el procesamiento de imágenes.
import SimpleITK as sitk
from scipy import ndimage
# Convierte la matriz de NumPy en una imagen de Simple ITK dado que permite saber donde se encuentra en el espacio físico para realizar el promediado.
def numpy_a_sitk(volumen, tamano_voxel):
    imagen = sitk.GetImageFromArray(volumen.astype(np.float32))
    imagen.SetSpacing(tuple(float(p) for p in reversed(tamano_voxel)))
    return imagen

# Determina cúanto hay que girar y mover una imagen para que quede una encima de la otra
# Primero, se pone el centro de una imagen sobre el centro de la otra. Segundo, .Tecero, compara un 5% de los pixeles al azar dado que sino el costo computacional es muy elevado. Luego, itera con un descenso de gradiente y frena cuando los pasos son muy pequeños o luego de 200 iteraciones. 

# Puntaje de similitud:
#   1. Superponer las imágenes en la posición actual.
#   2. Agrupar los píxeles según su brillo en la fija (50 grupos).
#   3. Calcular la entropía de cada grupo con los brillos de la móvil
#      y promediarlas.
#   4. Calcular la entropía de la móvil completa.
#   5. Puntaje = entropía móvil completa - entropía promedio de grupos.
#   6. Moverse un poco y repetir, quedándose con lo que sube el puntaje.
#
# Entropía = suma de proporción x log2(1 / proporción) de cada brillo.
# Alta = brillos mezclados; baja = casi puros.
#
# Ejemplo con 6 píxeles:
#
#   ALINEADAS
#   Fija:   10  10  50  50  90  90
#   Móvil:   5   5  30  30  60  60
#   Grupos: fija=10 -> (5, 5)  |  fija=50 -> (30, 30)  |  fija=90 -> (60, 60)
#   Entropía móvil completa = 1.58  |  Entropía grupos = 0 (puros)
#   Puntaje = 1.58 - 0 = 1.58  -> ALTO
#
#   CORRIDAS UN PÍXEL
#   Fija:   10  10  50  50  90  90
#   Móvil:   0   5   5  30  30  60
#   Grupos: fija=10 -> (0, 5)  |  fija=50 -> (5, 30)  |  fija=90 -> (30, 60)
#   Entropía móvil completa = 1.92  |  Entropía grupos = 1 (mezclados)
#   Puntaje = 1.92 - 1 = 0.92  -> BAJO
#
# Alineadas: grupos casi puros -> entropía baja -> puntaje ALTO.
# Desalineadas: grupos mezclados -> entropía alta -> puntaje BAJO.
def registrar_rigido(fija, movil):
    transformada_inicial = sitk.CenteredTransformInitializer(
        fija, movil, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registro = sitk.ImageRegistrationMethod()
    registro.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registro.SetMetricSamplingStrategy(registro.RANDOM)
    registro.SetMetricSamplingPercentage(0.05, seed=42)
    registro.SetInterpolator(sitk.sitkLinear)
    registro.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200
    )
    registro.SetOptimizerScalesFromPhysicalShift()
    registro.SetShrinkFactorsPerLevel([4, 2, 1])
    registro.SetSmoothingSigmasPerLevel([2, 1, 0])
    registro.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registro.SetInitialTransform(transformada_inicial, inPlace=False)

    transformada = registro.Execute(fija, movil)
    return transformada

# Resume en cuanto se desplazó la imagen para que queden superpuestas
def movimiento_de_transformada(transformada):
    parametros = transformada.GetParameters()
    rotacion_grados = np.degrees(np.abs(parametros[:3])).max()
    traslacion_mm = np.linalg.norm(parametros[3:6])
    return rotacion_grados, traslacion_mm

# Toma mpr-1 como referencia, las alinea y promedia vóxel a vóxel. Se va cancelando el ruido
# La idea de promediar aparece en el paper de OASIS pero se basaron en un atlas para la comparación.
def promediar_adquisiciones(volumenes, tamano_voxel):
    fija = numpy_a_sitk(volumenes[0], tamano_voxel)
    alineados = [volumenes[0].astype(np.float32)]
    movimientos = []

    for volumen in volumenes[1:]:
        movil = numpy_a_sitk(volumen, tamano_voxel)
        transformada = registrar_rigido(fija, movil)
        remuestreada = sitk.Resample(movil, fija, transformada, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        alineados.append(sitk.GetArrayFromImage(remuestreada))
        movimientos.append(movimiento_de_transformada(transformada))

    promedio = np.mean(alineados, axis=0)
    return promedio, movimientos


def filtro_gaussiano(volumen, tamano_voxel, sigma_mm=1.0):
    sigma_por_eje = [sigma_mm / paso for paso in tamano_voxel]
    return ndimage.gaussian_filter(volumen, sigma=sigma_por_eje)
 
 
def filtro_mediana(volumen, tamano=3):
    return ndimage.median_filter(volumen, size=tamano)

def rellenar_huecos_por_corte(mascara):
    rellena = mascara.copy()
    for eje in range(3):
        for indice in range(rellena.shape[eje]):
            corte = np.take(rellena, indice, axis=eje)
            corte_relleno = ndimage.binary_fill_holes(corte)
            if eje == 0:
                rellena[indice, :, :] = corte_relleno
            elif eje == 1:
                rellena[:, indice, :] = corte_relleno
            else:
                rellena[:, :, indice] = corte_relleno
    return rellena
 
 
def mascara_cabeza(volumen, tamano_voxel):
    imagen = numpy_a_sitk(volumen, tamano_voxel)
    mascara = sitk.GetArrayFromImage(sitk.OtsuThreshold(imagen, 0, 1, 200)).astype(bool)
 
    mascara = ndimage.binary_closing(mascara, iterations=3)
    mascara = rellenar_huecos_por_corte(mascara)
 
    etiquetas, cantidad = ndimage.label(mascara)
    if cantidad > 1:
        tamanos = ndimage.sum(mascara, etiquetas, range(1, cantidad + 1))
        mascara = etiquetas == (np.argmax(tamanos) + 1)
 
    return mascara
 
# Se resuelve el problema de que el resonador no genera una imagen con luminosidad pareja a tejidos que son similares. Homogeneizarlos es fundamental para un posterior procesado umbralizado.
# DOnde ve que hay una curva suave de cambios se estima que es por la iluminacion. Cuaando hay cambio brusco se da por cambios de tejido. Entonces homogeinza los cambios individuales estimando el bias de esa region
#Ellos lo resolvieron con un modelo cuadrático ajustado a un phantom, un objeto de prueba que se escanea en el mismo equipo. Ustedes no tienen ese phantom, así que necesitan un método que estime la corrección desde la propia imagen del paciente. Eso es lo que hace N4.

def corregir_bias_field(volumen, tamano_voxel, factor_reduccion=4, iteraciones=(50, 50, 50, 50)):
    imagen = numpy_a_sitk(volumen, tamano_voxel)
    mascara = mascara_cabeza(volumen, tamano_voxel)
 
    mascara_sitk = sitk.GetImageFromArray(mascara.astype(np.uint8))
    mascara_sitk.CopyInformation(imagen)
 
    imagen_reducida = sitk.Shrink(imagen, [factor_reduccion] * 3)
    mascara_reducida = sitk.Shrink(mascara_sitk, [factor_reduccion] * 3)
 
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iteraciones))
    corrector.Execute(imagen_reducida, mascara_reducida)
 
    bias = sitk.Exp(corrector.GetLogBiasFieldAsImage(imagen))
    corregida = imagen / bias
 
    return sitk.GetArrayFromImage(corregida), sitk.GetArrayFromImage(bias), mascara


    
# Usando solos los pixeles de la cabeza busca un valor que deje por debajo al 1% y otro al 99%. Entonces, estira la escala para que el primero valga 0 y el segundo valga 1. Todo lo que queda fuera de ese rango y el fondo lo pone en cero. 
# Para determinan que sean solo de la cabeza, en corregir bias se usa una máscara utilizando el método de Otsu que deteremina los píxeles que forman parte de la cabeza. 
def normalizar_intensidad(volumen, mascara, percentil_inferior=1, percentil_superior=99):
    valor_bajo, valor_alto = np.percentile(volumen[mascara], [percentil_inferior, percentil_superior])
    normalizada = (volumen - valor_bajo) / (valor_alto - valor_bajo)
    normalizada = np.clip(normalizada, 0, 1)
    normalizada[~mascara] = 0
    return normalizada, valor_bajo, valor_alto