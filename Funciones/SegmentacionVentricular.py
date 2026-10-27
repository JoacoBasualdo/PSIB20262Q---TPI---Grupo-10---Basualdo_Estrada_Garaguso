
"""Segmentación exploratoria de ventrículos laterales guiada por atlas anatómico.

Método: Harvard-Oxford subcortical (izquierda/derecha), proyectado al espacio
nativo mediante las transformaciones ya calculadas por Features.py; umbral de
LCR T1 local; conectividad 3D con semilla anatómica por hemisferio.

La máscara NO es una verdad de terreno. Inspeccionar en el PDF antes de usar
sus volúmenes para conclusiones clínicas.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage


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

    import SimpleITK as sitk
    from Funciones.Acondicionamiento import numpy_a_sitk

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
# COMPARACIÓN LONGITUDINAL (SE MANTIENE LA MISMA INTERFAZ)
# =============================================================================

def calcular_cambios_ventriculares(datos):

    tabla = datos.copy().sort_values(
        ["paciente", "anios_desde_primera", "visita"]
    ).reset_index(drop=True)

    grupos = tabla.groupby(
        "paciente",
        sort=False
    )

    # No saltarse el NaN de MR1: una referencia inválida no debe sustituirse
    # silenciosamente por MR2 al comparar las visitas.
    inicial_vent = grupos[
        "volumen_ventricular_ml"
    ].transform(
        lambda x: x.iloc[0]
    )

    inicial_cerebro = grupos[
        "volumen_tejido_ml"
    ].transform("first")

    tiempo = (
        tabla["anios_desde_primera"]
        - grupos["anios_desde_primera"].transform("first")
    )

    tabla["cambio_ventricular_ml"] = (
        tabla["volumen_ventricular_ml"]
        - inicial_vent
    )

    tabla["cambio_ventricular_pct"] = np.where(
        inicial_vent > 0,
        100 * tabla["cambio_ventricular_ml"] / inicial_vent,
        np.nan
    )

    tabla["cambio_cerebral_pct"] = np.where(
        inicial_cerebro > 0,
        100 * (
            tabla["volumen_tejido_ml"] - inicial_cerebro
        ) / inicial_cerebro,
        np.nan
    )

    tabla["tasa_ventricular_ml_anio"] = np.where(
        tiempo > 0,
        tabla["cambio_ventricular_ml"]
        / tiempo.replace(0, np.nan),
        np.nan
    )

    return tabla


# =============================================================================
# VISUALIZACIONES (EN ESTE MISMO ARCHIVO, COMO EN EL PROYECTO ACTUAL)
# =============================================================================

def visualizar_ventriculos(
    normalizada,
    mascara_ventricular,
    titulo,
    tamano_voxel=(1.0, 1.0, 1.25)
):
    """Muestra tres planos con el contorno estimado, para revisión visual."""

    if np.any(mascara_ventricular):

        indice_sagital = int(
            np.argmax(
                mascara_ventricular.sum(axis=(0, 1))
            )
        )

        indice_coronal = int(
            np.argmax(
                mascara_ventricular.sum(axis=(1, 2))
            )
        )

        indice_axial = int(
            np.argmax(
                mascara_ventricular.sum(axis=(0, 2))
            )
        )

    else:
        indice_coronal, indice_axial, indice_sagital = (
            dim // 2 for dim in normalizada.shape
        )

    cortes = [
        (
            "Sagital",
            np.rot90(normalizada[:, :, indice_sagital]),
            np.rot90(mascara_ventricular[:, :, indice_sagital]),
            tamano_voxel[1] / tamano_voxel[0]
        ),
        (
            "Coronal",
            np.flipud(normalizada[indice_coronal, :, :]),
            np.flipud(mascara_ventricular[indice_coronal, :, :]),
            tamano_voxel[1] / tamano_voxel[2]
        ),
        (
            "Axial",
            np.flipud(normalizada[:, indice_axial, :]),
            np.flipud(mascara_ventricular[:, indice_axial, :]),
            tamano_voxel[0] / tamano_voxel[2]
        ),
    ]

    fig, ejes = plt.subplots(
        1, 3,
        figsize=(14, 5)
    )

    fig.suptitle(
        f"{titulo} - Ventrículos laterales (segmentación experimental)"
    )

    for eje, (nombre, imagen, mascara, aspecto) in zip(
        ejes, cortes
    ):

        eje.imshow(
            imagen,
            cmap="gray",
            aspect=aspecto
        )

        if np.any(mascara):
            eje.contour(
                mascara.astype(float),
                levels=[0.5],
                colors="yellow",
                linewidths=1.5
            )

        eje.set_title(nombre)
        eje.axis("off")

    fig.tight_layout()

    return fig


def graficar_evolucion_ventricular(
    tabla,
    carpeta_salida
):
    """Mantiene el formato de los gráficos de la etapa ventricular."""

    os.makedirs(
        carpeta_salida,
        exist_ok=True
    )

    rutas = []

    for paciente, visitas in tabla.groupby("paciente"):

        visitas = visitas.sort_values(
            "anios_desde_primera"
        )

        fig, eje = plt.subplots(figsize=(9, 5))

        eje.plot(
            visitas["anios_desde_primera"],
            visitas["cambio_cerebral_pct"],
            marker="o",
            label="Cambio de tejido cerebral (%)"
        )

        eje.plot(
            visitas["anios_desde_primera"],
            visitas["cambio_ventricular_pct"],
            marker="s",
            linestyle="--",
            label="Cambio ventricular estimado (%)"
        )

        eje.axhline(
            0,
            color="gray",
            linewidth=1,
            linestyle=":"
        )

        eje.set(
            xlabel="Años desde la primera visita",
            ylabel="Cambio respecto de MR1 (%)",
            title=f"Cambios cerebrales y ventriculares - {paciente}"
        )

        eje.grid(alpha=0.3)
        eje.legend()

        fig.tight_layout()

        ruta = os.path.join(
            carpeta_salida,
            f"cerebro_ventriculos_{paciente}.png"
        )

        fig.savefig(
            ruta,
            dpi=170,
            bbox_inches="tight"
        )

        plt.close(fig)

        rutas.append(ruta)

    return rutas


# =============================================================================
# EXPORTACIÓN DE MÁSCARAS VENTRICULARES 3D
# =============================================================================

def guardar_mascara_ventricular(
    mascara_ventricular,
    ruta_mri_original,
    ruta_salida
):
    """Guarda la máscara binaria 3D en NIfTI comprimido (.nii.gz).

    Usa la affine del primer estudio mpr de la visita, ya que el promedio
    y la segmentación están sobre su misma grilla de vóxeles. Esto permite
    recuperar la máscara y superponerla con esa adquisición en un visor NIfTI.
    El archivo contiene 0 (fondo) y 1 (candidatos ventriculares).

    El atlas y el registro de la segmentación siguen siendo experimentales;
    esta función solo serializa la máscara, no valida el resultado.
    """

    import nibabel as nib

    mascara = np.asarray(
        mascara_ventricular,
        dtype=np.uint8
    )

    if mascara.ndim != 3:
        raise ValueError(
            "La máscara ventricular debe tener tres dimensiones"
        )

    imagen_original = nib.load(
        str(ruta_mri_original)
    )

    if tuple(imagen_original.shape[:3]) != tuple(mascara.shape):
        raise ValueError(
            "La máscara y el archivo mpr de referencia no tienen la misma grilla: "
            f"{mascara.shape} frente a {imagen_original.shape[:3]}"
        )

    cabecera = imagen_original.header.copy()
    cabecera.set_data_dtype(np.uint8)
    cabecera.set_data_shape(mascara.shape)

    imagen_mascara = nib.Nifti1Image(
        mascara,
        imagen_original.affine,
        header=cabecera
    )

    os.makedirs(
        os.path.dirname(os.path.abspath(ruta_salida)),
        exist_ok=True
    )

    nib.save(
        imagen_mascara,
        str(ruta_salida)
    )

    return ruta_salida


# =============================================================================
# COMPARACIÓN VISUAL: MR1 Y ÚLTIMA VISITA EN CORTES EQUIVALENTES
# =============================================================================

def visualizar_comparacion_ventricular(
    imagen_inicial,
    mascara_inicial,
    tamano_voxel_inicial,
    imagen_final,
    mascara_final,
    tamano_voxel_final,
    transformada_final_a_inicial,
    paciente,
    nombre_inicial,
    nombre_final,
):
    """Arma una figura de 3 planos x 2 visitas en una sola grilla anatómica.

    La transformación entre visitas devuelta por Features.py relaciona
    coordenadas físicas de la visita final (fija) con las de MR1 (móvil).
    Para remuestrear la imagen final en la grilla MR1, SimpleITK requiere
    el mapeo inverso: salida MR1 -> entrada visita final.

    Se eligen los índices de corte a partir de la unión de ambas máscaras
    ya alineadas; cada par MR1/final utiliza EXACTAMENTE el mismo índice en
    la grilla MR1. El registro es rígido: el parecido anatómico es aproximado
    y las variaciones reales de la anatomía se mantienen visibles.
    """

    import SimpleITK as sitk
    from Funciones.Acondicionamiento import numpy_a_sitk

    primera = np.asarray(imagen_inicial)
    ultima = np.asarray(imagen_final)

    mask_primera = np.asarray(
        mascara_inicial,
        dtype=bool
    )

    mask_ultima = np.asarray(
        mascara_final,
        dtype=bool
    )

    if (
        primera.shape != mask_primera.shape
        or ultima.shape != mask_ultima.shape
        or primera.ndim != 3
        or ultima.ndim != 3
    ):
        raise ValueError(
            "Cada imagen 3D debe coincidir en tamaño con su máscara"
        )

    referencia_sitk = numpy_a_sitk(
        primera,
        tamano_voxel_inicial
    )

    final_sitk = numpy_a_sitk(
        ultima,
        tamano_voxel_final
    )

    mascara_final_sitk = numpy_a_sitk(
        mask_ultima.astype(np.float32),
        tamano_voxel_final
    )

    if transformada_final_a_inicial is None:

        # Permite también el caso trivial de una sola visita.
        if (
            primera.shape != ultima.shape
            or not np.allclose(
                tamano_voxel_inicial,
                tamano_voxel_final
            )
        ):
            raise ValueError(
                "Se necesita el registro de la última visita a MR1"
            )

        inversa = sitk.Transform(
            3,
            sitk.sitkIdentity
        )

    else:
        inversa = transformada_final_a_inicial.GetInverse()

    ultima_alineada = sitk.GetArrayFromImage(
        sitk.Resample(
            final_sitk,
            referencia_sitk,
            inversa,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32
        )
    )

    mascara_final_alineada = (
        sitk.GetArrayFromImage(
            sitk.Resample(
                mascara_final_sitk,
                referencia_sitk,
                inversa,
                sitk.sitkNearestNeighbor,
                0.0,
                sitk.sitkUInt8
            )
        ) > 0
    )

    # Se usan cortes coincidentes, no los cortes más grandes de cada visita
    # por separado (que podrían corresponder a distintas posiciones).
    union = mask_primera | mascara_final_alineada

    if union.any():

        sagital = int(
            np.argmax(union.sum(axis=(0, 1)))
        )

        coronal = int(
            np.argmax(union.sum(axis=(1, 2)))
        )

        axial = int(
            np.argmax(union.sum(axis=(0, 2)))
        )

    else:
        coronal, axial, sagital = (
            int(n // 2)
            for n in primera.shape
        )

    def corte(volumen, mascara, plano):

        if plano == "Sagital":

            return (
                np.rot90(volumen[:, :, sagital]),
                np.rot90(mascara[:, :, sagital]),
                tamano_voxel_inicial[1] / tamano_voxel_inicial[0]
            )

        if plano == "Coronal":

            return (
                np.flipud(volumen[coronal, :, :]),
                np.flipud(mascara[coronal, :, :]),
                tamano_voxel_inicial[1] / tamano_voxel_inicial[2]
            )

        return (
            np.flipud(volumen[:, axial, :]),
            np.flipud(mascara[:, axial, :]),
            tamano_voxel_inicial[0] / tamano_voxel_inicial[2]
        )

    fig, ejes = plt.subplots(
        3, 2,
        figsize=(10, 13)
    )

    fig.suptitle(
        f"Comparación ventricular - {paciente}\n"
        f"Cortes equivalentes en el espacio de {nombre_inicial} (registro rígido)",
        fontsize=13
    )

    for fila, plano in enumerate(
        ("Sagital", "Coronal", "Axial")
    ):

        for columna, (imagen, mascara, nombre) in enumerate(
            (
                (primera, mask_primera, nombre_inicial),
                (ultima_alineada, mascara_final_alineada, nombre_final),
            )
        ):

            eje = ejes[fila, columna]

            fondo, seleccion, aspecto = corte(
                imagen,
                mascara,
                plano
            )

            eje.imshow(
                fondo,
                cmap="gray",
                aspect=aspecto,
                vmin=0,
                vmax=1
            )

            if seleccion.any():

                eje.contour(
                    seleccion.astype(float),
                    levels=[0.5],
                    colors="yellow",
                    linewidths=1.2
                )

            eje.set_title(
                f"{plano} - {nombre}",
                fontsize=11
            )

            eje.axis("off")

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    return fig
