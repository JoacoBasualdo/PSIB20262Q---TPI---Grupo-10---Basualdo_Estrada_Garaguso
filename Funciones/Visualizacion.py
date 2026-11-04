# =============================================================================
# VISUALIZACIÓN
# =============================================================================
# Todas las figuras del proyecto:
#   - Imágenes: cortes ortogonales, contornos de máscaras, ventrículos, comparación MR1/última.
#   - Histogramas: intensidades por visita y umbral de tejido.
#   - Evolución: volumen cerebral, ventrículos, fracción ventricular, análisis clínico.
#   - Validación: concordancia con OASIS y comparación con MRBrainS18.

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from Funciones.Procesamiento import numpy_a_sitk


# ----------------------------- IMÁGENES -----------------------------

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


# Muestra tres planos con el contorno de los ventrículos, para revisión visual.
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


# MR1 y la última visita sobre los mismos cortes anatómicos (la última se lleva a la grilla de MR1).
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


# ----------------------------- EVOLUCIÓN -----------------------------

# Un gráfico por paciente: cambio del volumen cerebral propio y del nWBV de OASIS desde MR1.
def guardar_graficos_evolucion(evolucion, carpeta_salida):
    """
    Genera un gráfico por paciente mostrando los cambios
    relativos del volumen cerebral y del nWBV de OASIS-2.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    rutas_guardadas = []

    for paciente, grupo in evolucion.groupby("paciente"):

        grupo = grupo.sort_values("anios_desde_primera")

        tiempo = grupo["anios_desde_primera"]

        cambio_volumen = grupo["cambio_desde_inicio_pct"]

        cambio_oasis = grupo["cambio_relativo_nwbv_oasis_pct"]

        # ----------------------------------------------------
        # CREACIÓN DEL GRÁFICO
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            tiempo,
            cambio_volumen,
            marker="o",
            linewidth=2,
            label="Volumen cerebral calculado (%)"
        )

        ax.plot(
            tiempo,
            cambio_oasis,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="nWBV OASIS-2 (%)"
        )

        # Línea de referencia: primera visita
        ax.axhline(
            y=0,
            linestyle=":",
            color="gray",
            linewidth=1
        )

        # ----------------------------------------------------
        # IDENTIFICACIÓN DE LAS VISITAS
        # ----------------------------------------------------

        for _, fila in grupo.iterrows():

            ax.annotate(
                f"MR{int(fila['visita'])}",
                (
                    fila["anios_desde_primera"],
                    fila["cambio_desde_inicio_pct"]
                ),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=9
            )

        # ----------------------------------------------------
        # FORMATO DEL GRÁFICO
        # ----------------------------------------------------

        ax.set_title(
            f"Evolución del volumen cerebral - {paciente}"
        )

        ax.set_xlabel(
            "Años desde la primera visita"
        )

        ax.set_ylabel(
            "Variación respecto de la primera visita (%)"
        )

        ax.grid(alpha=0.3)

        ax.legend()

        fig.tight_layout()

        # ----------------------------------------------------
        # GUARDAR EN RESULTADOS
        # ----------------------------------------------------

        ruta = os.path.join(
            carpeta_salida,
            f"evolucion_{paciente}.png"
        )

        fig.savefig(
            ruta,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        rutas_guardadas.append(ruta)

    return rutas_guardadas


# Un gráfico por paciente: cambio de tejido cerebral y de ventrículos desde MR1 (en %).
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


# Fracción ventricular de todos los pacientes en un mismo gráfico. Como está relativizada al
# volumen intracraneal de cada uno, se puede comparar entre pacientes con cabezas de distinto tamaño.
def graficar_fraccion_ventricular(tabla, ruta_salida):
    fig, eje = plt.subplots(figsize=(9, 5))
    for paciente, visitas in tabla.groupby("paciente", sort=False):
        visitas = visitas.sort_values("anios_desde_primera")
        eje.plot(visitas["anios_desde_primera"], visitas["fraccion_ventricular_pct"],
                 marker="o", label=f"{paciente} ({visitas['grupo'].iloc[0]})")
    eje.set(xlabel="Años desde la primera visita",
            ylabel="Ventrículos / volumen intracraneal (%)",
            title="Fracción ventricular por paciente")
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


# Un gráfico por paciente con tres paneles: tejido cerebral vs. nWBV de OASIS, ventrículos y
# CDR/MMSE. Usa la tabla por visita que devuelve Interpretacion.analizar_evolucion_clinica.
def graficar_analisis_clinico(detalle, carpeta_graficos):
    os.makedirs(carpeta_graficos, exist_ok=True)
    rutas_graficos = []

    for paciente, visitas in detalle.groupby("paciente", sort=False):
        visitas = visitas.sort_values("anios_desde_primera")
        primera = visitas.iloc[0]
        tiempo = visitas["anios_desde_primera"]

        fig, ejes = plt.subplots(
            3, 1, figsize=(10, 10), sharex=True
        )

        # Gráfico 1: volumen cerebral y nWBV
        ejes[0].plot(
            tiempo,
            visitas["cambio_cerebral_pct"],
            marker="o",
            label="Volumen cerebral propio (%)"
        )

        ejes[0].plot(
            tiempo,
            visitas["cambio_nwbv_oasis_pct"],
            marker="s",
            linestyle="--",
            label="nWBV OASIS-2 (%)"
        )

        ejes[0].set_ylabel("Cambio desde MR1 (%)")
        ejes[0].set_title("Tejido cerebral y nWBV de referencia")
        ejes[0].legend()

        # Gráfico 2: expansión ventricular estimada
        if visitas["cambio_ventricular_pct"].notna().any():
            ejes[1].plot(
                tiempo,
                visitas["cambio_ventricular_pct"],
                marker="o",
                label="Candidatos ventriculares (%)"
            )
            ejes[1].legend()
        else:
            ejes[1].text(
                0.5, 0.5,
                "No hay volumen ventricular inicial válido",
                transform=ejes[1].transAxes,
                ha="center",
                va="center"
            )

        ejes[1].set_ylabel("Cambio desde MR1 (%)")
        ejes[1].set_title(
            "Región ventricular estimada (sin validar)"
        )

        # Gráfico 3: evolución clínica
        eje_cdr = ejes[2]
        eje_mmse = eje_cdr.twinx()

        linea_cdr = eje_cdr.plot(
            tiempo,
            visitas["cdr"],
            marker="o",
            label="CDR"
        )

        linea_mmse = eje_mmse.plot(
            tiempo,
            visitas["mmse"],
            marker="s",
            linestyle="--",
            label="MMSE"
        )

        eje_cdr.set_ylabel("CDR (puntaje)")
        eje_mmse.set_ylabel("MMSE (puntaje)")
        eje_cdr.set_title("Indicadores clínicos de OASIS-2")

        eje_cdr.legend(
            linea_cdr + linea_mmse,
            ["CDR", "MMSE"],
            loc="best"
        )

        eje_cdr.set_xlabel("Años desde la primera visita")

        for eje in ejes:
            eje.grid(alpha=0.25)
            eje.axvline(
                float(tiempo.iloc[0]),
                linestyle=":",
                linewidth=1
            )

        ejes[2].set_xticks(tiempo.to_numpy())

        ejes[2].set_xticklabels([
            f"MR{int(v)}\n{t:.2f} a"
            for v, t in zip(visitas["visita"], tiempo)
        ])

        fig.suptitle(
            f"Análisis morfológico-clínico - "
            f"{paciente} ({primera['grupo']})"
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        ruta = os.path.join(
            carpeta_graficos,
            f"clinico_integrado_{paciente}.png"
        )

        fig.savefig(
            ruta,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close(fig)
        rutas_graficos.append(ruta)

    return rutas_graficos


# ----------------------------- VALIDACIÓN -----------------------------

# Concordancia con OASIS: nWBV de cada paciente a lo largo del tiempo, nuestro (rojo) y OASIS (negro)
def graficar_concordancia_oasis(datos, ruta):
    pacientes = list(datos["paciente"].unique())
    fig, ejes = plt.subplots(1, len(pacientes), figsize=(4.5 * len(pacientes), 4.5), squeeze=False)
    for eje, paciente in zip(ejes[0], pacientes):
        filas = datos[datos["paciente"] == paciente].sort_values("anios_desde_primera")
        anios = filas["anios_desde_primera"].values
        for columna, color, nombre in (("nwbv_propio", "tab:red", "Propio"), ("nwbv_oasis", "black", "OASIS")):
            valores = filas[columna].values
            eje.plot(anios, valores, "o", color=color, label=nombre)
            if len(filas) >= 2:
                a, b = np.polyfit(anios, valores, 1)
                eje.plot(anios, a * anios + b, "-", color=color, alpha=0.6)
        eje.set_title(f"{paciente} ({filas['grupo'].iloc[0]})")
        eje.set_xlabel("Años desde la primera visita")
        eje.set_ylabel("nWBV")
        eje.grid(alpha=0.3)
        eje.legend()
    fig.suptitle("Concordancia con OASIS: nWBV propio vs. nWBV publicado", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


# MRBrainS18: 3 cortes axiales con el contorno manual (verde) y el nuestro (rojo)
def graficar_validacion_mrbrains(t1_manual, ventriculos_manual, ventriculos_propios, sujeto, ruta):
    cortes_con_ventriculos = np.nonzero(ventriculos_manual.any(axis=(0, 1)))[0]
    if len(cortes_con_ventriculos) == 0:
        return None
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
    return ruta


# Histograma de la cabeza antes y después de corregir el bias field (N4).
# Cada uno se divide por su mediana para que queden en la misma escala, porque N4
# cambia la escala general de la imagen. Si la corrección funcionó, los picos de
# cada tejido quedan más altos y angostos (el mismo tejido tiene un brillo más parejo).
def graficar_histograma_bias(antes, despues, titulo="Histograma antes y después de N4", bins=200):
    antes = antes / np.median(antes)
    despues = despues / np.median(despues)
    limite = np.percentile(np.concatenate([antes, despues]), 99.5)

    fig, eje = plt.subplots(figsize=(10, 5))
    eje.hist(antes, bins=bins, range=(0, limite), density=True, histtype='step', linewidth=1.5, label="Antes de N4")
    eje.hist(despues, bins=bins, range=(0, limite), density=True, histtype='step', linewidth=1.5, label="Después de N4")
    eje.set_title(titulo)
    eje.set_xlabel("Intensidad relativa (dividida por la mediana)")
    eje.set_ylabel("Densidad de vóxeles")
    eje.legend()
    plt.tight_layout()
    return fig


# =============================================================================
# TEXTURA DEL TEJIDO PERIVENTRICULAR
# =============================================================================

# Tres planos con los ventrículos (contorno amarillo) y la banda periventricular donde se
# mide la textura (en magenta, semitransparente). Los cortes se eligen donde la banda tiene
# más vóxeles, igual que en visualizar_ventriculos.
def visualizar_banda_textura(normalizada, mascara_ventricular, mascara_banda, titulo,
                             tamano_voxel=(1.0, 1.0, 1.25)):
    referencia = mascara_banda if np.any(mascara_banda) else mascara_ventricular
    if np.any(referencia):
        indice_sagital = int(np.argmax(referencia.sum(axis=(0, 1))))
        indice_coronal = int(np.argmax(referencia.sum(axis=(1, 2))))
        indice_axial = int(np.argmax(referencia.sum(axis=(0, 2))))
    else:
        indice_coronal, indice_axial, indice_sagital = (dim // 2 for dim in normalizada.shape)

    def cortar(volumen):
        return [
            np.rot90(volumen[:, :, indice_sagital]),
            np.flipud(volumen[indice_coronal, :, :]),
            np.flipud(volumen[:, indice_axial, :]),
        ]

    from matplotlib.colors import ListedColormap
    nombres = ["Sagital", "Coronal", "Axial"]
    aspectos = [tamano_voxel[1] / tamano_voxel[0], tamano_voxel[1] / tamano_voxel[2],
                tamano_voxel[0] / tamano_voxel[2]]

    fig, ejes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"{titulo} - Banda periventricular donde se mide la textura")
    for eje, imagen, ventriculos, banda, nombre, aspecto in zip(
            ejes, cortar(normalizada), cortar(mascara_ventricular.astype(float)),
            cortar(mascara_banda.astype(float)), nombres, aspectos):
        eje.imshow(imagen, cmap="gray", aspect=aspecto)
        eje.imshow(np.ma.masked_where(banda < 0.5, banda), cmap=ListedColormap(["magenta"]), alpha=0.45,
                   aspect=aspecto, vmin=0, vmax=1)
        if np.any(ventriculos):
            eje.contour(ventriculos, levels=[0.5], colors="yellow", linewidths=1.2)
        eje.set_title(nombre)
        eje.axis("off")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Line2D([0], [0], color="yellow", lw=2), Patch(color="magenta", alpha=0.45)],
               ["Ventrículos", "Banda periventricular"], loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# La GLCM como imagen: cada casilla (i, j) es la probabilidad de que un vóxel de la banda
# tenga nivel de gris i y su vecino nivel j. Si la textura es homogénea, la probabilidad se
# concentra cerca de la diagonal (vecinos parecidos); si es heterogénea, se desparrama.
# Se muestra en escala logarítmica para que se vean también las casillas con pocos pares.
def graficar_glcm(glcm, titulo="Matriz de co-ocurrencia (GLCM)"):
    fig, eje = plt.subplots(figsize=(6, 5.5))
    imagen = eje.imshow(np.log10(glcm + 1e-6), cmap="viridis", origin="lower")
    eje.plot([0, glcm.shape[0] - 1], [0, glcm.shape[0] - 1], color="white", linestyle=":", linewidth=1)
    eje.set(title=titulo, xlabel="Nivel de gris del vecino (j)", ylabel="Nivel de gris del vóxel (i)")
    fig.colorbar(imagen, ax=eje, label="log10(probabilidad)")
    fig.tight_layout()
    return fig


# Un gráfico por paciente: cambio de las características de textura desde MR1 (en %),
# con el cambio del volumen ventricular como referencia.
def graficar_evolucion_textura(tabla, carpeta_salida,
                               caracteristicas=("contraste", "homogeneidad", "energia", "entropia")):
    os.makedirs(carpeta_salida, exist_ok=True)
    marcadores = ["o", "s", "^", "D", "v"]
    rutas = []
    for paciente, visitas in tabla.groupby("paciente", sort=False):
        visitas = visitas.sort_values("anios_desde_primera")
        fig, eje = plt.subplots(figsize=(9, 5))
        for caracteristica, marcador in zip(caracteristicas, marcadores):
            columna = f"cambio_{caracteristica}_pct"
            if columna in visitas.columns:
                eje.plot(visitas["anios_desde_primera"], visitas[columna], marker=marcador,
                         label=caracteristica.capitalize())
        if "volumen_ventricular_ml" in visitas.columns:
            inicial = visitas["volumen_ventricular_ml"].iloc[0]
            if np.isfinite(inicial) and inicial > 0:
                eje.plot(visitas["anios_desde_primera"],
                         100 * (visitas["volumen_ventricular_ml"] - inicial) / inicial,
                         color="gray", linestyle="--", label="Volumen ventricular (referencia)")
        eje.axhline(0, color="gray", linewidth=1, linestyle=":")
        eje.set(xlabel="Años desde la primera visita", ylabel="Cambio respecto de MR1 (%)",
                title=f"Textura periventricular - {paciente} ({visitas['grupo'].iloc[0]})")
        eje.grid(alpha=0.3)
        eje.legend()
        fig.tight_layout()
        ruta = os.path.join(carpeta_salida, f"textura_{paciente}.png")
        fig.savefig(ruta, dpi=170, bbox_inches="tight")
        plt.close(fig)
        rutas.append(ruta)
    return rutas


# Tasa de cambio por año (% respecto de MR1) de algunas características, para todos los
# pacientes en un mismo gráfico de barras, con el grupo de cada uno en el eje.
def graficar_tasas_textura(resumen, ruta_salida, caracteristicas=("contraste", "homogeneidad", "entropia")):
    columnas = [f"tasa_{c}_pct_anio" for c in caracteristicas if f"tasa_{c}_pct_anio" in resumen.columns]
    fig, eje = plt.subplots(figsize=(10, 5))
    posiciones = np.arange(len(resumen))
    ancho = 0.8 / max(len(columnas), 1)
    for k, columna in enumerate(columnas):
        eje.bar(posiciones + (k - (len(columnas) - 1) / 2) * ancho, resumen[columna], width=ancho,
                label=columna.replace("tasa_", "").replace("_pct_anio", "").capitalize())
    eje.set_xticks(posiciones)
    eje.set_xticklabels([f"{p}\n{g}" for p, g in zip(resumen["paciente"], resumen["grupo"])])
    eje.axhline(0, color="gray", linewidth=1)
    eje.set(ylabel="Tasa de cambio (% por año)", title="Tasa de cambio de la textura periventricular por paciente")
    eje.grid(axis="y", alpha=0.3)
    eje.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida