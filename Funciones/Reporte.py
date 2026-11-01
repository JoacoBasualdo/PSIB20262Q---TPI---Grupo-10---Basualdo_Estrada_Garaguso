# =============================================================================
# REPORTE
# =============================================================================
# Salidas del procesamiento:
#   - PDF por paciente: guarda todo lo que se imprime en la terminal y las figuras.
#   - Máscaras ventriculares 3D en NIfTI (.nii.gz), para abrirlas en un visor.
#   - Etapas: cada texto y figura queda marcado con la etapa del pipeline en la que se
#     generó (Lectura, Procesamiento, ...). La interfaz usa esa marca para mostrar los
#     resultados en una pestaña por etapa; el PDF no cambia.

import os
import sys
import numpy as np
import nibabel as nib
import textwrap
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Las figuras quedan abiertas hasta armar el PDF al final; se desactiva el aviso de "más de 20 figuras"
plt.rcParams["figure.max_open_warning"] = 0

TAMANIO_PAGINA = (8.27, 11.69)
MARGEN_SUPERIOR = 0.95
MARGEN_INFERIOR = 0.05
MARGEN_IZQUIERDO = 0.06
ALTO_LINEA = 0.013

_elementos = []
_etiquetas = []          # etapa de cada elemento de _elementos (misma longitud)
_texto_actual = []
_consola = sys.stdout
_etapa_actual = None
_subtitulo_actual = None
_aviso_etapa = None      # función que se llama cada vez que cambia la etapa (la usa la interfaz)


class _Duplicador:
    def write(self, texto):
        _consola.write(texto)
        _texto_actual.append(texto)

    def flush(self):
        _consola.flush()

    def __getattr__(self, nombre):
        return getattr(_consola, nombre)


def _agregar(tipo, contenido):
    _elementos.append((tipo, contenido))
    _etiquetas.append(_etapa_actual)


def _cerrar_texto():
    texto = "".join(_texto_actual).strip("\n")
    if texto:
        _agregar("texto", texto)
    _texto_actual.clear()


def iniciar_registro():
    global _etapa_actual, _subtitulo_actual
    _elementos.clear()
    _etiquetas.clear()
    _texto_actual.clear()
    _etapa_actual = None
    _subtitulo_actual = None
    sys.stdout = _Duplicador()


def agregar_subtitulo(texto):
    global _subtitulo_actual
    _cerrar_texto()
    _agregar("subtitulo", texto)
    _subtitulo_actual = texto
    _consola.write(f"\n=== {texto} ===\n")


def agregar_figura(fig):
    _cerrar_texto()
    _agregar("figura", fig)


# Agrega al reporte un gráfico que ya se guardó como imagen (los de interpretación y
# validación se guardan directamente en Resultados/Graficos).
def agregar_imagen(ruta):
    imagen = plt.imread(ruta)
    alto, ancho = imagen.shape[:2]
    fig = plt.figure(figsize=(ancho / 100, alto / 100), dpi=100)
    eje = fig.add_axes([0, 0, 1, 1])
    eje.imshow(imagen)
    eje.axis("off")
    agregar_figura(fig)


# A partir de acá, todo lo que se imprima o se agregue queda marcado con esta etapa.
def marcar_etapa(nombre):
    global _etapa_actual
    if nombre == _etapa_actual:
        return
    _cerrar_texto()
    _etapa_actual = nombre
    if _aviso_etapa is not None:
        _aviso_etapa(nombre, _subtitulo_actual)


# Registra una función que se llama con (etapa, subtítulo actual) cada vez que cambia la etapa
def al_cambiar_etapa(funcion):
    global _aviso_etapa
    _aviso_etapa = funcion


# Cambia hacia dónde se escribe lo que se imprime (la interfaz lo manda a su consola)
def cambiar_consola(nueva):
    global _consola
    _consola = nueva


# Agrupa lo registrado por etapa, en el orden de 'etapas'. Dentro de cada etapa se repite
# el subtítulo vigente (por ejemplo "Visita OAS2_0127_MR2") para saber a qué visita
# corresponde cada texto o figura.
def elementos_por_etapa(etapas):
    _cerrar_texto()
    grupos = {etapa: [] for etapa in etapas}
    ultimo_subtitulo = {etapa: None for etapa in etapas}
    subtitulo = None
    for (tipo, contenido), etapa in zip(_elementos, _etiquetas):
        if tipo == "subtitulo":
            subtitulo = contenido
            continue
        if etapa not in grupos:
            continue
        if subtitulo is not None and ultimo_subtitulo[etapa] != subtitulo:
            grupos[etapa].append(("subtitulo", subtitulo))
            ultimo_subtitulo[etapa] = subtitulo
        grupos[etapa].append((tipo, contenido))
    return [(etapa, grupos[etapa]) for etapa in etapas]


def _guardar_pagina(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def _portada(titulo):
    fig = plt.figure(figsize=TAMANIO_PAGINA)
    fig.text(0.5, 0.6, titulo, ha="center", fontsize=24, weight="bold")
    fig.text(0.5, 0.55, datetime.now().strftime("%d/%m/%Y %H:%M"), ha="center", fontsize=12)
    return fig


def _partir_lineas(texto, ancho=100):
    lineas = []
    for linea in texto.splitlines():
        lineas.extend(textwrap.wrap(linea, ancho) or [""])
    return lineas


def guardar_pdf(ruta_pdf, titulo="Resultados"):
    _cerrar_texto()
    sys.stdout = _consola

    with PdfPages(ruta_pdf) as pdf:
        _guardar_pagina(pdf, _portada(titulo))
        pagina = None
        y = MARGEN_SUPERIOR

        for tipo, contenido in _elementos:
            if tipo == "subtitulo":
                if pagina is not None:
                    _guardar_pagina(pdf, pagina)
                pagina = plt.figure(figsize=TAMANIO_PAGINA)
                pagina.text(MARGEN_IZQUIERDO, MARGEN_SUPERIOR, contenido,
                            fontsize=16, weight="bold", va="top")
                y = MARGEN_SUPERIOR - 0.05

            elif tipo == "texto":
                for linea in _partir_lineas(contenido):
                    if pagina is None or y < MARGEN_INFERIOR:
                        if pagina is not None:
                            _guardar_pagina(pdf, pagina)
                        pagina = plt.figure(figsize=TAMANIO_PAGINA)
                        y = MARGEN_SUPERIOR
                    pagina.text(MARGEN_IZQUIERDO, y, linea, family="monospace",
                                fontsize=8, va="top", parse_math=False)
                    y -= ALTO_LINEA

            else:
                if pagina is not None:
                    _guardar_pagina(pdf, pagina)
                    pagina = None
                _guardar_pagina(pdf, contenido)

        if pagina is not None:
            _guardar_pagina(pdf, pagina)

    print(f"Resultados guardados en {ruta_pdf}")

def imprimir_datos_clinicos(datos):
    fila = datos.iloc[0]
    print(f"Sujeto: {fila['Subject ID']}")
    print(f"Grupo: {fila['Group']}")
    print(f"Sexo: {fila['M/F']}    Mano dominante: {fila['Hand']}")
    print(f"Educación (años): {fila['EDUC']}    SES: {fila['SES']}")
    print(f"Cantidad de visitas: {len(datos)}")
    print()

    columnas = ["MRI ID", "Visit", "MR Delay", "Age", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    print(datos[columnas].to_string(index=False))


# Guarda la máscara ventricular de una visita como NIfTI, con la ubicación espacial de mpr-1.
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