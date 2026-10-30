# =============================================================================
# INTERPRETACIÓN
# =============================================================================
# A partir de la tabla de volúmenes por visita (Resultados/volumenes.csv) calcula cómo
# cambia cada paciente en el tiempo:
#   - Volumen cerebral: cambio desde MR1 y entre visitas, tasas en ml/año.
#   - Ventrículos: cambio desde MR1 (ml, % y fracción ventricular).
#   - Relación con la clínica: CDR y MMSE junto con los cambios morfológicos.
# No procesa imágenes: solo trabaja sobre la tabla. Los gráficos están en Visualizacion.py.

import numpy as np
import pandas as pd


# Valor de una columna en la primera y la última visita (o NaN si la columna no está)
def _primera_y_ultima(visitas, columna):
    if columna not in visitas.columns:
        return np.nan, np.nan
    return round(float(visitas[columna].iloc[0]), 3), round(float(visitas[columna].iloc[-1]), 3)


# ============================================================
# VOLUMEN CEREBRAL
# ============================================================

def calcular_evolucion(datos):
    """
    Calcula la evolución del volumen cerebral para cada paciente,
    comparando cada visita con la inicial y con la anterior.
    """

    tabla = datos.copy()

    # Ordenar las visitas cronológicamente
    tabla = tabla.sort_values(
        ["paciente", "anios_desde_primera", "visita"]
    ).reset_index(drop=True)

    grupos = tabla.groupby("paciente", sort=False)

    volumen = tabla["volumen_tejido_ml"].astype(float)
    tiempo = tabla["anios_desde_primera"].astype(float)

    # Valores correspondientes a la primera visita
    volumen_inicial = grupos["volumen_tejido_ml"].transform("first")
    tiempo_inicial = grupos["anios_desde_primera"].transform("first")

    # Valores de la visita anterior
    volumen_anterior = grupos["volumen_tejido_ml"].shift(1)
    tiempo_anterior = grupos["anios_desde_primera"].shift(1)

    # --------------------------------------------------------
    # CAMBIOS RESPECTO DE LA PRIMERA VISITA
    # --------------------------------------------------------

    tabla["cambio_desde_inicio_ml"] = (
        volumen - volumen_inicial
    )

    tabla["cambio_desde_inicio_pct"] = (
        100 * (volumen - volumen_inicial) / volumen_inicial
    )

    # --------------------------------------------------------
    # CAMBIOS ENTRE VISITAS CONSECUTIVAS
    # --------------------------------------------------------

    tabla["cambio_desde_anterior_ml"] = (
        volumen - volumen_anterior
    )

    tabla["cambio_desde_anterior_pct"] = (
        100 * (volumen - volumen_anterior) / volumen_anterior
    )

    # --------------------------------------------------------
    # TASA DE CAMBIO EN mL/AÑO
    # --------------------------------------------------------

    intervalo = tiempo - tiempo_anterior

    tabla["tasa_intervalo_ml_anio"] = np.where(
        intervalo > 0,
        (volumen - volumen_anterior) / intervalo,
        np.nan
    )

    # Tasa promedio desde la primera visita
    duracion = tiempo - tiempo_inicial

    tabla["tasa_promedio_desde_inicio_ml_anio"] = np.where(
        duracion > 0,
        (volumen - volumen_inicial) / duracion,
        np.nan
    )

    # --------------------------------------------------------
    # COMPARACIÓN CON nWBV DE OASIS-2
    # --------------------------------------------------------

    nwbv = tabla["nwbv_oasis"].astype(float)

    nwbv_inicial = grupos["nwbv_oasis"].transform("first")

    tabla["cambio_relativo_nwbv_oasis_pct"] = (
        100 * (nwbv - nwbv_inicial) / nwbv_inicial
    )

    return tabla


def resumir_pacientes(evolucion):
    """
    Genera un resumen de la evolución de cada paciente
    entre la primera y la última visita.
    """

    filas = []

    for paciente, grupo in evolucion.groupby("paciente"):

        grupo = grupo.sort_values("anios_desde_primera")

        primera = grupo.iloc[0]
        ultima = grupo.iloc[-1]

        volumen_inicial = primera["volumen_tejido_ml"]
        volumen_final = ultima["volumen_tejido_ml"]

        anios = (
            ultima["anios_desde_primera"]
            - primera["anios_desde_primera"]
        )

        cambio_ml = volumen_final - volumen_inicial

        cambio_pct = (
            100 * cambio_ml / volumen_inicial
        )

        tasa_anual = (
            cambio_ml / anios if anios > 0 else np.nan
        )

        filas.append({
            "paciente": paciente,
            "grupo": primera["grupo"],
            "cantidad_visitas": len(grupo),
            "anios_seguimiento": anios,
            "volumen_inicial_ml": volumen_inicial,
            "volumen_final_ml": volumen_final,
            "cambio_total_ml": cambio_ml,
            "cambio_total_pct": cambio_pct,
            "tasa_promedio_ml_anio": tasa_anual,
            "nwbv_oasis_inicial": primera["nwbv_oasis"],
            "nwbv_oasis_final": ultima["nwbv_oasis"],
            "cdr_inicial": primera["cdr"],
            "cdr_final": ultima["cdr"],
            "mmse_inicial": primera["mmse"],
            "mmse_final": ultima["mmse"]
        })

    return pd.DataFrame(filas)


# ============================================================
# VENTRÍCULOS
# ============================================================

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

    # Fracción ventricular (ventrículos / zona intracraneal, en %): cambio en puntos
    # porcentuales desde MR1. Permite comparar pacientes con cabezas de distinto tamaño.
    if "fraccion_ventricular_pct" in tabla.columns:
        fraccion_inicial = grupos["fraccion_ventricular_pct"].transform(lambda x: x.iloc[0])
        tabla["cambio_fraccion_ventricular_puntos"] = (
            tabla["fraccion_ventricular_pct"] - fraccion_inicial
        )

    return tabla


# ============================================================
# ANÁLISIS MORFOLÓGICO Y CLÍNICO LONGITUDINAL
# ============================================================

def analizar_evolucion_clinica(datos):
    """
    Compara la evolución del volumen cerebral, los candidatos
    ventriculares y los indicadores clínicos CDR y MMSE.

    Devuelve una tabla por visita y un resumen por paciente. Los gráficos
    los arma Visualizacion.graficar_analisis_clinico y el guardado main.py.
    """

    requeridas = {
        "paciente", "grupo", "visita", "anios_desde_primera",
        "volumen_tejido_ml", "volumen_ventricular_ml",
        "nwbv_oasis", "cdr", "mmse"
    }

    faltantes = requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {sorted(faltantes)}. "
            "Ejecutar primero la segmentación ventricular."
        )

    tabla = datos.copy().sort_values(
        ["paciente", "anios_desde_primera", "visita"]
    ).reset_index(drop=True)

    columnas_numericas = [
        "anios_desde_primera",
        "volumen_tejido_ml",
        "volumen_ventricular_ml",
        "nwbv_oasis",
        "cdr",
        "mmse"
    ]

    for columna in columnas_numericas:
        tabla[columna] = pd.to_numeric(
            tabla[columna], errors="coerce"
        )

    resultados = []
    resumen = []

    # --------------------------------------------------------
    # COMPARACIÓN LONGITUDINAL POR PACIENTE
    # --------------------------------------------------------

    for paciente, visitas in tabla.groupby("paciente", sort=False):

        visitas = visitas.copy().sort_values(
            "anios_desde_primera"
        )

        primera = visitas.iloc[0]
        ultima = visitas.iloc[-1]

        tiempo = visitas["anios_desde_primera"]

        duracion = float(
            tiempo.iloc[-1] - tiempo.iloc[0]
        )

        # Primera visita como referencia
        cerebro_0 = float(primera["volumen_tejido_ml"])
        ventriculos_0 = float(primera["volumen_ventricular_ml"])
        nwbv_0 = float(primera["nwbv_oasis"])

        # ----------------------------------------------------
        # CAMBIOS DEL VOLUMEN CEREBRAL
        # ----------------------------------------------------

        visitas["cambio_cerebral_ml"] = (
            visitas["volumen_tejido_ml"] - cerebro_0
        )

        visitas["cambio_cerebral_pct"] = (
            100 * visitas["cambio_cerebral_ml"] / cerebro_0
            if np.isfinite(cerebro_0) and cerebro_0 > 0
            else np.nan
        )

        # ----------------------------------------------------
        # CAMBIOS DEL VOLUMEN VENTRICULAR
        # ----------------------------------------------------

        visitas["cambio_ventricular_ml"] = (
            visitas["volumen_ventricular_ml"] - ventriculos_0
            if np.isfinite(ventriculos_0) and ventriculos_0 > 0
            else np.nan
        )

        visitas["cambio_ventricular_pct"] = (
            100 * visitas["cambio_ventricular_ml"] / ventriculos_0
            if np.isfinite(ventriculos_0) and ventriculos_0 > 0
            else np.nan
        )

        # ----------------------------------------------------
        # COMPARACIÓN CON nWBV DE OASIS
        # ----------------------------------------------------

        visitas["cambio_nwbv_oasis_pct"] = (
            100 * (visitas["nwbv_oasis"] - nwbv_0) / nwbv_0
            if np.isfinite(nwbv_0) and nwbv_0 > 0
            else np.nan
        )

        # ----------------------------------------------------
        # EVOLUCIÓN CLÍNICA
        # ----------------------------------------------------

        visitas["cambio_cdr"] = (
            visitas["cdr"] - primera["cdr"]
        )

        visitas["cambio_mmse_puntos"] = (
            visitas["mmse"] - primera["mmse"]
        )

        # ----------------------------------------------------
        # RELACIÓN ENTRE CEREBRO Y VENTRÍCULOS
        # ----------------------------------------------------

        cambio_cerebro = float(
            visitas["cambio_cerebral_ml"].iloc[-1]
        )

        cambio_vent = float(
            visitas["cambio_ventricular_ml"].iloc[-1]
        )

        if not np.isfinite(cambio_vent):
            patron = "No evaluable: volumen ventricular inicial inválido"

        elif cambio_cerebro < 0 and cambio_vent > 0:
            patron = "Menor volumen cerebral y mayor volumen ventricular"

        elif cambio_cerebro >= 0 and cambio_vent <= 0:
            patron = "Sin patrón de atrofia global y expansión ventricular"

        else:
            patron = "Cambios morfológicos no concordantes"

        # ----------------------------------------------------
        # RESUMEN POR PACIENTE
        # ----------------------------------------------------

        resumen.append({
            "paciente": paciente,
            "grupo": primera["grupo"],
            "visitas": len(visitas),
            "anios_seguimiento": round(duracion, 3),
            "cambio_cerebral_ml": round(cambio_cerebro, 2),
            "cambio_cerebral_pct": round(
                float(visitas["cambio_cerebral_pct"].iloc[-1]), 2
            ),
            "cambio_ventricular_ml": (
                round(cambio_vent, 2)
                if np.isfinite(cambio_vent) else np.nan
            ),
            "cambio_ventricular_pct": (
                round(float(visitas["cambio_ventricular_pct"].iloc[-1]), 2)
                if np.isfinite(visitas["cambio_ventricular_pct"].iloc[-1])
                else np.nan
            ),
            "cdr_inicial": primera["cdr"],
            "cdr_final": ultima["cdr"],
            "cambio_cdr": ultima["cdr"] - primera["cdr"],
            "mmse_inicial": primera["mmse"],
            "mmse_final": ultima["mmse"],
            "cambio_mmse_puntos": ultima["mmse"] - primera["mmse"],
            "fraccion_ventricular_inicial_pct": _primera_y_ultima(visitas, "fraccion_ventricular_pct")[0],
            "fraccion_ventricular_final_pct": _primera_y_ultima(visitas, "fraccion_ventricular_pct")[1],
            "patron_morfologico": patron,
            "ventriculos_sin_validar": True
        })

        resultados.append(visitas)

    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    detalle = pd.concat(resultados, ignore_index=True)
    resumen = pd.DataFrame(resumen)

    return detalle, resumen
