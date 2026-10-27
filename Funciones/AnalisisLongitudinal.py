import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# ANÁLISIS LONGITUDINAL DEL VOLUMEN CEREBRAL
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


# ============================================================
# RESUMEN LONGITUDINAL POR PACIENTE
# ============================================================

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
# VISUALIZACIÓN DE LA EVOLUCIÓN LONGITUDINAL
# ============================================================

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
