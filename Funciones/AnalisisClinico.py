
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ANÁLISIS MORFOLÓGICO Y CLÍNICO LONGITUDINAL
# ============================================================

def analizar_evolucion_clinica(datos, carpeta_resultados):
    """
    Compara la evolución del volumen cerebral, los candidatos
    ventriculares y los indicadores clínicos CDR y MMSE.

    Genera una tabla por visita, un resumen por paciente y
    gráficos longitudinales dentro de Resultados.
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

    os.makedirs(carpeta_resultados, exist_ok=True)

    carpeta_graficos = os.path.join(
        carpeta_resultados, "Graficos"
    )
    os.makedirs(carpeta_graficos, exist_ok=True)

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
    rutas_graficos = []

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
            "patron_morfologico": patron,
            "ventriculos_sin_validar": True
        })

        resultados.append(visitas)

        # ----------------------------------------------------
        # GRÁFICOS DEL ANÁLISIS INTEGRADO
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    detalle = pd.concat(resultados, ignore_index=True)
    resumen = pd.DataFrame(resumen)

    detalle.to_csv(
        os.path.join(
            carpeta_resultados,
            "analisis_clinico_integrado.csv"
        ),
        index=False
    )

    resumen.to_csv(
        os.path.join(
            carpeta_resultados,
            "resumen_clinico_integrado.csv"
        ),
        index=False
    )

    return detalle, resumen, rutas_graficos
