# =============================================================================
# CONCORDANCIA CON OASIS: ¿MEDIMOS LA MISMA ATROFIA QUE LOS AUTORES DEL DATASET?
# =============================================================================
#
# OASIS publica, para cada visita, el nWBV (volumen cerebral normalizado: fracción del
# cráneo ocupada por cerebro), calculado con su propio método automático (registro contra
# un atlas y segmentación con modelos estadísticos). No es una segmentación manual, pero es
# una referencia independiente de la nuestra.
#
# Nuestro nWBV propio = volumen de tejido medido por nosotros / eTIV de OASIS.
#
# Para cada paciente se compara:
#   - Tasa de atrofia: pendiente de la recta nWBV vs. años, en ambos métodos.
#     Es lo más importante: el objetivo es medir CÓMO CAMBIA el cerebro, y la pendiente
#     usa todas las visitas (promedia el ruido de cada una).
#   - Diferencia de nivel: cuánto más alto o más bajo da nuestro nWBV en promedio.
#     Un corrimiento constante no afecta la tasa de atrofia.
#   - Correlación entre las dos series (solo si el paciente tiene 3 o más visitas).
#
# Uso (desde la carpeta del repo, después de correr main.py):
#   python validar_oasis.py
# Resultado: Resultados/concordancia_oasis.csv y Resultados/Graficos/concordancia_oasis.png

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_RESULTADOS = "Resultados"
RUTA_CSV_VISITAS = os.path.join(RUTA_RESULTADOS, "volumenes.csv")   # lo genera main.py
RUTA_CSV_SALIDA = os.path.join(RUTA_RESULTADOS, "concordancia_oasis.csv")
RUTA_FIGURA = os.path.join(RUTA_RESULTADOS, "Graficos", "concordancia_oasis.png")


# Pendiente de la recta que mejor ajusta los puntos (cambio por año)
def pendiente(anios, valores):
    return np.polyfit(anios, valores, 1)[0]


def comparar_paciente(filas):
    filas = filas.sort_values("anios_desde_primera")
    anios = filas["anios_desde_primera"].values
    propio = filas["nwbv_propio"].values
    oasis = filas["nwbv_oasis"].values

    pendiente_propia = pendiente(anios, propio)
    pendiente_oasis = pendiente(anios, oasis)

    return {
        "paciente": filas["paciente"].iloc[0],
        "grupo": filas["grupo"].iloc[0],
        "visitas": len(filas),
        "anios_seguimiento": round(anios.max(), 2),
        # Tasa de atrofia: cambio de nWBV por año (negativo = pierde cerebro)
        "pendiente_nwbv_propio_anio": round(pendiente_propia, 4),
        "pendiente_nwbv_oasis_anio": round(pendiente_oasis, 4),
        # Cuántas veces la nuestra respecto de la de OASIS (1 = iguales)
        "relacion_pendientes": round(pendiente_propia / pendiente_oasis, 2) if pendiente_oasis != 0 else np.nan,
        # Lo mismo en % del valor inicial por año, para comparar pacientes entre sí
        "tasa_propia_pct_anio": round(100 * pendiente_propia / propio[0], 2),
        "tasa_oasis_pct_anio": round(100 * pendiente_oasis / oasis[0], 2),
        # Corrimiento promedio de nivel (propio - OASIS)
        "diferencia_media_nwbv": round(float(np.mean(propio - oasis)), 4),
        # Con 2 visitas la correlación siempre da ±1, así que no se informa
        "correlacion": round(float(np.corrcoef(propio, oasis)[0, 1]), 3) if len(filas) >= 3 else np.nan,
    }


# Figura: nWBV de cada paciente a lo largo del tiempo, nuestro (rojo) y OASIS (negro)
def graficar(datos, ruta):
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


def main():
    if not os.path.exists(RUTA_CSV_VISITAS):
        print(f"No se encontró {RUTA_CSV_VISITAS}. Correr main.py primero.")
        return

    datos = pd.read_csv(RUTA_CSV_VISITAS)
    tabla = pd.DataFrame([comparar_paciente(filas) for _, filas in datos.groupby("paciente", sort=False)])
    tabla.to_csv(RUTA_CSV_SALIDA, index=False)
    graficar(datos, RUTA_FIGURA)

    print("\n================ CONCORDANCIA CON OASIS (nWBV) ================")
    print(tabla.to_string(index=False))

    # Concordancia global con todas las visitas de todos los pacientes
    correlacion_global = np.corrcoef(datos["nwbv_propio"], datos["nwbv_oasis"])[0, 1]
    print(f"\nCorrelación con todas las visitas juntas: r = {correlacion_global:.3f}")
    print("(Incluye diferencias entre pacientes; para la tasa de atrofia mirar las pendientes por paciente.)")
    print(f"\nTabla guardada en: {RUTA_CSV_SALIDA}")
    print(f"Figura guardada en: {RUTA_FIGURA}")


if __name__ == "__main__":
    main()
