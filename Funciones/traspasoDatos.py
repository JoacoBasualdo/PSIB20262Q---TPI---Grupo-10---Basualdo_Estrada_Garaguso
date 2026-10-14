import tarfile
import os

# 1. Rutas exactas
ruta_tar = r"C:\Users\Joaquin Basualdo\Downloads\OAS2_RAW_PART2.tar.gz" # <-- Asegurate que termine en .tar.gz
ruta_destino = r"C:\Users\Joaquin Basualdo\Desktop\ITBA\4to 1er cuatri\PSIB\Trabajos finales\Final TPI\Dataset"

# 2. Los pacientes que necesitamos
pacientes_clave = ["OAS2_0048", "OAS2_0070", "OAS2_0127"]

os.makedirs(ruta_destino, exist_ok=True)

print("Abriendo el archivo de 24GB (puede tardar unos segundos)...")
try:
    with tarfile.open(ruta_tar, "r:gz") as tar:
        miembros = tar.getmembers()
        elementos_a_extraer = []
        
        # Filtramos solo lo que nos importa
        for miembro in miembros:
            if any(paciente in miembro.name for paciente in pacientes_clave):
                elementos_a_extraer.append(miembro)
        
        print(f"Se encontraron {len(elementos_a_extraer)} archivos para estos 3 pacientes. Extrayendo...")
        
        # Extraemos solo esa selección
        tar.extractall(path=ruta_destino, members=elementos_a_extraer)
        print("¡Extracción selectiva completada con éxito!")

except Exception as e:
    print(f"Ocurrió un error: {e}")