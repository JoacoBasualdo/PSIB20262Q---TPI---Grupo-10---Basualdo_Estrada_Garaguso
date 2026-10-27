# Descarga el dataset OASIS-2 y extrae SOLO los pacientes que usamos, dejando la
# estructura de carpetas que espera main.py:
#
#   Dataset/
#       oasis_longitudinal_demographics.xlsx
#       OAS2_RAW_PART1/OAS2_XXXX_MR1/RAW/mpr-1.nifti.img (+ .hdr)
#       OAS2_RAW_PART2/OAS2_XXXX_MR1/RAW/...
#
# Los .tar.gz pesan ~10 GB y ~8 GB. El script los lee "en streaming": va
# descomprimiendo a medida que baja y guarda solo los archivos de los pacientes
# elegidos, así que nunca ocupa 18 GB en disco.
#
# Si los .tar.gz ya están bajados en la carpeta Dataset (por ejemplo con aria2c),
# los lee desde el disco en vez de bajarlos de nuevo. Un .tar.gz que todavía tiene
# al lado su archivo .aria2 está incompleto y se espera a que termine.
#
# Uso (desde la carpeta del repo):
#   python descargar_dataset.py
#
# Fuente: https://sites.wustl.edu/oasisbrains/home/oasis-2/
# Al usar los datos hay que citar: Marcus et al., 2010, J. Cogn. Neurosci. 22, 2677-2684.

import os
import sys
import tarfile
import time
import urllib.request
import pandas as pd

RUTA_DATASET = "Dataset"
RUTA_DATOS_CLINICOS = os.path.join(RUTA_DATASET, "oasis_longitudinal_demographics.xlsx")

URL_DEMOGRAFIA = "https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
URLS_RAW = [
    "https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz",
    "https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz",
]

PACIENTES = ["OAS2_0048", "OAS2_0070", "OAS2_0087", "OAS2_0127"]

# Los archivos dentro del .tar están ordenados por paciente. Si es True, deja de
# descargar una parte apenas pasa el último paciente pedido (ahorra varios GB).
# Si al final faltan visitas, ponerlo en False y volver a correr.
CORTAR_AL_TERMINAR = True

# Segundos sin recibir datos antes de abandonar (evita que quede colgado para siempre)
TIEMPO_LIMITE = 60


# Descarga un archivo chico (el Excel clínico) si todavía no existe en el disco
def descargar_archivo(url, destino):
    if os.path.exists(destino):
        print(f"Ya existe {destino}, no se descarga.")
        return
    print(f"Descargando {url} ...")
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(pedido, timeout=TIEMPO_LIMITE) as respuesta, open(destino, "wb") as archivo:
        archivo.write(respuesta.read())


# Busca a qué paciente pertenece un archivo del .tar
# ej: "OAS2_RAW_PART1/OAS2_0048_MR1/RAW/mpr-1.nifti.img" -> "OAS2_0048"
def paciente_de(nombre):
    for parte in nombre.split("/"):
        if parte.startswith("OAS2_") and "_MR" in parte:
            return parte.split("_MR")[0]
    return None


# Envoltorio de la descarga: deja pasar los datos al .tar.gz y cada tanto imprime el avance
class _Progreso:
    # Envuelve la descarga para mostrar cuántos MB van bajando, a qué velocidad y cuánto falta
    def __init__(self, respuesta, total, cada_mb=50):
        self.respuesta = respuesta
        self.total = total
        self.leido = 0
        self.ultimo_aviso = 0
        self.cada = cada_mb * 1024 ** 2
        self.inicio = time.time()

    def read(self, n=-1):
        datos = self.respuesta.read(n)
        self.leido += len(datos)
        if self.leido - self.ultimo_aviso >= self.cada:
            self.ultimo_aviso = self.leido
            segundos = time.time() - self.inicio
            velocidad = self.leido / segundos / 1024 ** 2 if segundos > 0 else 0
            texto = f"    {self.leido / 1024 ** 3:.2f} GB  |  {velocidad:.1f} MB/s"
            if self.total and velocidad > 0:
                restante = (self.total - self.leido) / 1024 ** 2 / velocidad / 60
                texto += f"  |  {self.leido / self.total * 100:.0f}% de la parte  |  ~{restante:.0f} min para leerla entera"
            print(texto, flush=True)
        return datos


# Devuelve de dónde leer el .tar.gz: del disco si ya está bajado completo, si no de internet
def abrir_fuente(url, carpeta):
    local = os.path.join(carpeta, os.path.basename(url))
    if os.path.exists(local + ".aria2"):
        sys.exit(f"{local} todavía se está descargando (existe {local}.aria2). "
                 "Esperá a que aria2c termine y volvé a correr.")
    if os.path.exists(local):
        total = os.path.getsize(local)
        print(f"\nLeyendo archivo local {local} ({total / 1024 ** 3:.1f} GB)")
        return open(local, "rb"), total

    print(f"\nLeyendo {url}")
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        respuesta = urllib.request.urlopen(pedido, timeout=TIEMPO_LIMITE)
    except Exception as error:
        sys.exit(f"No se pudo conectar con {url}\n  Error: {error}\n"
                 "  Probá abrir ese link en el navegador para ver si el servidor responde "
                 "o si pide iniciar sesión / solicitar acceso.")
    total = int(respuesta.headers.get("Content-Length") or 0)
    print(f"  Conectado (HTTP {respuesta.status}). Tamaño: {total / 1024 ** 3:.1f} GB")
    return respuesta, total


# Recorre el .tar.gz archivo por archivo y guarda solo los de los pacientes pedidos.
# Si ya pasó el último paciente pedido, deja de leer (los archivos vienen ordenados).
def extraer_pacientes(url, pacientes, destino):
    ultimo = max(pacientes)
    extraidos = 0
    respuesta, total = abrir_fuente(url, destino)

    with respuesta:
        flujo = _Progreso(respuesta, total)

        # "r|gz" = modo streaming: lee el .tar.gz de corrido sin poder volver atrás
        with tarfile.open(fileobj=flujo, mode="r|gz") as tar:
            for miembro in tar:
                paciente = paciente_de(miembro.name)
                if paciente is None:
                    continue

                if paciente in pacientes:
                    ruta_final = os.path.join(destino, miembro.name)
                    if miembro.isfile() and os.path.exists(ruta_final) and os.path.getsize(ruta_final) == miembro.size:
                        continue
                    if miembro.isfile():
                        print(f"  + {miembro.name}")
                        extraidos += 1
                    if hasattr(tarfile, "data_filter"):
                        tar.extract(miembro, destino, filter="data")
                    else:
                        tar.extract(miembro, destino)

                elif CORTAR_AL_TERMINAR and paciente > ultimo:
                    print(f"  Se pasó {ultimo}: se deja de leer esta parte.")
                    break

    print(f"  Archivos nuevos extraídos: {extraidos}")


# Compara las visitas que figuran en el Excel con las carpetas que quedaron en disco
def verificar(pacientes):
    datos = pd.read_excel(RUTA_DATOS_CLINICOS)
    todo_ok = True
    print("\nVerificación:")
    for paciente in pacientes:
        esperadas = datos.loc[datos["Subject ID"] == paciente, "MRI ID"].tolist()
        faltan = []
        for mri_id in esperadas:
            encontrada = any(
                os.path.isdir(os.path.join(RUTA_DATASET, parte, mri_id, "RAW"))
                for parte in os.listdir(RUTA_DATASET)
                if parte.startswith("OAS2_RAW_PART")
            )
            if not encontrada:
                faltan.append(mri_id)
        estado = "OK" if not faltan else f"FALTAN {faltan}"
        print(f"  {paciente}: {len(esperadas) - len(faltan)}/{len(esperadas)} visitas  {estado}")
        todo_ok &= not faltan
    return todo_ok


# Flujo completo: Excel clínico -> revisar qué falta -> extraer de cada parte -> verificar
def main():
    os.makedirs(RUTA_DATASET, exist_ok=True)
    descargar_archivo(URL_DEMOGRAFIA, RUTA_DATOS_CLINICOS)

    datos = pd.read_excel(RUTA_DATOS_CLINICOS)
    desconocidos = [p for p in PACIENTES if p not in set(datos["Subject ID"])]
    if desconocidos:
        sys.exit(f"Estos pacientes no están en el Excel: {desconocidos}")

    if verificar(PACIENTES):
        print("\nYa están todas las visitas. No hace falta descargar nada.")
        return

    for url in URLS_RAW:
        extraer_pacientes(url, PACIENTES, RUTA_DATASET)

    if verificar(PACIENTES):
        print("\nDataset listo.")
    else:
        print("\nFaltan visitas. Probá con CORTAR_AL_TERMINAR = False y volvé a correr.")


if __name__ == "__main__":
    main()
