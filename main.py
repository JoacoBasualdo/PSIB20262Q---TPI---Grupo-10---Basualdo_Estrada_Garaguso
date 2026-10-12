# ==========================================
# DEL DATASET: EVALUACIÓN CLÍNICA Y MÉTRICAS VOLUMÉTRICAS
# ==========================================
#
# Group: Etiqueta general del paciente durante el estudio.
#   - 'Nondemented' = Sano
#   - 'Demented' = Con Alzheimer/Demencia
#   - 'Converted' = Empezó sano y desarrolló demencia durante el estudio.
#
# MMSE: Puntuación del Mini-Mental State Examination. 
#   - Test estandarizado de evaluación cognitiva (0 a 30). 
#   - Un puntaje menor a 24 suele indicar deterioro cognitivo.
#
# CDR: Calificación Clínica de Demencia (Clinical Dementia Rating). 
#   - Etiqueta médica del nivel de gravedad:
#       0.0 = Sano / Normal
#       0.5 = Demencia muy leve
#       1.0 = Demencia leve
#       2.0 = Demencia moderada
#
# eTIV: Estimated Total Intracranial Volume (Volumen intracraneal total estimado).
#   - Representa el tamaño total del cráneo (medido en mm^3). 
#   - Permanece constante en el tiempo en adultos y sirve para normalizar 
#     las medidas del cerebro de distintas personas.
#
# nWBV: Normalized Whole Brain Volume (Volumen cerebral total normalizado).
#   - Es la proporción del cráneo que efectivamente está ocupada por tejido cerebral. 
#   - Se calcula dividiendo el volumen del cerebro por el eTIV. 
#   - Variable fundamental: una caída en este valor indica atrofia y pérdida de masa encefálica.
#
# ASF: Atlas Scaling Factor (Factor de escalamiento del atlas).
#   - Coeficiente matemático que indica cuánto hay que comprimir o expandir el cráneo 
#     del paciente para que coincida con un molde de cerebro estándar (atlas de Talairach). 
#   - Es inversamente proporcional al eTIV.



# La idea es seleccionar un paciente sano, uno enfermo y uno converted. 
# El paciente 70 tuvo 5 visitias entre los 80 y 86 años y siempre estuvo sano.
# El paciente 48 tuvo 5 visitas entre los 66 y 69. Ingreso y se mantuvo con un nivel de demencia leve (1.0)
# El paciente 127 tuvo 5 visitas a lo largo de 7 años. Ingreso con sano y a partir de la seguna sesion tenia demencia muy leve
# El paciente 87 entro con 1.0 de demencia y aumento a 2.0 en solo 2 años.

import os
from Funciones.Lectura import cargar_volumen_nifti
from Funciones.Visualizacion import visualizar_planos_ortogonales

def main():
    paciente_id = "OAS2_0127"
    # El dataset OASIS-2 tiene hasta 5 visitas posibles por paciente
    visitas = ["MR1", "MR2", "MR3", "MR4", "MR5"]
    
    for visita in visitas:
        print(f"\n--- Buscando datos de {paciente_id} - Visita {visita} ---")
        
        # Armamos el nombre de la carpeta (ej: OAS2_0127_MR1)
        carpeta_visita = f"{paciente_id}_{visita}"
        
        # Agregamos "OAS2_RAW_PART1" al path para que coincida con tu árbol
        ruta_imagen = os.path.join("Dataset", "OAS2_RAW_PART2", carpeta_visita, "RAW", "mpr-1.nifti.img")
        
        try:
            volumen = cargar_volumen_nifti(ruta_imagen)
            print(f"Carga exitosa. Dimensiones: {volumen.shape}")
            
            # Mostramos el gráfico
            visualizar_planos_ortogonales(volumen, titulo=f"Paciente {paciente_id} - Visita {visita}")
            
        except FileNotFoundError:
            # Es normal que algunos pacientes no tengan todas las visitas
            print(f"Aviso: No se encontró la imagen en la ruta: {ruta_imagen}")

if __name__ == "__main__":
    main()