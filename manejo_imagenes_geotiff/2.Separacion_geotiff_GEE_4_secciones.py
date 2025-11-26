import os
import glob
import shutil
import re # Usaremos expresiones regulares para leer los nombres

# --- 1. Configuración de Carpetas ---

# Carpeta que contiene las 1600 imágenes
carpeta_fuente = r'C:\Users\aleja\Desktop\google_earths\manejo_imagenes_geotiff\GEE_Exports-20251027T201846Z-1-001'

# Carpeta principal donde se crearán los 4 cuadrantes
carpeta_destino_base = 'imagenes_divididas2'

# Nombres de las 4 subcarpetas
# (Puedes cambiar estos nombres si lo deseas)
carpeta_q1 = os.path.join(carpeta_destino_base, 'cuadrante_1_sup_izq') # Fila 0-19, Col 0-19
carpeta_q2 = os.path.join(carpeta_destino_base, 'cuadrante_2_sup_der') # Fila 0-19, Col 20-39
carpeta_q3 = os.path.join(carpeta_destino_base, 'cuadrante_3_inf_izq') # Fila 20-39, Col 0-19
carpeta_q4 = os.path.join(carpeta_destino_base, 'cuadrante_4_inf_der') # Fila 20-39, Col 20-39

# Lista de todas las carpetas que necesitamos crear
carpetas_a_crear = [carpeta_destino_base, carpeta_q1, carpeta_q2, carpeta_q3, carpeta_q4]

# --- 2. Crear las carpetas de destino ---

print("Creando estructura de carpetas de destino...")
for carpeta in carpetas_a_crear:
    # exist_ok=True evita que el script falle si la carpeta ya existe
    os.makedirs(carpeta, exist_ok=True)

# --- 3. Encontrar todas las imágenes ---

print(f"Buscando archivos .tif en '{carpeta_fuente}'...")
search_criteria = os.path.join(carpeta_fuente, "*.tif")
lista_archivos = glob.glob(search_criteria)

if not lista_archivos:
    print(f"Error: No se encontraron archivos .tif en '{carpeta_fuente}'.")
    exit()

print(f"Se encontraron {len(lista_archivos)} archivos. Empezando a clasificar y mover...")

# --- 4. Definir el patrón del nombre de archivo ---
# Esto busca archivos que tengan el formato "NUMERO_NUMERO.tif"
# Por ejemplo: "0_0.tif", "15_39.tif", "25_10.tif"
# re.compile() crea un objeto de expresión regular para eficiencia
patron_nombre = re.compile(r'^(\d+)_(\d+)\.tif$')

# Contadores para el resumen
contador_q1 = 0
contador_q2 = 0
contador_q3 = 0
contador_q4 = 0
contador_omitidos = 0

# --- 5. Iterar, clasificar y mover ---

for ruta_archivo_completa in lista_archivos:
    # Obtenemos solo el nombre del archivo (ej: "15_35.tif")
    nombre_archivo = os.path.basename(ruta_archivo_completa)
    
    # Comparamos el nombre con nuestro patrón
    coincidencia = patron_nombre.match(nombre_archivo)
    
    if not coincidencia:
        # Si el nombre no coincide (ej: "mapa_general.tif"), lo omitimos
        print(f"  [AVISO] Omitiendo '{nombre_archivo}': no sigue el formato FILA_COLUMNA.tif")
        contador_omitidos += 1
        continue

    try:
        # Extraemos los números como texto (grupos 1 y 2 del patrón)
        # y los convertimos a enteros
        fila = int(coincidencia.group(1))
        columna = int(coincidencia.group(2))

        # --- Lógica de clasificación (el núcleo del script) ---
        
        destino_final = None
        
        # Cuadrante 1: Superior Izquierdo (Filas 0-19, Cols 0-19)
        if (0 <= fila <= 19) and (0 <= columna <= 19):
            destino_final = os.path.join(carpeta_q1, nombre_archivo)
            contador_q1 += 1
            
        # Cuadrante 2: Superior Derecho (Filas 0-19, Cols 20-39)
        elif (0 <= fila <= 19) and (20 <= columna <= 39):
            destino_final = os.path.join(carpeta_q2, nombre_archivo)
            contador_q2 += 1
            
        # Cuadrante 3: Inferior Izquierdo (Filas 20-39, Cols 0-19)
        elif (20 <= fila <= 39) and (0 <= columna <= 19):
            destino_final = os.path.join(carpeta_q3, nombre_archivo)
            contador_q3 += 1
            
        # Cuadrante 4: Inferior Derecho (Filas 20-39, Cols 20-39)
        elif (20 <= fila <= 39) and (20 <= columna <= 39):
            destino_final = os.path.join(carpeta_q4, nombre_archivo)
            contador_q4 += 1
            
        else:
            # Si un archivo se llama "50_50.tif", caería aquí
            print(f"  [AVISO] Omitiendo '{nombre_archivo}': fila/columna fuera del rango 0-39.")
            contador_omitidos += 1
            continue

        # --- Mover el archivo ---
        # ADVERTENCIA: shutil.move BORRA el archivo de la carpeta original
        shutil.move(ruta_archivo_completa, destino_final)

    except Exception as e:
        print(f"  [ERROR] No se pudo mover '{nombre_archivo}': {e}")
        contador_omitidos += 1

# --- 6. Reporte Final ---
print("\n--- Proceso de separación completado ---")
print(f"Archivos en Cuadrante 1 (Sup-Izq): {contador_q1}")
print(f"Archivos en Cuadrante 2 (Sup-Der): {contador_q2}")
print(f"Archivos en Cuadrante 3 (Inf-Izq): {contador_q3}")
print(f"Archivos en Cuadrante 4 (Inf-Der): {contador_q4}")
print("------------------------------------------")
total_movidos = contador_q1 + contador_q2 + contador_q3 + contador_q4
print(f"Total de archivos movidos: {total_movidos}")
print(f"Total de archivos omitidos (error o formato incorrecto): {contador_omitidos}")
print(f"\n¡Listo! Revisa la carpeta '{carpeta_destino_base}'.")