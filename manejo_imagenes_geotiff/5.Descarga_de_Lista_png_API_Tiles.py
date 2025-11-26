import requests
import os
import time
from tqdm import tqdm # Para la barra de progreso

# --- CONFIGURACIÓN ---
# 🔑 ¡IMPORTANTE! Pega tu clave de API aquí
API_KEY = "AIzaSyAqDfiMVArgZ729lzEPAqkdselJtqjbW8E"

# Archivo de entrada (generado en el Paso 1)
TILE_LIST_FILE = "lista_de_teselas.txt"

# Carpeta de salida para las imágenes
OUTPUT_FOLDER = "teselas_descargadas"
# ---------------------

# ▶️ 1. Crear la carpeta de salida
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Archivos se guardarán en: ./{OUTPUT_FOLDER}/")

# ⚙️ 2. Obtener un token de sesión
# Esto es crucial para decirle a la API que queremos "satélite"
session_url = "https://tile.googleapis.com/v1/createSession?key=" + API_KEY
session_payload = {
    "mapType": "satellite",
    "language": "en-US",
    "region": "MX" # Opcional, pero bueno para tu área
}
try:
    resp = requests.post(session_url, json=session_payload)
    resp.raise_for_status() # Lanza un error si la petición falla
    session_token = resp.json()['session']
    print(f"✅ Sesión de satélite obtenida con éxito.")
except Exception as e:
    print(f"🚨 Error fatal al obtener la sesión de Google.")
    print(f"Verifica tu API_KEY o que la API 'Map Tiles' esté habilitada.")
    print(f"Error: {e}")
    exit()


# 📥 3. Leer la lista de teselas y descargar
print(f"Leyendo teselas desde: {TILE_LIST_FILE}")

# Leemos todas las líneas del archivo
with open(TILE_LIST_FILE, 'r') as f:
    tiles = f.readlines()

total_tiles = len(tiles)
print(f"Iniciando descarga de {total_tiles} teselas (esto puede tardar)...")

# Creamos la barra de progreso con TQDM
for line in tqdm(tiles, unit="tesela"):
    try:
        x, y, z = line.strip().split(',')
        
        # Formato del nombre de archivo: Z_X_Y.png
        # (Guardamos Z al inicio, es más fácil para organizar)
        filename = f"{z}_{x}_{y}.png"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        # 💡 Optimización: Si el archivo ya existe, lo saltamos
        if os.path.exists(output_path):
            continue # Salta al siguiente loop
            
        # Construimos la URL de la tesela
        tile_url = f"https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session={session_token}&key={API_KEY}"
        
        # Hacemos la petición
        response = requests.get(tile_url)
        response.raise_for_status() # Error si la descarga falla
        
        # Guardamos la imagen
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        # Pequeña pausa para no saturar la API
        time.sleep(0.01) # 10 milisegundos

    except requests.exceptions.RequestException as e:
        print(f"\nAdvertencia: Falló la descarga de {line.strip()}. Error: {e}")
    except Exception as e:
        print(f"\nAdvertencia: Error procesando {line.strip()}. Error: {e}")

print(f"\n🎉 ¡Descarga completada! {total_tiles} teselas están en ./{OUTPUT_FOLDER}/")