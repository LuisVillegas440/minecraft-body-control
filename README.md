# Minecraft Body Control

Control experimental de Minecraft usando la webcam, deteccion corporal, gestos de mano y movimiento de cabeza.

## Requisitos

- Windows
- Python 3.14 o compatible
- Webcam
- Minecraft con controles por defecto

## Instalacion

Clona el repositorio:

```powershell
git clone https://github.com/LuisVillegas440/minecraft-body-control.git
cd minecraft-body-control
```

Crea y activa un entorno virtual:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Instala las dependencias:

```powershell
pip install -r requirements.txt
```

Ejecuta el proyecto desde la raiz:

```powershell
python main.py
```

## Controles

- Mano izquierda con indice: avanzar y diagonales.
- Mano izquierda con gesto OK: retroceder.
- Mano izquierda con gesto rock: correr.
- Mano derecha con puno: click izquierdo / atacar.
- Mano derecha con OK o pinch: click derecho / usar.
- Mano derecha con indice: saltar.
- Cabeza: control de POV cuando esta activado.

## Atajos

- `F8`: activar/desactivar control de POV.
- `F9`: recentrar cabeza.
- `F12`: activar/desactivar overlay de debug.
- `Q`: salir.

## Notas

- Ejecuta `python main.py` desde la carpeta raiz del proyecto para que encuentre los archivos `.task`.
- Minecraft debe tener el foco para recibir teclado y mouse.
- Si Minecraft se ejecuta como administrador, puede que Python tambien necesite ejecutarse como administrador para enviar input.
- Si no detecta la webcam, prueba cambiar `cv2.VideoCapture(0, cv2.CAP_DSHOW)` por `cv2.VideoCapture(1, cv2.CAP_DSHOW)` en `main.py`.
- Si cambiaste los controles de Minecraft, ajusta las teclas esperadas en el codigo.

## Dependencias

El archivo `requirements.txt` contiene solo las dependencias directas que usa el codigo:

- `mediapipe`
- `opencv-python`
- `numpy`
- `pynput`

Pip instalara automaticamente las dependencias internas que necesiten esas librerias.
