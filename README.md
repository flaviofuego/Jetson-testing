# Jetson Orin NX Super — Setup Guide

**Objetivo:** Vision por computadora (CV) + ROS2  
**Flujo:** SSH desde Windows + trabajo directo en la Jetson  
**JetPack target:** 6.x (Ubuntu 22.04, ROS2 Humble/Jazzy)

---

## FASE 0 — Requisitos previos (en tu PC Windows)

> **IMPORTANTE:** NVIDIA SDK Manager NO es compatible con Windows nativo.
> Solo corre en Ubuntu 18.04 / 20.04 / 22.04. Necesitas una de estas opciones:

### Opcion A — VM Ubuntu 22.04 (recomendada, mas estable)
- [x] Instalar **VirtualBox** o **VMware Workstation Player** (gratuito)
- [x] Crear VM con Ubuntu 22.04 (minimo 50 GB disco, 8 GB RAM)
- [x] Configurar **USB passthrough** en la VM para que detecte la Jetson
  - VirtualBox: Settings → USB → agregar filtro para dispositivo NVIDIA APX
  - VMware: VM → Removable Devices → conectar el USB de la Jetson a la VM
- [x] Instalar SDK Manager dentro de la VM Ubuntu

### Opcion B — WSL2 + usbipd-win (avanzado)
- [x] Instalar WSL2 con Ubuntu 22.04:
  ```powershell
  wsl --install -d Ubuntu-22.04
  ```
- [x] Instalar `usbipd-win` para pasar el USB de la Jetson a WSL2:
  ```powershell
  winget install usbipd
  # Luego dentro de WSL2:
  usbipd list                        # buscar la Jetson (NVIDIA APX)
  usbipd bind --busid <ID>
  usbipd attach --wsl --busid <ID>
  ```
- [x] Instalar SDK Manager dentro de WSL2 Ubuntu

### Herramientas comunes (en Windows)
- [x] **NVIDIA SDK Manager** (instalar en la VM o WSL2)
  - Descargar desde: https://developer.nvidia.com/sdk-manager
  - Requiere cuenta NVIDIA Developer (gratuita)
- [x] **Cable USB-C** para conexion en modo recovery
- [x] **SSH client:** VS Code + extension Remote-SSH
- [x] Windows Terminal

---

## FASE 1 — Flashear JetPack 6.x con SDK Manager

### 1.1 Poner la Jetson en modo Recovery
1. Desconectar alimentación
2. Mantener presionado el botón **RECOVERY** (FC REC)
3. Conectar alimentación mientras sostienes RECOVERY
4. Soltar el botón después de 2 segundos
5. Conectar USB-C entre Jetson (puerto de flash) y tu PC

Verificar en WSL2:
```bash
lsusb | grep -i nvidia
# Debe mostrar: ID 0955:7323 NVIDIA Corp. APX
```

### 1.2 Instalar SDK Manager en WSL2
```bash
# Dentro de WSL2 Ubuntu 22.04
sudo dpkg -i sdkmanager_*_amd64.deb
sdkmanager --cli install \
  --product Jetson \
  --version 6.2 \
  --targetos Linux \
  --host
```

O usar la GUI:
```bash
sdkmanager
# Seleccionar: Jetson Orin NX → JetPack 6.x → incluir DeepStream (opcional)
```

### 1.3 Configuracion inicial post-flash
Al arrancar por primera vez con monitor+teclado conectado:
- Crear usuario y contraseña
- Configurar red (WiFi o Ethernet)
- Habilitar SSH:
  ```bash
  sudo systemctl enable ssh
  sudo systemctl start ssh
  ip addr show  # anotar la IP
  ```

---

## FASE 2 — Configurar SSH desde Windows

### 2.1 Generar clave SSH en Windows
```powershell
ssh-keygen -t ed25519 -C "jetson-dev"
# Guardar en: C:\Users\flavi\.ssh\id_ed25519_jetson
```

### 2.2 Copiar clave a la Jetson
```powershell
type $env:USERPROFILE\.ssh\id_ed25519_jetson.pub | ssh usuario@<JETSON_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 2.3 Configurar ~/.ssh/config en Windows
```
Host jetson
    HostName <JETSON_IP>
    User <tu-usuario>
    IdentityFile ~/.ssh/id_ed25519_jetson
    ServerAliveInterval 30
```

Conectar:
```powershell
ssh jetson
```

### 2.4 VS Code Remote SSH
- Instalar extension: `ms-vscode-remote.remote-ssh`
- Ctrl+Shift+P → "Remote-SSH: Connect to Host" → `jetson`

---

## FASE 3 — Setup del entorno en la Jetson

### 3.1 Actualizar sistema
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git wget nano htop nvtop python3-pip python3-venv
```

### 3.2 Verificar GPU y CUDA
```bash
nvidia-smi
nvcc --version
# JetPack 6.x trae CUDA 12.x
```

### 3.3 Verificar Python y pip
```bash
python3 --version  # debe ser 3.10+
pip3 install --upgrade pip
```

### 3.4 Instalar PyTorch (wheel oficial NVIDIA para Jetson)
```bash
# JetPack 6.x — PyTorch 2.x para Jetson
pip3 install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl
```
> Verificar la URL mas reciente en: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

### 3.5 Instalar torchvision
```bash
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
cd torchvision && python3 setup.py install --user
```

### 3.6 Verificar PyTorch con GPU
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # Orin
```

---

## FASE 4 — Vision por Computadora

### 4.1 OpenCV con CUDA (viene con JetPack)
```bash
python3 -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
```

Si no tiene CUDA habilitado, instalar desde JetPack:
```bash
sudo apt install -y python3-opencv
# O compilar desde fuente con CUDA support (proceso largo ~2h)
```

### 4.2 YOLOv8 con Ultralytics
```bash
pip3 install ultralytics
# Inferencia de prueba:
yolo predict model=yolov8n.pt source=0  # camara
```

### 4.3 TensorRT (optimización de modelos)
```bash
# Viene preinstalado con JetPack 6.x
python3 -c "import tensorrt as trt; print(trt.__version__)"

# Exportar modelo YOLO a TensorRT:
yolo export model=yolov8n.pt format=engine device=0
```

### 4.4 Camara (CSI o USB)
```bash
# Probar camara USB
ls /dev/video*
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Camara CSI (IMX219) con GStreamer:
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink
```

---

## FASE 5 — ROS2 Humble (sobre Ubuntu 22.04 / JetPack 6.x)

### 5.1 Instalar ROS2 Humble
```bash
# Configurar repositorios
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Instalar
sudo apt update
sudo apt install -y ros-humble-desktop python3-rosdep python3-colcon-common-extensions
```

### 5.2 Configurar entorno ROS2
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Inicializar rosdep
sudo rosdep init
rosdep update
```

### 5.3 Verificar ROS2
```bash
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_python listener
```

### 5.4 Paquetes utiles para CV + ROS2
```bash
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-vision-msgs \
  ros-humble-sensor-msgs \
  ros-humble-camera-ros
```

---

## FASE 6 — Optimizaciones de performance

### 6.1 Modo de maxima potencia
```bash
sudo nvpmodel -m 0   # MAX performance
sudo jetson_clocks   # fijar frecuencias maximas
```

### 6.2 Monitoreo en tiempo real
```bash
sudo apt install -y python3-pip
sudo pip3 install jetson-stats
sudo jtop  # equivalente a htop pero para Jetson
```

### 6.3 Swap (recomendado para compilaciones)
```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 8G /mnt/8GB.swap
sudo chmod 600 /mnt/8GB.swap
sudo mkswap /mnt/8GB.swap
sudo swapon /mnt/8GB.swap
echo '/mnt/8GB.swap none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## FASE 7 — Estructura del workspace de desarrollo

```
~/
├── ros2_ws/           # workspace ROS2
│   └── src/
├── models/            # modelos ONNX, TensorRT engines
├── datasets/          # datasets para entrenamiento/testing
└── scripts/           # scripts utilitarios
```

```bash
mkdir -p ~/ros2_ws/src ~/models ~/datasets ~/scripts
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## Comandos de referencia rapida

| Tarea | Comando |
|-------|---------|
| Ver uso GPU | `nvidia-smi` o `sudo jtop` |
| Modo max power | `sudo nvpmodel -m 0` |
| Estado camara CSI | `nvgstcapture-1.0` |
| Source ROS2 | `source /opt/ros/humble/setup.bash` |
| Build workspace | `cd ~/ros2_ws && colcon build` |
| Ver topics activos | `ros2 topic list` |
| Inferencia YOLO | `yolo predict model=yolov8n.pt source=0` |

---

## Recursos utiles

- [JetPack SDK Download](https://developer.nvidia.com/embedded/jetpack)
- [PyTorch para Jetson (NVIDIA Forums)](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Zoo (modelos precompilados)](https://www.elinux.org/Jetson_Zoo)
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [jetson-stats (jtop)](https://github.com/rbonghi/jetson_stats)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
