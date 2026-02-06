import torch

# Verificar versión
print(f"Versión de PyTorch: {torch.__version__}")

# Verificar si detecta la GPU (Si instalaste la versión CUDA)
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
