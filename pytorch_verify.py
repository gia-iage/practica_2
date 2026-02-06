import torch

# Verificar versión
print(f"Versión de PyTorch: {torch.__version__}")

cuda_disponible = torch.cuda.is_available()
print(f"¿CUDA disponible?: {cuda_disponible}")

device = torch.device("cuda" if cuda_disponible else "cpu")

print(f"---------------------------------------------")
if device.type == 'cpu':
    print(f"✅ Dispositivo activo: CPU (Procesador)")
    print(f"   Hilos utilizados por PyTorch: {torch.get_num_threads()}")
else:
    print(f"✅ Dispositivo activo: GPU ({torch.cuda.get_device_name(0)})")
print(f"---------------------------------------------")

x = torch.rand(5, 3).to(device)
print("\nPrueba de cálculo (Tensor 5x3 generado en CPU):")
print(x)

