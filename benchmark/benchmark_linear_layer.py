import os
import argparse
import time
import statistics
import csv
import re

# ==================================================
# Parámetros globales
# ==================================================

WARMUP_ITERS = 5
MEASURE_ITERS = 50
    
# ==================================================
# Parseo de argumentos (ANTES de importar torch)
# ==================================================

parser = argparse.ArgumentParser(
    description="Benchmark CPU/GPU: Linear layer (inferencia y entrenamiento)"
)
parser.add_argument(
    "--device",
    choices=["cpu", "gpu"],
    default="cpu",
    help="Dispositivo de ejecución"
)
parser.add_argument(
    "--dtype",
    choices=["fp32", "fp16", "bf16"],
    default="fp32",
    help="Tipo de dato"
)
parser.add_argument(
    "--input",
    type=int,
    required=True,
    help="Dimensión de entrada de la capa densa"
)

parser.add_argument(
    "--output",
    type=int,
    required=True,
    help="Dimensión de salida de la capa densa"
)
args = parser.parse_args()

# ==================================================
# Configuración de hilos CPU (solo relevante en CPU)
# ==================================================

CPU_THREADS = int(os.environ.get("CPU_THREADS", "1"))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

import torch

# ==================================================
# Selección de dtype
# ==================================================

if args.dtype == "fp32":
    TORCH_DTYPE = torch.float32
elif args.dtype == "fp16":
    TORCH_DTYPE = torch.float16
elif args.dtype == "bf16":
    TORCH_DTYPE = torch.bfloat16
else:
    raise RuntimeError("dtype no soportado")

# ==================================================
# Validación de combinaciones
# ==================================================

if args.device == "cpu":
    if args.dtype != "fp32":
        raise RuntimeError(
            "En CPU solo se permite fp32 para benchmarking fiable"
        )

if args.device == "gpu":
    if args.dtype == "bf16":
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError(
                "BF16 requiere GPU Ampere o posterior"
            )

# ==================================================
# Configuración TF32 (solo Ampere+)
# ==================================================

if args.device == "gpu":
    if args.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

# ==================================================
# Utilidades
# ==================================================

def get_cpu_model():
    model = "Unknown_CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    model = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass
    return model

def sanitize_cpu_name(cpu_name):
    """
    Convierte por ejemplo:
    'Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz'
    en:
    'Intel_Xeon_Silver_4216'
    """
    # Quitar marcas registradas
    cpu_name = cpu_name.replace("(R)", "")
    cpu_name = cpu_name.replace("(TM)", "")

    # Quitar palabras irrelevantes
    cpu_name = cpu_name.replace("CPU", "")
    
    # Quitar frecuencia (@ 2.10GHz)
    if "@" in cpu_name:
        cpu_name = cpu_name.split("@")[0]

    # Sustituir separadores problemáticos por espacio
    cpu_name = re.sub(r"[\/\\:*?\"<>|]", " ", cpu_name)

    # Limpiar espacios múltiples
    cpu_name = " ".join(cpu_name.split())

    # Sustituir espacios por _
    cpu_name = cpu_name.replace(" ", "_")

    return cpu_name

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_op(op, device):
    times = []

    # Warm-up
    for _ in range(WARMUP_ITERS):
        op()
    sync(device)

    # Medición
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        op()
        sync(device)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return mean_time, std_time


# ==================================================
# Benchmarks
# ==================================================

def benchmark_linear_forward(batch_size, in_features, out_features, device):
    layer = torch.nn.Linear(
        in_features, out_features, bias=True
    ).to(device=device, dtype=TORCH_DTYPE)

    x = torch.randn(
        (batch_size, in_features),
        device=device,
        dtype=TORCH_DTYPE
    )
    
    def op():
        with torch.no_grad():
            return layer(x)

    mean_t, std_t = time_op(op, device)
    flops = 2 * batch_size * in_features * out_features
    gflops = flops / mean_t / 1e9
    return mean_t, std_t, gflops


def benchmark_linear_training(batch_size, in_features, out_features, device):
    layer = torch.nn.Linear(
        in_features, out_features, bias=True
    ).to(device=device, dtype=TORCH_DTYPE)

    x = torch.randn(
        (batch_size, in_features),
        device=device,
        dtype=TORCH_DTYPE
    )

    target = torch.randn(
        (batch_size, out_features),
        device=device,
        dtype=TORCH_DTYPE
    )

    criterion = torch.nn.MSELoss()

    def op():
        layer.zero_grad(set_to_none=True)
        y = layer(x)
        loss = criterion(y, target)
        loss.backward()

    mean_t, std_t = time_op(op, device)
    # Aproximación FLOPs:
        # Forward + backward ≈ 3 × GEMM
    flops = 6 * batch_size * in_features * out_features
    gflops = flops / mean_t / 1e9
    return mean_t, std_t, gflops


# ==================================================
# Programa principal
# ==================================================

def main():
    batch_sizes = [1, 64, 256, 1024, 4096]
    results = []

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible en este sistema")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name_sanitized = gpu_name.replace(" ", "_")
        label = f"gpu_{gpu_name_sanitized}_{args.dtype}_{args.input}x{args.output}"
        label_pretty = f"GPU {gpu_name_sanitized} (CUDA)"
    else:
        device = torch.device("cpu")
        cpu_name = get_cpu_model()
        cpu_name_sanitized = sanitize_cpu_name (cpu_name)
        label = f"cpu_{cpu_name_sanitized}_{CPU_THREADS}T_{args.dtype}_{args.input}x{args.output}"
        label_pretty = (
            f"CPU {cpu_name_sanitized} ({CPU_THREADS} thread/s)"
        )
        
    output_csv = f"benchmark_linear_{label}.csv"
 
    print("==============================================")
    print(" Benchmark Linear Layer (CPU/GPU)")
    print("==============================================")
    print(f"Dispositivo: {label_pretty}")
    print(f"Tipo de dato: {args.dtype}")
    print(f"Entrada: {args.input} | Salida: {args.output}")
    print(f"Warm-up: {WARMUP_ITERS} iteraciones")
    print(f"Mediciones: {MEASURE_ITERS} iteraciones")
    print(f"Salida CSV: {output_csv}")
    
    for B in batch_sizes:
        print(f"\n--- Batch size = {B} ---")

        t_mean, t_std, gflops = benchmark_linear_forward(B, args.input, args.output, device)
        print(
            f"INFERENCE | "
            f"Tiempo medio = {t_mean:.6f} s "
            f"(± {t_std:.6f}) | "
            f"{gflops:.2f} GFLOP/s"
        )
        results.append(["inference", B, args.input, args.output, label, t_mean, t_std, gflops])

        t_mean, t_std, gflops = benchmark_linear_training(B, args.input, args.output, device)
        print(
            f"TRAINING | "
            f"Tiempo medio = {t_mean:.6f} s "
            f"(± {t_std:.6f}) | "
            f"{gflops:.2f} GFLOP/s"
        )
        results.append(["training", B, args.input, args.output, label, t_mean, t_std, gflops])

    # -------------------------
    # Guardar CSV
    # -------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "batch_size",
            "in",
            "out",
            "config",
            "time_mean_s",
            "time_std_s",
            "gflops"
        ])
        writer.writerows(results)

    print(f"\nResultados guardados en: {output_csv}")


if __name__ == "__main__":
    main()
