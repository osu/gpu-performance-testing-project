import torch
import time
from utils import metrics

def run_gaming_benchmark():
    """
    Simulates a gaming benchmark by running a GPU-intensive workload.
    Performs several heavy matrix multiplications and collects GPU utilization metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set workload parameters:
    matrix_size = 2048  # Bigger matrices lead to heavier computations
    iterations = 50     # Number of iterations in the simulated workload
    
    # Generate two random matrices and move them to GPU
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    
    # Perform matrix multiplications in a loop to simulate GPU load
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)
        # (Optional) Synchronize after each operation to get accurate timings
        torch.cuda.synchronize()
    end = time.time()
    
    workload_time = end - start

    # Retrieve simulated GPU utilization after the workload
    gpu_utilization = metrics.get_gpu_utilization()

    return {
        "workload_time_sec": workload_time,
        "gpu_utilization_percent": gpu_utilization,
        "details": "Simulated gaming workload using matrix multiplications"
    }
