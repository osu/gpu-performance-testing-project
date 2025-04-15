import torch
import time
from utils import metrics

def run_stress_benchmark():
    """
    Runs a GPU stress test by performing heavy matrix multiplications.
    This test is designed to push the GPU towards maximum power usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Increase matrix size and iterations to create substantial load.
    matrix_size = 4096  # Large matrix to push GPU computations
    iterations = 100    # More iterations for sustained GPU load
    
    # Create two large random matrices on the GPU.
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    
    start = time.time()
    for i in range(iterations):
        # Perform matrix multiplication.
        C = torch.mm(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.time()
    
    workload_time = end - start
    avg_time = workload_time / iterations if iterations > 0 else 0.0
    
    # Retrieve GPU utilization as seen by nvidia-smi.
    gpu_util = metrics.get_gpu_utilization()
    
    return {
        "workload_time_sec": workload_time,
        "average_time_per_iteration": avg_time,
        "gpu_utilization_percent": gpu_util,
        "details": f"Stress test with matrix size {matrix_size} over {iterations} iterations"
    }
