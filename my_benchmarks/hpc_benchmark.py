import torch
import time

def run_hpc_benchmark():
    """
    Runs a simple deep learning workload on the GPU.
    Creates a dummy linear model and runs a forward pass.
    Returns performance metrics including execution time (in seconds).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = 1024
    output_size = 1024
    batch_size = 8192  # Increased to ensure measurable time
    model = torch.nn.Linear(input_size, output_size).to(device)
    dummy_input = torch.randn(batch_size, input_size, device=device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.time()

    exec_time = end - start
    throughput = batch_size / exec_time if exec_time > 0 else float("inf")

    return {
        "exec_time_sec": exec_time,
        "throughput": throughput,
        "details": f"Forward pass on batch size {batch_size}, input dim {input_size}"
    }
