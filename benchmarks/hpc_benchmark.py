import torch
import time

def run_hpc_benchmark():
    """
    Runs a simple deep learning workload on the GPU.
    Creates a dummy linear model and runs a forward pass.
    Returns performance metrics including execution time (in seconds).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy model: a single fully connected layer
    input_size = 1024
    output_size = 1024
    model = torch.nn.Linear(input_size, output_size).to(device)
    
    # Dummy input data
    dummy_input = torch.randn(1024, input_size, device=device)
    
    # Warm-up (to stabilize GPU performance)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Benchmark: measure time taken for a forward pass
    start = time.time()
    with torch.no_grad():
        output = model(dummy_input)
        # Ensure all GPU work is done
        torch.cuda.synchronize()
    end = time.time()
    
    exec_time = end - start

    # (Optionally, calculate throughput or other metrics)
    throughput = dummy_input.shape[0] / exec_time

    # Return the results
    return {
        "exec_time_sec": exec_time,
        "throughput": throughput,
        "details": "Forward pass of dummy linear model"
    }
