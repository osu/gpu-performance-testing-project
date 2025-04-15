
# GPU Benchmarking Project

This project is designed to automatically run HPCDL and Gaming benchmarks on your NVIDIA RTX 4070, collect performance metrics, and generate a CSV report for analysis.

## Features

- **GPU Verification:** Checks whether the GPU in the system is an NVIDIA RTX 4070.
- **HPCDL Benchmark:** Runs a deep learning style benchmark using PyTorch (a simple network forward pass on dummy data).
- **Gaming Benchmark (Simulated):** Simulates a graphics-intensive workload and collects GPU utilization data.
- **Automated Reporting:** Aggregates performance metrics and outputs a CSV report.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://your-repo-link.git
   cd gpu_benchmark_project
   ```

2. **Install the Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Recommended) Install PyTorch with CUDA support:**
   If you're using an NVIDIA GPU (like the RTX 4070), install PyTorch with CUDA to run benchmarks on GPU:
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

5. **Run the benchmarking script:**
   ```bash
   python main.py
   ```

## Configuration

Adjust the benchmark parameters (e.g., number of iterations, descriptions) in the `config.yaml` file:

```yaml
gpu_model: "NVIDIA GeForce RTX 4070"
benchmarks:
  hpcdl:
    enabled: true
    iterations: 3
    description: "Training a dummy neural network to benchmark GPU performance."
  gaming:
    enabled: true
    iterations: 3
    description: "Simulated gaming benchmark workload."
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyYAML](https://pyyaml.org/)
- [pandas](https://pandas.pydata.org/)

## Project Structure

```
gpu_benchmark_project/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── my_benchmarks/
│   ├── __init__.py
│   ├── hpc_benchmark.py
│   ├── gaming_benchmark.py
│   └── stress_benchmark.py       # New stress benchmark module
└── utils/
    ├── __init__.py
    ├── metrics.py
    ├── report.py
    └── plotting.py               # New dynamic plotting module

```

## How It Works

1. **Configuration and GPU Check:**  
   The script loads benchmark configuration from `config.yaml` and uses `nvidia-smi` to verify that the installed GPU is the NVIDIA RTX 4070. A warning is issued if the GPU does not match.

2. **Running Benchmarks:**  
   - **HPCDL Benchmark:** Uses PyTorch to create a simple neural network that performs a forward pass on dummy data. Execution time and throughput are collected.
   - **Gaming Benchmark:** Simulates a gaming workload by performing heavy matrix multiplications and collects GPU utilization data through `nvidia-smi`.

3. **Reporting:**  
   Benchmark results from each iteration (including type, execution time, and utilization) are aggregated and exported to a CSV file using Pandas.

4. **Extensibility:**  
   This project structure makes it easy to add additional benchmarks or customize the tests for other GPU models.

## Running the Project

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Execute the main script:

   ```bash
   python main.py
   ```

3. After completion, review the `benchmark_report.csv` file that contains all the aggregated metrics.
   ```

Simply copy the above content into your `README.md` file in your GitHub repository, and you’re all set!
