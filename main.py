import yaml
import subprocess
import sys
import os
import time
import pandas as pd

# Import benchmark functions and utilities
from benchmarks import hpc_benchmark, gaming_benchmark
from utils import metrics, report

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def verify_gpu(expected_model):
    """Uses nvidia-smi to check GPU model."""
    try:
        # Query GPU name using nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding="utf-8"
        ).strip()
        print("Detected GPU:", output)
        if expected_model.lower() not in output.lower():
            print(f"Warning: Expected GPU model '{expected_model}' not found. Detected: {output}")
        else:
            print(f"GPU model verified: {output}")
        return output
    except Exception as e:
        print("Error while verifying GPU:", e)
        sys.exit(1)

def main():
    # Load configuration
    config = load_config()

    # Verify that the GPU is the expected one
    expected_gpu = config.get("gpu_model", "")
    verify_gpu(expected_gpu)

    benchmark_results = []

    # Run HPCDL benchmark if enabled
    hpc_config = config["benchmarks"]["hpcdl"]
    if hpc_config.get("enabled", False):
        print("Running HPCDL benchmark...")
        for i in range(hpc_config.get("iterations", 1)):
            print(f"  HPCDL Iteration {i + 1}")
            result = hpc_benchmark.run_hpc_benchmark()
            result["iteration"] = i + 1
            result["benchmark"] = "HPCDL"
            benchmark_results.append(result)
            # Pause between iterations if needed
            time.sleep(2)

    # Run Gaming benchmark if enabled
    gaming_config = config["benchmarks"]["gaming"]
    if gaming_config.get("enabled", False):
        print("Running Gaming benchmark...")
        for i in range(gaming_config.get("iterations", 1)):
            print(f"  Gaming Iteration {i + 1}")
            result = gaming_benchmark.run_gaming_benchmark()
            result["iteration"] = i + 1
            result["benchmark"] = "Gaming"
            benchmark_results.append(result)
            time.sleep(2)

    # Generate report (CSV)
    report_file = "benchmark_report.csv"
    report.generate_report(benchmark_results, report_file)
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()
