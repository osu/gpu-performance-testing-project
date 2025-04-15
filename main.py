import yaml
import subprocess
import sys
import os
import time
import pandas as pd

# Import benchmark functions and utilities
from my_benchmarks import hpc_benchmark, gaming_benchmark, stress_benchmark
from utils import metrics, report, plotting

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def verify_gpu(expected_model):
    """Uses nvidia-smi to verify the GPU model."""
    try:
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
    # Load configuration and verify GPU.
    config = load_config()
    expected_gpu = config.get("gpu_model", "")
    verify_gpu(expected_gpu)
    
    benchmark_results = []
    
    # Run HPCDL benchmark
    hpc_config = config["benchmarks"].get("hpcdl", {})
    if hpc_config.get("enabled", False):
        print("Running HPCDL benchmark...")
        for i in range(hpc_config.get("iterations", 1)):
            print(f"  HPCDL Iteration {i + 1}")
            result = hpc_benchmark.run_hpc_benchmark()
            result["iteration"] = i + 1
            result["benchmark"] = "HPCDL"
            benchmark_results.append(result)
            time.sleep(2)
            
    # Run Gaming benchmark
    gaming_config = config["benchmarks"].get("gaming", {})
    if gaming_config.get("enabled", False):
        print("Running Gaming benchmark...")
        for i in range(gaming_config.get("iterations", 1)):
            print(f"  Gaming Iteration {i + 1}")
            result = gaming_benchmark.run_gaming_benchmark()
            result["iteration"] = i + 1
            result["benchmark"] = "Gaming"
            benchmark_results.append(result)
            time.sleep(2)
    
    # Run Stress benchmark
    stress_config = config["benchmarks"].get("stress", {})
    if stress_config.get("enabled", False):
        print("Running Stress benchmark...")
        for i in range(stress_config.get("iterations", 1)):
            print(f"  Stress Iteration {i + 1}")
            result = stress_benchmark.run_stress_benchmark()
            result["iteration"] = i + 1
            result["benchmark"] = "Stress"
            benchmark_results.append(result)
            time.sleep(2)
    
    # Generate report (CSV)
    report_file = "benchmark_report.csv"
    report.generate_report(benchmark_results, report_file)
    
    # Generate dynamic and static plots.
    try:
        data = pd.read_csv(report_file)
        # Choose which metric to plot (for instance, 'throughput' or 'workload_time_sec')
        plotting.plot_results_dynamic(data, metric="throughput")
        plotting.save_static_plot(data, metric="throughput")
    except Exception as e:
        print("Error generating dynamic plot:", e)
    
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()
