import pandas as pd

def generate_report(results, filename="benchmark_report.csv"):
    """
    Generates a CSV report from the benchmark results.
    
    :param results: List of dictionaries with benchmark results.
    :param filename: Name of the output CSV file.
    """
    if not results:
        print("No benchmark results to report.")
        return

    # Normalize keys across results
    df = pd.DataFrame(results)

    # Fill missing values with NaN (so columns are consistent)
    df = df.fillna("")

    # Reorder columns for readability
    columns_order = [
        "benchmark",
        "iteration",
        "exec_time_sec",
        "workload_time_sec",
        "throughput",
        "gpu_utilization_percent",
        "details"
    ]

    # Keep only columns that exist
    columns_to_use = [col for col in columns_order if col in df.columns]
    df = df[columns_to_use]

    # Format float columns (round to 5 decimals)
    float_cols = df.select_dtypes(include="number").columns
    df[float_cols] = df[float_cols].applymap(lambda x: round(x, 5))

    # Save as CSV
    df.to_csv(filename, index=False)
    print(f"📄 Report saved as '{filename}'")
