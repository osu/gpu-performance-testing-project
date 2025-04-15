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
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print("Report saved as", filename)
