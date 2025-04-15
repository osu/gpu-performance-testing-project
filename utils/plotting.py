import matplotlib.pyplot as plt

def plot_results_dynamic(data, metric="throughput"):
    """
    Plots benchmark results dynamically using matplotlib.
    
    :param data: pandas DataFrame containing benchmark results.
    :param metric: The column name to plot (e.g., 'throughput', 'exec_time_sec').
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    
    # Only plot rows where the metric is not null
    filtered_data = data[data[metric].notna()]
    if filtered_data.empty:
        print("No data available to plot dynamic graph.")
        return

    iterations = filtered_data["iteration"].astype(float)
    y_data = filtered_data[metric].astype(float)
    
    ax.clear()
    ax.plot(iterations, y_data, marker="o", linestyle="-")
    ax.set_title(f"Benchmark {metric} Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    
    plt.draw()
    plt.pause(2)  # Pause briefly to display the plot.

def save_static_plot(data, metric="throughput", filename="benchmark_plot.png"):
    """
    Generates and saves a static plot image from benchmark results.
    
    :param data: pandas DataFrame containing benchmark results.
    :param metric: Column name to plot.
    :param filename: Name of the file to save the plot.
    """
    # Filter rows where the metric is not null
    filtered_data = data[data[metric].notna()]
    if filtered_data.empty:
        print("No data available for static plot.")
        return

    iterations = filtered_data["iteration"].astype(float)
    y_data = filtered_data[metric].astype(float)
    
    plt.figure()
    plt.plot(iterations, y_data, marker="o", linestyle="-")
    plt.title(f"Benchmark {metric} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")
