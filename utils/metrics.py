import subprocess

def get_gpu_utilization():
    """
    Uses nvidia-smi to fetch the current GPU utilization percentage.
    Returns the utilization as a float.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
        # In case there are multiple GPUs, take the first one
        utilization = float(output.split('\n')[0])
        return utilization
    except Exception as e:
        print("Error fetching GPU utilization:", e)
        return None

def get_gpu_name():
    """
    Returns the GPU name as a string.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        return output.split('\n')[0]
    except Exception as e:
        print("Error fetching GPU name:", e)
        return "Unknown GPU"
