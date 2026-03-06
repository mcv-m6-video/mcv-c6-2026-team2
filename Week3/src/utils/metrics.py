import time

import numpy as np


def evaluate(
    method,
    gt: np.ndarray,
    name: str,
    results: dict,
    *params,
    output_postprocess=None,
    mask: np.ndarray | None = None,
    threshold: int = 3,
    num_iters: int = 1,
):
    results["method"].append(name)

    raw_output, mean_elapsed_time, std_elapsed_time, actual_num_iters = runtime_compute(
        method,
        *params,
        num_iters=num_iters,
    )
    print(f"Elapsed time: {mean_elapsed_time:.3f} (+/-{std_elapsed_time:.3f})")
    results["mean_runtime"].append(mean_elapsed_time)
    results["std_runtime"].append(std_elapsed_time)
    results["num_iters"].append(actual_num_iters)

    if output_postprocess is not None:
        output = output_postprocess(raw_output)
    else:
        output = raw_output

    # Compute MSEN, PEPN and efficiency w.r.t. MSEN
    msen = mse_compute(output, gt, mask=mask)
    print(f"MSEN: {msen:.3f}")
    results["msen"].append(msen)

    pepn = pep_compute(output, gt, threshold=threshold, mask=mask)
    print(f"PEPN: {pepn:.3f}")
    results["pepn"].append(pepn)

    eff = custom_efficiency_compute(msen, mean_elapsed_time)
    print(f"Efficiency: {eff:.3f}")
    results["efficiency"].append(eff)

    return output, raw_output, results


def mse_compute(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None):
    error_coords = (prediction - gt) ** 2
    error: np.ndarray = np.sum(error_coords, axis=-1)

    if mask is not None:
        error = error[mask]
    else:
        error = error.flatten()

    mse = np.mean(error)

    return mse


def pep_compute(
    prediction: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
):
    error_coords = (prediction - gt) ** 2
    error: np.ndarray = np.sqrt(np.sum(error_coords, axis=-1))

    if mask is not None:
        error = error[mask]
    else:
        error = error.flatten()

    erroneous_pixels = error > threshold
    num_erroneous_pixels = np.sum(erroneous_pixels)
    pep = num_erroneous_pixels / len(error)

    return pep


def runtime_compute(func, *params, num_iters: int = 1, max_time: float = 120):
    """
    Computes the runtime of the function execution.
    If more than 1 iteration is specified, only the last output will be returned.
    """
    time_samples = []
    for it in range(1, num_iters + 1):
        start_time = time.time()
        output = func(*params)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_samples.append(elapsed_time)
        if elapsed_time > max_time:
            break

    mean_elapsed_time = np.mean(time_samples)
    std_elapsed_time = np.std(time_samples)
    return output, mean_elapsed_time, std_elapsed_time, it


def custom_efficiency_compute(metric: float, runtime: float, lower_better: bool = True):
    """
    This is an efficiency metric just to compare these models.
    It follows no standards.
    FORMULA:
    - (1 / metric) / time | if lower_better == True
    - metric / time | if lower_better == False
    """
    if lower_better:
        eff = (1 / metric) / runtime
    else:
        eff = metric / runtime
    
    return eff