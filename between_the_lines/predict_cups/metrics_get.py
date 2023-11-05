import numpy as np
import scipy.stats as stats

def bootstrap_confidence_interval(labels, prediction, metric_function, num_samples=1000, confidence_level=0.95):
    """
    Compute a 95% confidence interval for a given metric using the bootstrap method.

    Parameters:
        labels (list or array-like): The true labels or target values.
        prediction (list or array-like): The predicted values or model outputs (Can be probabilities or logits for AUC)
        metric_function (function): A function that takes two input arrays (labels, prediction)
                                    and returns a scalar representing the metric value.
                                    For example, metric_function(labels, prediction) could
                                    calculate accuracy, mean squared error, etc.
        num_samples (int, optional): The number of bootstrap samples to generate.
                                     Default is 1000.
        confidence_level (float, optional): The desired confidence level for the interval.
                                            Default is 0.95 (corresponding to a 95% confidence interval).

    Returns:
        dict: A tuple containing lower_bound, upper_bound, average_value, and standard_error.
    """
    num_data_points = len(labels)
    metric_values = []

    for _ in range(num_samples):
        # Generate a bootstrap sample
        indices = np.random.choice(num_data_points, num_data_points, replace=True)
        bootstrap_labels = np.array(labels)[indices]
        bootstrap_prediction = np.array(prediction)[indices]

        # Calculate the metric for the bootstrap sample
        metric_value = metric_function(bootstrap_labels, bootstrap_prediction)
        metric_values.append(metric_value)

    # Calculate the confidence interval
    alpha = 1.0 - confidence_level
    lower_percentile = alpha / 2.0
    upper_percentile = 1.0 - lower_percentile

    lower_bound = np.percentile(metric_values, 100 * lower_percentile)
    upper_bound = np.percentile(metric_values, 100 * upper_percentile)
    # get the average
    average_value = np.mean(metric_values)
    # get standard error
    standard_error = stats.sem(metric_values)
    metrics_return = {
        f'lower_bound': lower_bound,
        f'upper_bound': upper_bound,
        'average_value': average_value,
        'standard_error': standard_error
    }
    return metrics_return