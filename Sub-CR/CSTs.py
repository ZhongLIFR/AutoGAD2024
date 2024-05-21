import numpy as np

def compute_custom_statistic_base(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:top_5_percent_index*2]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = mean_A + (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic

def compute_custom_statistic_equal(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:top_5_percent_index*2]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic


def compute_custom_statistic(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic