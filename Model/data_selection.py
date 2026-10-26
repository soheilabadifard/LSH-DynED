# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.cluster import KMeans
import random
import numpy as np
np.random.seed(101)
random.seed(101)


def apply_selection_criteria1(data_points, n_select, n_clusters=None):
    """
    Selects diverse and relevant samples from a list of data points.

    Args:
        data_points (list): A list of data points (numpy arrays).
        n_select (int): Number of samples to select.
        n_clusters (int): Number of clusters to use for diversity. Defaults to n_select if not provided.

    Returns:
        list: Selected data points.
    """
    if n_clusters is None or n_clusters > len(data_points):
        n_clusters = min(n_select, len(data_points))

    data_matrix = np.array(data_points)
    kmeans = KMeans(n_clusters=n_clusters, random_state=101).fit(data_matrix)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_points_indices = np.where(labels == cluster_id)[0]
        cluster_points = data_matrix[cluster_points_indices]
        distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
        sorted_indices = np.argsort(distances)
        # Ensure at least one point per cluster is selected
        selected_indices.append(cluster_points_indices[sorted_indices[0]])

    # If n_select > n_clusters, distribute the remaining selections across clusters
    additional_selections_needed = n_select - len(selected_indices)
    if 0 < additional_selections_needed < len(data_points) - n_select:
        cluster_distribution = np.bincount(labels, minlength=n_clusters)
        cluster_weights = cluster_distribution / cluster_distribution.sum()
        additional_per_cluster = np.round(cluster_weights * additional_selections_needed).astype(int)

        for cluster_id, additional_count in enumerate(additional_per_cluster):
            if additional_count > 0:
                cluster_points_indices = np.where(labels == cluster_id)[0]
                cluster_points = data_matrix[cluster_points_indices]
                distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
                sorted_indices = np.argsort(distances)
                # Start from 1 since the closest (0th) is already selected
                for i in range(1, min(len(sorted_indices), 1 + additional_count)):
                    if cluster_points_indices[sorted_indices[i]] not in selected_indices:
                        selected_indices.append(cluster_points_indices[sorted_indices[i]])

    # Adjust if we have selected too many due to rounding
    return [data_points[idx] for idx in selected_indices]


def apply_selection_criteria2(data_points, n_select, n_clusters=None):
    """
    Selects diverse and relevant samples from a list of data points.

    Args:
        data_points (list): A list of data points (numpy arrays).
        n_select (int): Number of samples to select.
        n_clusters (int): Optional. Number of clusters to use for diversity.

    Returns:
        list: Selected data points.
    """
    if not data_points:
        return []

    if n_clusters is None or n_clusters > len(data_points):
        n_clusters = min(n_select, len(data_points))

    data_matrix = np.array(data_points)

    # When the requested number of selections equals or exceeds the number of available data points,
    # return all data points without clustering.
    if n_select >= len(data_points):
        return data_points

    kmeans = KMeans(n_clusters=n_clusters, random_state=101).fit(data_matrix)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []

    # Initially, select the closest point to each cluster center.
    for cluster_id in range(n_clusters):
        cluster_points_indices = np.where(labels == cluster_id)[0]
        if len(cluster_points_indices) <= n_select // n_clusters:
            # If the cluster is small enough, select all points from it.
            selected_indices.extend(cluster_points_indices.tolist())
        else:
            # Otherwise, select the closest points to the cluster center.
            cluster_points = data_matrix[cluster_points_indices]
            distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
            sorted_indices = np.argsort(distances)
            num_to_select = max(1, n_select // n_clusters)
            selected_indices.extend(cluster_points_indices[sorted_indices[:num_to_select]].tolist())

    # Adjust the selection to match the exact requested number, n_select,
    # by adding or removing selections from the largest clusters.
    current_selection_count = len(selected_indices)
    if current_selection_count < n_select:
        # If we have room for more, add additional selections from larger clusters.
        additional_needed = n_select - current_selection_count
        # This could be further refined to select additional points from the largest clusters.
    elif current_selection_count > n_select:
        # If we've selected too many, trim the selections starting from the least representative points.
        # This step requires additional logic to identify which points to trim.
        pass

    # Ensure the final list doesn't exceed n_select due to rounding errors or other factors.
    return [data_points[idx] for idx in selected_indices[:n_select]]


def apply_selection_criteria(data_points, n_select, n_clusters=None):
    if not data_points:
        return []

    if n_clusters is None or n_clusters > len(data_points):
        n_clusters = min(n_select, len(data_points))

    data_matrix = np.array(data_points)
    if n_select >= len(data_points):
        return data_points

    kmeans = KMeans(n_clusters=n_clusters, random_state=101).fit(data_matrix)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Determine how many to initially select from each cluster
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    base_selection_per_cluster = [min(size, max(1, n_select // n_clusters)) for size in cluster_sizes]

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_points_indices = np.where(labels == cluster_id)[0]
        cluster_points = data_matrix[cluster_points_indices]
        distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
        sorted_indices = np.argsort(distances)
        selected_indices.extend(cluster_points_indices[sorted_indices[:int(base_selection_per_cluster[cluster_id])]].tolist())

    # Adjust if necessary to ensure exactly n_select items are chosen
    adjustment_needed = n_select - len(selected_indices)
    if adjustment_needed > 0:
        # Additional selection logic here
        pass  # Placeholder for additional selection logic

    return [data_points[idx] for idx in selected_indices[:int(n_select)]]


def apply_selection_criteria3(data_points, n_select, n_clusters=None):
    if not data_points:
        return []

    if n_clusters is None or n_clusters > len(data_points):
        n_clusters = min(n_select, len(data_points))

    data_matrix = np.array(data_points)
    if n_select >= len(data_points):
        return data_points

    kmeans = KMeans(n_clusters=n_clusters, random_state=101).fit(data_matrix)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Determine initial selection from each cluster
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    base_selection_per_cluster = [min(size, max(1, n_select // n_clusters)) for size in cluster_sizes]

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_points_indices = np.where(labels == cluster_id)[0]
        cluster_points = data_matrix[cluster_points_indices]

        # Calculate distances from cluster center
        distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)

        # Index of the closest point
        closest_index = np.argmin(distances)
        selected_indices.append(cluster_points_indices[closest_index])

        # Select farthest points for the remaining quota
        if base_selection_per_cluster[cluster_id] > 1:
            sorted_indices = np.argsort(-distances)  # Sort distances in descending order
            # Exclude the closest already selected, take the next farthest points
            selected_indices.extend(
                cluster_points_indices[sorted_indices[1:base_selection_per_cluster[cluster_id]]].tolist())

    # Adjust if necessary to ensure exactly n_select items are chosen
    adjustment_needed = n_select - len(selected_indices)
    if adjustment_needed > 0:
        # Additional selection logic here to handle cases where the division isn't perfect
        pass  # Placeholder for additional selection logic

    return [data_points[idx] for idx in selected_indices[:n_select]]


def apply_selection_criteria_randomly(data_points, n_select):
    if not data_points:
        return []

    # Ensure n_select does not exceed the number of available data points
    n_select = int(min(n_select, len(data_points)))

    # Select n_select items randomly from the data_points list
    selected_points = random.sample(data_points, n_select)

    return selected_points
