import random

def apply_selection_criteria_randomly(data_points, n_select):
    if not data_points:
        return []
    n_select = int(min(n_select, len(data_points)))
    selected_points = random.sample(data_points, n_select)

    return selected_points
