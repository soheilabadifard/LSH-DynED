import numpy as np
from data_selection import apply_selection_criteria_randomly
from data_related_functions import convert_to_array
import faiss

class SlidingWindowLSH:
    def __init__(self, window_size, num_of_class, class_list, num_features, vector_size, size_to_train, size_to_test):
        self.window_size = window_size  # number of samples in each class
        self.num_of_class = num_of_class  # number of classes
        self.class_list = class_list  # list of classes
        self.num_features = num_features
        self.window = {i: [] for i in class_list}  # dictionary of lists for each class
        self.window_stats = {i: 0 for i in class_list}  # dictionary of number of samples in each class window
        self.overall_count = {i: {'count': 0, 'ratio': 0.0} for i in class_list}  # dictionary of number of samples in
        self.newly_added = {i: 0 for i in class_list}  # dictionary of number of samples added in each class window
        self.ready_for_train = False  # flag to check if the window is ready for training
        self.vector_size = vector_size
        self.size_to_train = size_to_train
        self.size_to_test = size_to_test

    def add_to_window(self, dx, dy):
        if self.window_stats[dy] > self.window_size:
            self.window[dy].pop(0)
            self.window_stats[dy] -= 1
        self.window[dy].append(dx)
        self.window_stats[dy] += 1
        self.newly_added[dy] += 1
        self.overall_count[dy]['count'] += 1
        self.update_ratio()
        self.check_flag()

    def update_ratio(self):
        total = sum([i['count'] for i in self.overall_count.values()])
        for key in self.overall_count.keys():
            self.overall_count[key]['ratio'] = self.overall_count[key]['count'] / total

    def check_flag(self):
        if sum([i for i in self.newly_added.values()]) >= 100:
            self.ready_for_train = True

    def get_data_for_pred(self):
        data = []
        labels = []
        each_class = self.size_to_test // self.num_of_class
        for cls in self.class_list:
            data.extend(self.window[cls][-each_class:])
            labels.extend([cls] * len(self.window[cls][-each_class:]))
        return data, labels

    def fetch_data_points(self, class_key, data_point_ids):
        lst = []
        for i in data_point_ids:
            lst.append(self.window[class_key][i])
        return lst

    def get_selected_data_for_train_faiss(self, call='train'):

        if call == 'train':
            self.ready_for_train = False
            self.newly_added = {i: 0 for i in self.class_list}

        n_per_class = self.size_to_train

        final_data = []
        final_labels = []

        ww_avg_ratio = sum([i['ratio'] * i['count'] for i in self.overall_count.values() if i['count'] > 0]) / sum(
            [i['count'] for i in self.overall_count.values() if i['count'] > 0])

        classes_with_high_ratio = [key for key, value in self.overall_count.items() if value['ratio'] > ww_avg_ratio]

        for key in self.overall_count.keys():
            if key not in classes_with_high_ratio and len(self.window[key]) > 0:
                final_data.extend(convert_to_array(self.window[key])[-n_per_class:])
                final_labels.extend([key] * len(convert_to_array(self.window[key])[-n_per_class:]))

        lsh_each_class = {i: faiss.IndexLSH(self.num_features, self.vector_size) for i in
                          classes_with_high_ratio}
        table = {i: {} for i in classes_with_high_ratio}
        for key in classes_with_high_ratio:
            lsh_each_class[key].add(convert_to_array(self.window[key]))

            table[key] = faiss.vector_to_array(lsh_each_class[key].codes)

        for key, item in table.items():
            unique, counts = np.unique(item, return_counts=True)
            select_from_bucket = np.maximum(1, np.round((counts / np.sum(counts)) * n_per_class))

            for i in range(len(unique)):
                if counts[i] == 1:
                    final_data.extend(
                        convert_to_array(self.fetch_data_points(key, np.where(item == unique[i])[0])).tolist())
                    final_labels.extend([key] * counts[i])
                else:
                    metadata = self.fetch_data_points(key, np.where(item == unique[i])[0])
                    selected_points = apply_selection_criteria_randomly(convert_to_array(metadata).tolist(),
                                                                        select_from_bucket[i])
                    final_data.extend(selected_points)
                    final_labels.extend([key] * len(selected_points))

        aux = list(zip(final_data, final_labels))
        np.random.shuffle(aux)
        final_data, final_labels = zip(*aux)
        return list(final_data), list(final_labels)
