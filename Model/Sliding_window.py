import numpy as np
from LSH import EnhancedLSH
from data_selection import apply_selection_criteria, apply_selection_criteria_randomly
from Model.data_related_functions import convert_to_array
import faiss
from math import sqrt, ceil

np.random.seed(101)


class SlidingWindowLSH:
    def __init__(self, window_size, num_of_class, class_list, num_features, vector_size, size_to_train, size_to_test,
                 sampling_mode='faiss_lsh_majority'):
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
        self.sampling_mode = sampling_mode
        # each class overall

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

    def get_raw_data_for_train(self):
        data = []
        labels = []
        each_class = self.size_to_train // self.num_of_class
        for key in self.window.keys():
            data.extend(self.window[key][-each_class:])
            labels.extend([key] * len(self.window[key][-each_class:]))
        return data, labels

    def get_data_for_pred(self):
        # create a list of n samples, first class 1 then class 2 and so on
        data = []
        labels = []
        each_class = self.size_to_test // self.num_of_class
        for cls in self.class_list:
            data.extend(self.window[cls][-each_class:])
            labels.extend([cls] * len(self.window[cls][-each_class:]))
        return data, labels

    def get_data_for_pred2(self, class_label, n):
        # create a list of n samples, first class 1 then class 2 and so on
        data = []
        labels = []
        each_class = n // self.num_of_class
        if min(self.window_stats.values()) < each_class:
            each_class = min(self.window_stats.values())
        data.extend(self.window[class_label][-each_class:])
        labels.extend([class_label] * len(self.window[class_label][-each_class:]))
        return data, labels

    def fetch_data_points(self, class_key, data_point_ids):
        lst = []
        for i in data_point_ids:
            lst.append(self.window[class_key][i])
        return lst

    def get_selected_data_for_train(self, call='train', n=None):
        # print(f"window stats: {self.overall_count}")
        if call == 'train':
            self.ready_for_train = False
            self.newly_added = {i: 0 for i in self.class_list}

        n_per_class = n // len(self.class_list)  # Ensure integer division

        final_data = []
        final_labels = []

        lsh_each_class = {i: EnhancedLSH(self.num_features, 10) for i in self.class_list}
        for key in self.window.keys():
            if len(self.window[key]) > 0:
                lsh_each_class[key].fit(convert_to_array(self.window[key]))
                for point_id, point in enumerate(convert_to_array(self.window[key])):
                    lsh_each_class[key].add_point(point_id, point)

            total_points = sum(len(ids) for _, ids in lsh_each_class[key].get_hash_table().items())
            table = lsh_each_class[key].get_hash_table()
            for hash_code, ids in table.items():
                bucket_size = len(ids)
                # Calculate the number of points to select from this bucket
                select_from_bucket = max(1, round((bucket_size / total_points) * n_per_class))
                if bucket_size == 1:
                    final_data.extend(convert_to_array(self.fetch_data_points(key, ids)).tolist())
                    final_labels.extend([key] * len(ids))
                else:
                    metadata = self.fetch_data_points(key, ids)
                    selected_points = apply_selection_criteria(convert_to_array(metadata).tolist(), select_from_bucket,
                                                               1)
                    final_data.extend(selected_points)
                    final_labels.extend([key] * len(selected_points))

        return final_data, final_labels

    def get_selected_data_for_train_faiss(self, call='train'):

        if call == 'train':
            self.ready_for_train = False
            self.newly_added = {i: 0 for i in self.class_list}

        n_per_class = self.size_to_train  # Ensure integer division
        total_points = 0

        final_data = []
        final_labels = []

        # calculate the average ratio of all classes
        # avg_ratio = sum([i['ratio'] for i in self.overall_count.values()]) / len(self.overall_count)
        # w_avg_ratio = sum([i['ratio']/i['count'] for i in self.overall_count.values() if i['count'] > 0]) / sum([1/i['count'] for i in self.overall_count.values() if i['count'] > 0])
        ww_avg_ratio = sum([i['ratio'] * i['count'] for i in self.overall_count.values() if i['count'] > 0]) / sum(
            [i['count'] for i in self.overall_count.values() if i['count'] > 0])

        classes_with_high_ratio = [key for key, value in self.overall_count.items() if value['ratio'] > ww_avg_ratio]

        # let's add the data point to final data where the ratio is low
        for key in self.overall_count.keys():
            if key not in classes_with_high_ratio and len(self.window[key]) > 0:
                final_data.extend(convert_to_array(self.window[key])[-n_per_class:])
                final_labels.extend([key] * len(convert_to_array(self.window[key])[-n_per_class:]))

        lsh_each_class = {i: faiss.IndexLSH(self.num_features, self.vector_size) for i in
                          classes_with_high_ratio}
        table = {i: {} for i in classes_with_high_ratio}
        for key in classes_with_high_ratio:
            # if len(self.window[key]) > 0:
            lsh_each_class[key].add(convert_to_array(self.window[key]))

            # total_points += len(self.window[key])
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

    def get_selected_data_for_train_random(self, call='train'):

        if call == 'train':
            self.ready_for_train = False
            self.newly_added = {i: 0 for i in self.class_list}

        n_per_class = self.size_to_train
        final_data = []
        final_labels = []

        for key in self.class_list:
            class_points = convert_to_array(self.window[key]).tolist()
            if len(class_points) == 0:
                continue
            selected_points = apply_selection_criteria_randomly(class_points, n_per_class)
            final_data.extend(selected_points)
            final_labels.extend([key] * len(selected_points))

        aux = list(zip(final_data, final_labels))
        if len(aux) == 0:
            return [], []
        np.random.shuffle(aux)
        final_data, final_labels = zip(*aux)
        return list(final_data), list(final_labels)

    def get_selected_data_for_train_resampler(self, call='train'):
        """Balance the per-class window with an external resampler (sampler_zoo) -- the ablation of
        the LSH-RHP selection module. Returns (data, labels) at ~size_to_train per class, exactly the
        same contract as the LSH/random paths, so the rest of LSH-DynED is unchanged."""
        from sampler_zoo import resample_balanced
        if call == 'train':
            self.ready_for_train = False
            self.newly_added = {i: 0 for i in self.class_list}
        x_parts, y_parts = [], []
        for key in self.class_list:
            if len(self.window[key]) > 0:
                arr = convert_to_array(self.window[key])
                x_parts.append(np.asarray(arr, dtype=np.float64))
                y_parts.extend([key] * len(arr))
        if not x_parts:
            return [], []
        x_bal, y_bal = resample_balanced(np.vstack(x_parts), np.asarray(y_parts),
                                         self.size_to_train, self.sampling_mode)
        data = x_bal.tolist()
        labels = y_bal.tolist() if hasattr(y_bal, 'tolist') else list(y_bal)
        aux = list(zip(data, labels))
        if len(aux) == 0:
            return [], []
        np.random.shuffle(aux)
        data, labels = zip(*aux)
        return list(data), list(labels)

    def get_selected_data_for_train_active(self, call='train'):
        if self.sampling_mode == 'random_per_class':
            return self.get_selected_data_for_train_random(call=call)
        if self.sampling_mode in ('cluster', 'spe', 'iht', 'smote', 'gsmote'):
            return self.get_selected_data_for_train_resampler(call=call)
        return self.get_selected_data_for_train_faiss(call=call)
