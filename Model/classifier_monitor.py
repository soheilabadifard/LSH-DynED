import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('always')


class Metrics:
    def __init__(self, actual: list, predicted: list, cls_list: list):
        """Initializes a Metrics object.
    Args:
      actual: A list of the actual labels.
      predicted: A list of the predicted labels.
      cls_list: A list of the classes in the data.
    """
        self._true_class_list = cls_list
        self._actual = actual
        self._predicted = predicted

        self._confusion_matrix = None

        self._g_mean = 0.0

        self._calculate_confusion_matrix()
        self._calculate_other_metrics()

    def _calculate_confusion_matrix(self):
        """Calculates the confusion matrix."""
        self._confusion_matrix = confusion_matrix(self._actual, self._predicted, labels=self._true_class_list)

    def _calculate_other_metrics(self):
        results = classification_report_imbalanced(y_true=self._actual, y_pred=self._predicted,
                                                   labels=self._true_class_list,
                                                   output_dict=True,
                                                   zero_division=0)
        self._g_mean = results['avg_geo']

    def get_scores(self):
        a = {'g_mean': self._g_mean}
        for cls1 in self._true_class_list:
            for cls2 in self._true_class_list:
                a[f'CM[{int(cls1)}][{int(cls2)}]'] = self._confusion_matrix[int(cls1)][int(cls2)]
        return a


class StreamConfusionMatrix:
    def __init__(self, cls_list: list):
        """Initializes a StreamConfusionMatrix object.
    Args:
      cls_list: A list of the classes in the data.
    """
        self._true_class_list = cls_list
        # self._class_list = cls_list if cls_list[0] == 0 else [i - 1 for i in cls_list]
        self._class_begin_zero = cls_list[0] == 0
        self._confusion_matrix = np.zeros((len(cls_list), len(cls_list)), dtype=int) if self._class_begin_zero else \
            np.zeros((len(cls_list) + 1, len(cls_list) + 1), dtype=int)

    def update_confusion(self, y_true, y_pred):
        """Updates the confusion matrix.
    Args:
      y_true: An integer representing the true label.
      y_pred: An integer representing the predicted label.
    """
        # update confusion matrix
        self._confusion_matrix[int(y_true)][int(y_pred)] += 1

    def get_confusion_matrix(self):
        """Returns the confusion matrix.
    Returns:
      A NumPy array of the confusion matrix.
    """
        if self._class_begin_zero:
            return self._confusion_matrix
        else:
            return self._confusion_matrix[1:, 1:]

    def reset(self):
        """Resets the confusion matrix."""
        self._confusion_matrix = np.zeros((len(self._true_class_list), len(self._true_class_list)), dtype=int) if \
            self._class_begin_zero else np.zeros((len(self._true_class_list) + 1, len(self._true_class_list) + 1),
                                                 dtype=int)

    def calculate_kappa(self):
        """Calculates the Cohen's Kappa statistic for the confusion matrix."""
        po = np.trace(self._confusion_matrix) / np.sum(self._confusion_matrix)  # Observed agreement
        pe_rows = np.sum(self._confusion_matrix, axis=1)  # Sum over rows
        pe_cols = np.sum(self._confusion_matrix, axis=0)  # Sum over columns
        pe = np.sum(pe_rows * pe_cols) / np.sum(self._confusion_matrix) ** 2  # Expected agreement

        kappa = (po - pe) / (1 - pe)
        return kappa


class SlidingWindowForMetrics:
    def __init__(self, class_list):
        self.prediction_window = []
        self.true_window = []
        self.window_size = 500
        self.result_df = pd.DataFrame(columns=['kappa'],dtype=float)
        self.class_info = class_list
        self.matrix = None

    def add_to_window(self, pred,  true):
        self.prediction_window.append(pred)
        self.true_window.append(true)
        if len(self.prediction_window) > self.window_size:
            self.prediction_window.pop(0)
            self.true_window.pop(0)
        if len(self.prediction_window) >= 5:
            self.calculate_metrics()

    def calculate_metrics(self):
        kappa = cohen_kappa_score(y1=self.true_window, y2=self.prediction_window, labels=self.class_info)
        data = {'kappa': kappa}

        self.result_df = pd.concat([self.result_df, pd.DataFrame([data])], ignore_index=True)

    def get_results_df(self):
        self.result_df['kappa'] = self.result_df['kappa'].replace('?', 0).astype(float)
        self.result_df['kappa'] = self.result_df['kappa'].replace(np.nan, 0).astype(float)
        adjusted_data = self.result_df.iloc[:496]
        remaining_data = self.result_df.iloc[496:]

        first_chunk_avg = adjusted_data.mean().to_frame().T
        remaining_chunks_avg = remaining_data.groupby((remaining_data.index - 496) // 500).mean()

        combined_avg = pd.concat([first_chunk_avg, remaining_chunks_avg], ignore_index=True)

        last_chunk_size = len(self.result_df) - 496 - (len(self.result_df) - 496) // 500 * 500
        last_range = (496 + (len(self.result_df) - 496) // 500 * 500 + last_chunk_size) + 4

        ranges = [500] + [500 * (i + 1) for i in range(1, len(combined_avg) - 1)] + [last_range]
        combined_avg['count'] = ranges

        combined_avg = combined_avg[['count'] + [col for col in combined_avg.columns if col != 'count']]
        return combined_avg


def calculate_accuracy(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1
    accuracy = correct / len(actual)
    return accuracy


def check_true(dy, y_hat):
    return 1 if (dy == y_hat) else 0

