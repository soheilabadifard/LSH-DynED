import numpy as np
import pandas as pd
from scipy.stats import gmean
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
# from multiclass_auc import Estimator
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('always')

np.random.seed(101)


class IncrementalMetrics:
    def __init__(self, cls_list: list):
        """Initializes an IncrementalGMetrics object.
    Args:
      cls_list: The list of classes in the data.
    """
        self._true_class_list = cls_list
        self._class_list = cls_list if cls_list[0] == 0 else cls_list - 1
        self._class_begin_zero = cls_list[0] == 0
        self._epsilon = 1e-8

        self._num_classes = len(cls_list)
        self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=int)  # rows are true labels
        # and columns are predicted labels

        self._macro_avg_g_mean = 0
        # self._micro_avg_g_mean = 0

        self._macro_avg_recall = 0
        self._micro_avg_recall = 0

        self._macro_avg_precision = 0
        self._micro_avg_precision = 0

        self._macro_avg_f1 = 0
        # self._micro_avg_f1 = 0

        self._specificity = 0

        #  a dictionary to keep track of the number of samples seen for each class
        self._class_statistics = {i: {'count': 0, 'true_positive': 0, 'false_positive': 0, 'false_negative': 0,
                                      'true_negative': 0, 'gm': 0.0, 'recall': 0.0, 'precision': 0.0,
                                      'specificity': 0.0,
                                      'f1': 0.0}
                                  for i in self._true_class_list}

    def update_confusion(self, y_true, y_pred):
        """Updates confusion matrix.
    Args:
      y_true: An integer representing the true label.
      y_pred: An integer representing the predicted label.
    """
        # update confusion matrix
        self._confusion_matrix[y_true][y_pred] += 1
        self.update_statistics(y_true)

    # update class statistics based on the confusion matrix
    def update_statistics(self, y_true):
        """Updates the G-Mean, recall, and precision for each class.
        Args:
            y_true: An integer representing the true label.
        """
        self._class_statistics[y_true]['count'] += 1
        self._class_statistics[y_true]['true_positive'] = self._confusion_matrix[y_true][y_true]
        self._class_statistics[y_true]['false_negative'] = np.sum(self._confusion_matrix[y_true]) - \
                                                           self._confusion_matrix[y_true][y_true]
        self._class_statistics[y_true]['false_positive'] = np.sum(self._confusion_matrix[:, y_true]) - \
                                                           self._confusion_matrix[y_true][y_true]
        self._class_statistics[y_true]['true_negative'] = np.sum(self._confusion_matrix) - np.sum(
            self._confusion_matrix[y_true]) - np.sum(self._confusion_matrix[:, y_true]) + \
                                                          self._confusion_matrix[y_true][y_true]

        # Calculate recall (sensitivity) for class y_true
        self._class_statistics[y_true]['recall'] = self._class_statistics[y_true]['true_positive'] / (
                self._class_statistics[y_true]['true_positive'] + self._class_statistics[y_true][
            'false_negative'] + self._epsilon)

        # Calculate precision for class y_true
        self._class_statistics[y_true]['precision'] = self._class_statistics[y_true]['true_positive'] / (
                self._class_statistics[y_true]['true_positive'] + self._class_statistics[y_true][
            'false_positive'] + self._epsilon)

        # Calculate specificity for class y_true
        self._class_statistics[y_true]['specificity'] = self._class_statistics[y_true]['true_negative'] / (
                self._class_statistics[y_true]['true_negative'] + self._class_statistics[y_true][
            'false_positive'] + self._epsilon)

        # Calculate g-mean for class y_true
        self._class_statistics[y_true]['gm'] = np.sqrt(
            self._class_statistics[y_true]['recall'] * self._class_statistics[y_true]['specificity'])

        # Calculate f1 for class y_true
        self._class_statistics[y_true]['f1'] = 2 * (
                self._class_statistics[y_true]['precision'] * self._class_statistics[y_true]['recall']) / \
                                               (self._class_statistics[y_true]['precision'] +
                                                self._class_statistics[y_true]['recall'])

    def get_specificity(self):
        """Returns the overall specificity."""
        self._specificity = np.mean([self._class_statistics[cls]['specificity'] for cls in self._true_class_list])
        return self._specificity

    def get_macro_avg_g_mean(self):
        """Returns the macro-average G-Mean.
        Returns:
        A float of the macro-average G-Mean.
        """
        self._macro_avg_g_mean = gmean([self._class_statistics[cls]['gm'] for cls in self._true_class_list])
        return self._macro_avg_g_mean

    def get_macro_avg_recall(self):
        """Returns the macro-average recall.
        Returns:
        A float of the macro-average recall.
        """
        self._macro_avg_recall = np.mean([self._class_statistics[cls]['recall'] for cls in self._true_class_list])
        return self._macro_avg_recall

    def get_macro_avg_precision(self):
        """Returns the macro-average precision.
        Returns:
        A float of the macro-average precision.
        """
        # Calculate macro precision
        self._macro_avg_precision = np.mean([self._class_statistics[cls]['precision'] for cls in self._true_class_list])
        return self._macro_avg_precision

    # def get_micro_avg_g_mean(self):
    #    """Returns the micro-average G-Mean.
    #    Returns:
    #    A float of the micro-average G-Mean.
    #    """
    #    return self._micro_avg_g_mean

    def get_micro_avg_recall(self):
        """Returns the micro-average recall.
        Returns:
        A float of the micro-average recall.
        """
        self._micro_avg_recall = np.sum(
            [self._class_statistics[cls]['true_positive'] for cls in self._true_class_list]) / \
                                 np.sum([self._class_statistics[cls]['true_positive'] + self._class_statistics[cls][
                                     'false_negative'] for cls in self._true_class_list])
        return self._micro_avg_recall

    def get_micro_avg_precision(self):
        """Returns the micro-average precision.
        Returns:
        A float of the micro-average precision.
        """
        # Calculate micro precision
        self._micro_avg_precision = np.sum(
            [self._class_statistics[cls]['true_positive'] for cls in self._true_class_list]) / np.sum(
            [self._class_statistics[cls]['true_positive'] + self._class_statistics[cls]['false_positive'] for cls in
             self._true_class_list])
        return self._micro_avg_precision

    def get_macro_avg_f1(self):
        """Returns the macro-average F1 score.
        micro F1 score gives equal importance to each observation, whereas macro F1 score gives equal importance to each class.
        Micro F1 score often does not return an objective measure of model performance when the classes are imbalanced, whilst macro F1 score is able to do so.
        Returns:
        A float of the macro-average F1 score.
        """
        self._macro_avg_f1 = np.mean([self._class_statistics[cls]['f1'] for cls in self._true_class_list])
        return self._macro_avg_f1

    # def get_micro_avg_f1(self):
    #    """Returns the micro-average F1 score.
    #    micro F1 score gives equal importance to each observation, whereas macro F1 score gives equal importance to each class.
    #    Micro F1 score often does not return an objective measure of model performance when the classes are imbalanced, whilst macro F1 score is able to do so.
    #    Returns:
    #    A float of the micro-average F1 score.
    #    """
    #    self._micro_avg_f1 = np.sum(
    #        [self._class_statistics[cls]['true_positive'] for cls in self._true_class_list]) / \
    #                         np.sum([self._class_statistics[cls]['true_positive'] for cls in self._true_class_list]) + \
    #                         1 / 2 * np.sum(
    #        [self._class_statistics[cls]['false_positive'] + self._class_statistics[cls]['false_negative'] for cls in
    #         self._true_class_list])
    #    return self._micro_avg_f1

    def get_confusion_matrix(self):
        """Returns the confusion matrix.
        Returns:
        A NumPy array of the confusion matrix.
        """
        return self._confusion_matrix

    def get_g_means(self):
        """Returns the G-Mean for each class.
    Returns:
      A NumPy array of the G-Mean for each class.
    """
        return [self._class_statistics[cls]['gm'] for cls in self._true_class_list]

    def get_recalls(self):
        """Returns the recall for each class.
    Returns:
      A NumPy array of the recall for each class.
    """
        return [self._class_statistics[cls]['recall'] for cls in self._true_class_list]

    def get_precisions(self):
        """Returns the precision for each class.
    Returns:
      A NumPy array of the precision for each class.
    """
        return [self._class_statistics[cls]['precision'] for cls in self._true_class_list]

    def get_specificities(self):
        """Returns the specificity for each class.
        Returns:
        A NumPy array of the specificity for each class.
        """
        return [self._class_statistics[cls]['specificity'] for cls in self._true_class_list]

    def get_class_statistics(self):
        # return the tp for each class with class indication
        return {cls: self._class_statistics[cls]['true_positive'] for cls in self._true_class_list}

    def reset(self):
        self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=int)
        self._macro_avg_g_mean = 0
        # self._micro_avg_g_mean = 0
        self._macro_avg_recall = 0
        self._micro_avg_recall = 0
        self._macro_avg_precision = 0
        self._micro_avg_precision = 0
        self._macro_avg_f1 = 0
        # self._micro_avg_f1 = 0
        self._specificity = 0
        self._class_statistics = {i: {'count': 0, 'true_positive': 0, 'false_positive': 0, 'false_negative': 0,
                                      'true_negative': 0, 'gm': 0.0, 'recall': 0.0, 'precision': 0.0,
                                      'specificity': 0.0,
                                      'f1': 0.0}
                                  for i in self._true_class_list}


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

        self._avg_geo = 0.0
        self._recall = 0.0
        self._precision = 0.0
        self._f1 = 0.0
        self._specificity = 0.0
        self._kappa = 0.0

        self._calculate_confusion_matrix()
        self._calculate_other_metrics()

    def _calculate_confusion_matrix(self):
        """Calculates the confusion matrix."""
        self._confusion_matrix = confusion_matrix(self._actual, self._predicted, labels=self._true_class_list)

    def _calculate_other_metrics(self):
        results = classification_report_imbalanced(y_true=self._actual, y_pred=self._predicted,
                                                   labels=self._true_class_list,
                                                   # labels=np.unique(self._actual),
                                                   #labels=np.intersect1d(self._true_class_list,
                                                   #                     np.unique(self._predicted)),
                                                   #labels=np.unique(self._predicted),
                                                   output_dict=True,
                                                   zero_division=1)
        self._recall = results['avg_rec']
        self._precision = results['avg_pre']
        self._f1 = results['avg_f1']
        # imblearn's avg_geo, not the paper's mG-Mean; the reported tables recompute mG-Mean from the CM[i][j] columns
        self._avg_geo = results['avg_geo']
        # self._specificity = results['avg_spe']
        self._kappa = cohen_kappa_score(y1=self._actual, y2=self._predicted, labels=self._true_class_list)

    def get_scores(self):
        a = {'avg_geo': self._avg_geo,
             'recall': self._recall,
             'precision': self._precision,
             'f1': self._f1,
             'kappa': self._kappa}
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
        #self.prediction_prob_window = []
        self.true_window = []
        self.window_size = 500
        self.result_df = pd.DataFrame(columns=['avg_geo', 'recall', 'precision', 'f1', 'kappa'],
                                      dtype=float)
        self.class_info = class_list
        self.matrix = None
        # self.est = Estimator(self.window_size, len(self.class_info))

    def add_to_window(self, pred,  true):
        # self.est.add(prob, true, check_true(true, pred))
        self.prediction_window.append(pred)
        self.true_window.append(true)
        #self.prediction_prob_window.append(prob.tolist())
        if len(self.prediction_window) > self.window_size:
            self.prediction_window.pop(0)
            self.true_window.pop(0)
            #self.prediction_prob_window.pop(0)
        self.matrix = confusion_matrix(self.true_window, self.prediction_window, labels=self.class_info)
        if len(self.prediction_window) >= 5:
            self.calculate_metrics()

    def calculate_metrics(self):
        results = classification_report_imbalanced(y_true=self.true_window, y_pred=self.prediction_window,
                                                   labels=self.class_info,
                                                   # labels=np.unique(self._actual),
                                                   #labels=np.intersect1d(np.unique(self.true_window),
                                                   #                      np.unique(self.prediction_window)),
                                                   #labels=np.unique(self.prediction_window),
                                                   output_dict=True,
                                                   zero_division=1)

        recall = results['avg_rec']
        precision = results['avg_pre']
        f1 = results['avg_f1']
        gm = results['avg_geo']
        kappa = cohen_kappa_score(y1=self.true_window, y2=self.prediction_window, labels=self.class_info)
        #pmauc = roc_auc_score(np.array(self.true_window), np.array(self.prediction_prob_window), labels=self.class_info.tolist(), multi_class='ovo', average='macro')
        # pmauc = self.est.get_pmauc()
        data = {'avg_geo': gm, 'recall': recall, 'precision': precision,
                'f1': f1, 'kappa': kappa}

        self.result_df = pd.concat([self.result_df, pd.DataFrame([data])], ignore_index=True)

    def get_window(self):
        return self.prediction_window, self.true_window

    def get_results_df(self):
        return self.result_df


def calculate_accuracy(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1
    accuracy = correct / len(actual)
    return accuracy


def check_true(dy, y_hat):
    return 1 if (dy == y_hat) else 0


"""corrected_classified = np.trace(self.matrix)
        number_instances_total = np.sum(self.matrix)

        number_instances = np.sum(self.matrix, axis=0)  # Sum over columns for each class
        predicted_instances = np.sum(self.matrix, axis=1)  # Sum over rows for each class

        mul = np.sum(number_instances * predicted_instances)

        if (number_instances_total ** 2 - mul) != 0:
            kappa = ((number_instances_total * corrected_classified) - mul) / (number_instances_total ** 2 - mul)
        else:
            kappa = 1.0

        num_classes = self.matrix.shape[0]
        gmean = 1.0

        for i in range(num_classes):
            if number_instances[i] != 0:
                gmean *= (self.matrix[i, i] / number_instances[i])

        gm = np.power(gmean, 1.0 / num_classes)

        precision_sum = 0
        existing_classes = 0
        for i in range(num_classes):
            if predicted_instances[i] != 0:
                class_precision = self.matrix[i, i] / predicted_instances[i]
                precision_sum += class_precision
                existing_classes += 1

        precision = precision_sum / existing_classes if existing_classes > 0 else 0

        recall_sum = 0
        existing_classes = 0
        for i in range(num_classes):
            if number_instances[i] != 0:
                class_recall = self.matrix[i, i] / number_instances[i]
                recall_sum += class_recall
                existing_classes += 1

        recall = recall_sum / existing_classes if existing_classes > 0 else 0"""
