import numpy as np
from river.drift.adwin import ADWIN
import classifier as cls
import Sliding_window as sw
from diversity import mmr, kappa_metric
from sklearn.cluster import KMeans
import random


class Ensemble(object):
    def __init__(self, num_components: int, num_features: int, class_list: list, max_pool_size: int, wind_size: int,
                 ftr: list, size_to_train: int, size_to_test: int, vector: int):
        self.vector = vector
        self.lamda = 0.6
        self.stream_acc = []
        self.num_components = num_components
        self.num_cls = len(class_list)
        self.class_list = class_list
        self.num_features = num_features
        self.features = ftr
        self.sample_size_train = size_to_train
        self.sample_size_test = size_to_test

        self.component_error_accuracy = None
        self.component_error = None
        self.model = {'active': [cls.Classifier(class_list=self.class_list,
                                                ftr=self.features) for _ in range(num_components)],
                      'reserve': [], 'combined': []}
        self.max_pool_size = max_pool_size
        self.new_cdd = ADWIN(delta=7e-1)
        self.drift = False
        self.drift_detected_count = 0
        self.tmp_drift_detected_count = 0

        self.window = sw.SlidingWindowLSH(wind_size, self.num_cls, class_list, num_features - 1, vector, size_to_train,size_to_test)

        self.overall_accuracy = 0
        self.predicted_samples = 0
        self.aux_cnt = 0

    def predict(self, dx, dy):
        y_hat = []
        for i in range(len(self.model['active'])):
            y_hat.append(self.model['active'][i].predict_prob(dx, dy))
        pred = self._soft_voting(y_hat, dy)
        self._update_new_cdd(dy, pred)
        self.aux_cnt += 1
        self.learn_one(dx, dy)
        if self.aux_cnt == 1000:
            self.aux_cnt = 0
        self.window.add_to_window(dx, dy)
        self.predicted_samples += 1
        return pred

    def learn_one(self, dx, dy):
        if self.drift:
            self._add_classifier()

        if self.aux_cnt == 1000 and len(self.model['reserve']) != 0 or self.drift:
            self._update_ensemble()
            self.drift = False

        for i in range(len(self.model['active'])):
            self.model['active'][i].learn_one(dx, dy)

    def _update_new_cdd(self, y, y_hat):
        x = 1 if (y == y_hat) else 0
        self.new_cdd.update(x)
        if self.new_cdd.drift_detected:
            self.drift_detected_count += 1
            self.drift = True

    def _soft_voting(self, predictions, true_label):
        """
        :param predictions: list of dictionaries of probability for each class, each dictionary is for one component
        :return: final vote
        """
        predictions_np = np.zeros((len(predictions), len(self.class_list)))
        for i in range(len(predictions)):
            for j in predictions[i].keys():
                predictions_np[i][int(j)] = predictions[i][int(j)]
        mean_predictions = np.mean(predictions_np, axis=0)
        if len(np.unique(mean_predictions)) == 1:
            valid_indices = [i for i, x in enumerate(mean_predictions) if x != true_label]
            return random.choice(valid_indices)
        return np.argmax(np.mean(predictions_np, axis=0))

    def learn_whole(self, x, y):
        for learners in self.model['active']:
            for i in range(len(x)):
                learners.learn_one(x[i], y[i])

    def _add_classifier(self):
        self.model['reserve'].append(cls.Classifier(class_list=self.class_list, ftr=self.features))
        data, labels = self.window.get_selected_data_for_train_faiss(call='new classifier')
        self.model['reserve'][-1].learn_whole(data, labels)

    def _update_ensemble(self):

        self.drift = False

        self.model['combined'] = self.model['active'] + self.model['reserve']
        self.model['active'] = self.model['reserve'] = []

        self.component_error = {j: None for j in range(len(self.model['combined']))}
        self.component_error_accuracy = {j: 0.0 for j in range(len(self.model['combined']))}

        data, labels = self.window.get_data_for_pred()
        cluster = KMeans(n_clusters=2, random_state=101)
        for j in range(len(self.model['combined'])):
            self.model['combined'][j].prediction_error(data, labels)
            self.component_error[j] = self.model['combined'][j].prediction_list
            self.component_error_accuracy[j] = self.model['combined'][j].get_aux_accuracy()

        cluster.fit(list(self.component_error.values()))
        labels = cluster.labels_
        unique_labels = set(labels)
        selected_indices = []
        for i in unique_labels:
            indices = [j for j in range(len(labels)) if labels[j] == i]
            sorted_indices = sorted(indices, key=lambda x: self.component_error_accuracy[x], reverse=True)
            selected_indices += sorted_indices[:self.num_components]

        diversity = kappa_metric([self.component_error[j] for j in selected_indices])
        new, score = mmr(selected_indices, diversity, [self.component_error_accuracy[j] for j in selected_indices], self.lamda,
                         self.num_components)
        for j in new:
            self.model['active'].append(self.model['combined'][j])

        data, labels = self.window.get_selected_data_for_train_faiss(call='new classifier')
        for learner in self.model['active']:
            learner.learn_whole(data, labels)

        self.model['reserve'] = [cpmt for cpmt in self.model['combined'] if cpmt not in self.model['active']]
        self.model['combined'] = []

        # check pool size
        if len(self.model['reserve']) > self.max_pool_size:
            self.model['reserve'] = sorted(self.model['reserve'], key=lambda x: x.get_kappa())
            self.model['reserve'] = self.model['reserve'][:self.max_pool_size]

    def set_values_for_lamda_update(self, str_acc):
        self.stream_acc.append(str_acc)

    def update_lamda(self):
        slope = 0
        if len(self.stream_acc) > 100 and (self.stream_acc[-1] - self.stream_acc[-100]) != 0:
            slope = (self.stream_acc[-1] - self.stream_acc[-100]) / 100
        if slope >= 0:
            if self.lamda <= 0.9:
                self.lamda += 0.1
        else:
            if self.lamda >= 0.1:
                self.lamda -= 0.1

