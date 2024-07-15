from river.tree import HoeffdingTreeClassifier
from data_related_functions import prepare_data_with_features
from classifier_monitor import StreamConfusionMatrix
import numpy as np

class Classifier(HoeffdingTreeClassifier):

    def __init__(self, class_list, ftr):
        super().__init__(grace_period=50, delta=0.9)  # delta=0.9, grace_period=50, grace_period=50, delta=0.9

        self.y_prediction_prob = None  # variable to store the prediction probability
        self.classifier_confusion_matrix = StreamConfusionMatrix(class_list)
        self.cls_list = class_list  # list of classes
        self.num_of_class = len(class_list)  # number of classes
        self.feature_list = ftr

        self._aux_accuracy = 0.0  # variable to calculate the accuracy over last n samples
        self.accuracy = 0.0  # variable for overall accuracy

        self.y_prediction = None  # variable to store the prediction

        self.predicted_samples = 0  # count of predicted samples till now
        self.true_predicted = 0  # count of true predicted samples till now

        self.max_length = 500  # max length of the list to store the prediction error
        self.prediction_list = []  # list of prediction 1 if prediction true 0 if false

        self.sample_count_per_class = {i: 0 for i in class_list}  # count of samples per class
        self.prediction_per_class = {i: [] for i in class_list}  # list of prediction per class
        self.true_prediction_count_per_class = {i: 0 for i in class_list}  # count of true prediction per class
        self.accuracy_per_class = {i: 0.0 for i in class_list}  # accuracy per class

    def predict(self, dx, dy=None):
        """
        Predict the class of the sample
        :param dx: features of the sample in format of dictionary
        :param dy: true label of the sample
        :return: predicted class of the sample
        """
        self.predicted_samples += 1
        self.y_prediction = super().predict_one(dx)

        if self.y_prediction is None:
            rnd = None
            flag = True
            while flag:
                rnd = np.random.choice(self.cls_list)
                if rnd != dy:
                    flag = False
            self.y_prediction = rnd

        self.sample_count_per_class[dy] += 1
        self.prediction_per_class[dy].append(1 if self.y_prediction == dy else 0)
        self._keep_max_length_prediction_per_class()
        self._calc_metrics(dy)
        self.classifier_confusion_matrix.update_confusion(dy, self.y_prediction)
        return self.y_prediction

    def predict_prob(self, dx, dy=None):
        """
        Predict the probability of the sample
        :param dx: features of the sample in format of dictionary
        :param dy: true label of the sample
        :return: predicted probability of the sample
        """
        self.y_prediction_prob = super().predict_proba_one(dx)
        self.y_prediction = self.predict(dx, dy)

        if not self.y_prediction_prob:
            self.y_prediction_prob = {int(i): 1/len(self.cls_list) for i in self.cls_list}

        return self.y_prediction_prob

    def learn_whole(self, dx, dy):
        """
        Learn the whole dataset
        :param dx: features of the samples in format of dictionary
        :param dy: true label of the samples
        """
        for x, y in zip(dx, dy):
            prepared_x, prepared_y = prepare_data_with_features(x, y, self.feature_list)
            self.learn_one(prepared_x, prepared_y)

    def prediction_error(self, dx, dy):
        """
        Calculate the prediction error of the given samples
        :param dx: features of the samples in format of dictionary
        :param dy: true label of the samples
        """
        self.prediction_list = []
        for x, y in zip(dx, dy):
            y_prediction_aux = super().predict_one(x)
            self.prediction_list.append(1 if y_prediction_aux == y else 0)
        self._calc_aux_accuracy()

    def _calc_aux_accuracy(self):
        self._aux_accuracy = sum(self.prediction_list) / len(self.prediction_list)

    def _calc_metrics(self, dy):
        if self.y_prediction == dy:
            self.true_predicted += 1
            self.true_prediction_count_per_class[dy] += 1
        self.accuracy = (self.true_predicted / self.predicted_samples) * 100
        self.accuracy_per_class[dy] = (self.true_prediction_count_per_class[dy] / self.sample_count_per_class[dy]) * 100

    def _keep_max_length_prediction_error(self):
        if len(self.prediction_list) > self.max_length:
            self.prediction_list = self.prediction_list[-self.max_length:]

    def _keep_max_length_prediction_per_class(self):
        for i in self.prediction_per_class:
            if len(self.prediction_per_class[i]) > self.max_length:
                self.prediction_per_class[i] = self.prediction_per_class[i][-self.max_length:]

    def get_accuracy(self):
        return self.accuracy

    def get_accuracy_per_class(self):
        return self.accuracy_per_class

    def get_aux_accuracy(self):
        return self._aux_accuracy

    def __lt__(self, other):
        return self.accuracy > other.accuracy

    def __repr__(self, **kwargs) -> str:
        return str(self.accuracy)

    def get_predicted_samples_count(self):
        return self.predicted_samples

    def get_weight(self):
        return self.true_predicted / self.predicted_samples

    def get_kappa(self):
        return self.classifier_confusion_matrix.calculate_kappa()
