
# By Lior Sidi and Hadar Klein
# based on Ordinal Classifier implementation
# Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification." European Conference on Machine Learning. Springer Berlin Heidelberg, 2001.
# chapter 2

import numpy as np

from Utils import bin_y, ordinal_y

class OrdinalClassifier(object):
    def __init__(self, order, model_class, model_params):
        self.order = order
        self.model_class = model_class
        self.model_params = model_params
        self.clfs = []

    def fit(self, x, y):
        '''
        :param y: y categorial (from pd.qcut)
        :param x: (n x features) array
        '''
        y = ordinal_y(y)
        for y_ in y:
            clf = self.model_class(**self.model_params)
            clf.fit(x, y_)
            self.clfs.append(clf)

    def predict_proba(self, x):
        '''
        :param x: (n x features) array
        :return: pred_probas.transpose()   (nx3 array of predicted probs for each class)
        '''
        pred_probas = np.ndarray((len(self.order), x.shape[0]))
        pred_probas[0] = 1 - self.clfs[0].predict_proba(x)[:, 1]
        pred_probas[len(self.order) - 1] = self.clfs[len(self.order) - 2].predict_proba(x)[:, 1]
        for class_val in range(1, len(self.order) - 1):
            y_i_prev = self.clfs[class_val - 1].predict_proba(x)[:, 1]
            y_i = self.clfs[class_val].predict_proba(x)[:, 1]
            pred_probas[class_val] = y_i_prev - y_i
        return pred_probas.transpose().astype(float)

    def predict(self, x):
        '''
        :param x: (n x features) array
        :return: pred   (nx3 array of predicted probs for each class)
        '''
        pred = self.predict_proba(x)
        pred = np.array(pred)
        for i in range(pred.shape[0]):
            t = np.max(pred[i])
            tmp = pred[i]
            tmp[tmp < t] = 0.
            tmp[tmp >= t] = 1.

        return pred

class BinaryClassifier(object):
    def __init__(self, order, model_class, model_params):
        self.order = order
        self.model_class = model_class
        self.model_params = model_params
        self.clfs = []

    def fit(self, x, y):
        '''
        :param y: y categorial (from pd.qcut)
        :param x: (n x features) array
        '''
        y = bin_y(y)
        y = y.transpose()
        for y_ in y:
            clf = self.model_class(**self.model_params)
            clf.fit(x, y_)
            self.clfs.append(clf)

    def predict_proba(self, x):
        '''
        :param x: (n x features) array
        :return: pred_probas.transpose()   (nx3 array of predicted probs for each class)
        '''
        pred_probas = np.ndarray((len(self.order), x.shape[0]))
        for class_val in range(0, len(self.order) - 1):
            y_i = self.clfs[class_val].predict_proba(x)[:, 1]
            pred_probas[class_val] = y_i
        return pred_probas.transpose().astype(float)

    def predict(self, x):
        '''
        :param x: (n x features) array
        :return: pred   (nx3 array of predicted probs for each class)
        '''
        pred = self.predict_proba(x)
        pred = np.array(pred)
        for i in range(pred.shape[0]):
            t = max(pred[i])
            tmp = pred[i]
            tmp[tmp < t] = 0.
            tmp[tmp >= t] = 1.
        return pred