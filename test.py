
# By Lior Sidi and Hadar Klein
# based on Ordinal Classifier implementation
# Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification." European Conference on Machine Learning. Springer Berlin Heidelberg, 2001.
# chapter 2

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from sklearn import datasets
import pandas as pd

import numpy as np

from OrdinalClassifier import OrdinalClassifier, BinaryClassifier
from Utils import bin_y, load_facebook, load_bike_rental, load_OnlineNewsPopularity, load_student_grades


def main():
    uci_datasets = [
        {
            'name': 'boston',
            'df': datasets.load_boston(),
            'order': ["low", "medium", "high"]
        }, {
            'name': 'facebook',
            'df': load_facebook(),
            'order': ["low", "medium", "high"]
        }, {
            'name': 'bike',
            'df': load_bike_rental(),
            'order': ["low", "medium", "high"]
        }, {
            'name': 'news',
            'df': load_OnlineNewsPopularity(),
            'order': ["low", "medium", "high"]
        }, {
            'name': 'student',
            'df': load_student_grades(),
            'order': ["low", "medium", "high"]
        }]

    models = [{
        'name': 'GradientBoosting',
        'clf_model': GradientBoostingClassifier,
        'model_params': {'n_estimators': 10}
    }, {
        'name': 'RandomForest',
        'clf_model': RandomForestClassifier,
        'model_params': {'n_estimators': 10}
    }, {
        'name': 'DecisionTree',
        'clf_model': DecisionTreeClassifier,
        'model_params': {}
    }]

    results = []
    for uci_dataset in uci_datasets:
        df = uci_dataset['df']
        order = uci_dataset['order']
        X, y = shuffle(df.data, df.target, random_state=13)
        y = pd.qcut(y, len(order), labels=order)
        offset = int(X.shape[0] * 0.7)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]
        y_test = bin_y(y_test)

        for model in models:
            res = {}
            clf_model = model['clf_model']
            model_params = model['model_params']

            res['dataset'] = uci_dataset['name']
            res['model'] = model['name']
            ocf = OrdinalClassifier(order, clf_model, model_params)
            ocf.fit(X_train, y_train)
            preds_ocf = ocf.predict(X_test)

            bcf = BinaryClassifier(order, clf_model, model_params)
            bcf.fit(X_train, y_train)
            preds_bcf = bcf.predict(X_test)
            preds_bcf = np.nan_to_num(preds_bcf)
            print model
            print preds_bcf
            res['acc_ordinal'] = accuracy_score(y_test, preds_ocf)
            res['acc_binary'] = accuracy_score(y_test, preds_bcf)
            del preds_ocf
            del preds_bcf

            del bcf
            del ocf
            results.append(res)

    df_res = pd.DataFrame(results)
    print df_res
    df_res.to_csv('results.csv')

if __name__ == '__main__':
    main()