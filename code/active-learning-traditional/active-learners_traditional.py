import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification
from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.data import SklearnDataset
from small_text.initialization import random_initialization, random_initialization_stratified
from small_text.query_strategies import RandomSampling

from qbc_strategy import QBC

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


# dummy function for tokenization and preprocessing steps
def dummy_fun(doc):
    return doc


def prepare_data(df):
    # tf-idf vectorize feature vectors
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,
                                 preprocessor=dummy_fun, token_pattern=None, max_df=0.8, min_df=2)
    le = LabelEncoder()

    X_df = vectorizer.fit_transform(df['preprocessed_text'])
    print("vocabulary size:", len(vectorizer.vocabulary_))

    # single label preparation
    y_df = le.fit_transform(df['main_category_level1'])

    # multi label preparation
    multi_cols = ['level1_traffic_lights', 'level1_lighting', 'level1_signage', 'level1_bicycle_parking',
                  'level1_obstacles', 'level1_cycling_traffic_management', 'level1_cycle_path_quality',
                  'level1_misc']
    y_df_multi = df[multi_cols].to_numpy()

    return X_df, y_df, y_df_multi


# simulates an initial labeling to warm-start the active learning process
def initialize_active_learner(active_learner, y_train, x_train, multi_label=False):

    if multi_label:
        indices_initial = random_initialization_stratified(y_train, n_samples=20)
    else:
        indices_initial = random_initialization(x_train, n_samples=20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


def evaluate(active_learner, train, test):
    y_pred_train = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = f1_score(test.y, y_pred_test, average='micro')

    print('Train accuracy: {:.2f}'.format(f1_score(train.y, y_pred_train, average='micro')))
    print('Test accuracy: {:.2f}'.format(test_acc))

    print(classification_report(test.y, y_pred_test))

    return test_acc


def pool_transformation(pool):
    return [e.sum(axis=0) for e in pool]


# run 5-fold CV
def AlCV(X_in, y_in, qs, num_classes, clf_model, split_nr, multi_label):
    X = X_in
    y = y_in

    if multi_label:
        skf = IterativeStratification(n_splits=5, order=1)
    else:
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)

    split = 1

    for train_index, test_index in skf.split(X, y):

        print("Split", split)

        if split >= split_nr:

            # prepare dataset
            X_train, X_test = X_in[train_index].toarray(), X_in[test_index].toarray()
            if multi_label:
                y_train, y_test = sparse.csr_matrix(y_in[train_index]), sparse.csr_matrix(y_in[test_index])
            else:
                y_train, y_test = y_in[train_index], y_in[test_index]

            train = SklearnDataset(X_train, y_train)
            test = SklearnDataset(X_test, y_test)

            # init the model and query strategy
            model = clf_model
            classifier_kwargs = {'multi_label': multi_label}
            clf_factory = SklearnClassifierFactory(model, num_classes, kwargs=classifier_kwargs)

            query_strategy = qs
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train, reuse_model=True)
            indices_labeled = initialize_active_learner(active_learner, train.y, train.x, multi_label)

            # active learning loop
            num_samples = 20
            tmp = (len(X_train) - num_samples) / num_samples
            if tmp > int(tmp):
                num_queries = min(int(tmp) + 1, 30)
            else:
                num_queries = min(int(tmp), 30)

            # time stamps
            query_times = []
            user_simulation_times = []
            update_times = []
            evaluation_times = []
            overall_iteration_times = []

            # get results per iteration
            results = [evaluate(active_learner, train[indices_labeled], test)]

            # get labels per iteration
            labels_in_pool = [train.y[indices_labeled]]

            for i in range(num_queries):

                start_time = time.time()

                # ...where each iteration consists of labelling 20 samples
                if i == num_queries - 1:
                    num_samples = len(X_train) - (num_queries * num_samples)

                indices_queried = active_learner.query(num_samples=num_samples)

                query_time = time.time()
                query_times.append(query_time - start_time)

                # Simulate user interaction here. Replace this for real-world usage.
                y = train.y[indices_queried]

                user_simulation_time = time.time()
                user_simulation_times.append(user_simulation_time - query_time)

                # Return the labels for the current query to the active learner.
                active_learner.update(y)

                indices_labeled = np.concatenate([indices_queried, indices_labeled])

                update_time = time.time()
                update_times.append(update_time - user_simulation_time)

                print('---------------')
                print(f'Iteration #{i} ({len(indices_labeled)} samples)')
                results.append(evaluate(active_learner, train[indices_labeled], test))

                labels_in_pool.append(train.y[indices_labeled])

                evaluation_time = time.time()
                evaluation_times.append(evaluation_time - update_time)

                overall_iteration_times.append(evaluation_time - start_time)

            if multi_label:
                labels_in_pool = pool_transformation(labels_in_pool)

            summary = {
                'results': results,
                'query_times': query_times,
                'user_simulation_times': user_simulation_times,
                'update_times': update_times,
                'evaluation_times': evaluation_times,
                'overall_iteration_times': overall_iteration_times,
                'labels_in_pool': labels_in_pool
            }

            print(summary)

        split += 1


def evaluate_gridsearch_params(X_d, y_d, n_c, city, query_strategy, multi_label=False):
    clf = LogisticRegression()
    param_grid = {'C': [10, 100, 1000], 'class_weight': ['balanced'], 'penalty': ['l1', 'l2'], 'solver': ['saga']}
    gridsearch_clf = GridSearchCV(clf, param_grid, refit=True, verbose=1, n_jobs=-1)

    print(city)
    print()
    AlCV(X_in=X_d, y_in=y_d, qs=query_strategy, num_classes=n_c, clf_model=gridsearch_clf,
         split_nr=1, multi_label=multi_label)


def evaluate_fix_params(X_d, y_d, n_c, city, query_strategy, multi_label=False):
    if multi_label:
        c = 100
    else:
        c = 10
    clf = LogisticRegression(C=c, class_weight='balanced', penalty='l2', solver='saga')

    print(city)
    print()
    AlCV(X_in=X_d, y_in=y_d, qs=query_strategy, num_classes=n_c, clf_model=clf,
         split_nr=1, multi_label=multi_label)


if __name__ == '__main__':
    n_classes = 8
    df_raddialoge = pd.read_pickle("../../data/dataset-preprocessed.pkl")

    # get separate dataframes for each city
    df_b = df_raddialoge[df_raddialoge['dataset'] == 'B']
    df_e = df_raddialoge[df_raddialoge['dataset'] == 'E']
    df_m = df_raddialoge[df_raddialoge['dataset'] == 'M']

    # prepare datasets for use with scikit classifiers
    X_b, y_b, y_b_multi = prepare_data(df_b)
    X_e, y_e, y_e_multi = prepare_data(df_e)
    X_m, y_m, y_m_multi = prepare_data(df_m)

    # AL experiments: single class prediction, random sampling, logistic regression (fixed hyperparameters)
    # uncomment to run
    '''
    evaluate_fix_params(X_b, y_b, n_classes, 'Bonn', query_strategy=RandomSampling(), multi_label=False)
    evaluate_fix_params(X_e, y_e, n_classes, 'Ehrenfeld', query_strategy=RandomSampling(), multi_label=False)
    evaluate_fix_params(X_m, y_m, n_classes, 'Moers', query_strategy=RandomSampling(), multi_label=False)
    '''

    # AL experiments: multi class prediction, random sampling, logistic regression (fixed hyperparameters)
    # uncomment to run
    '''
    evaluate_fix_params(X_b, y_b_multi, n_classes, 'Bonn', query_strategy=RandomSampling(), multi_label=True)
    evaluate_fix_params(X_e, y_e_multi, n_classes, 'Ehrenfeld', query_strategy=RandomSampling(), multi_label=True)
    evaluate_fix_params(X_m, y_m_multi, n_classes, 'Moers', query_strategy=RandomSampling(), multi_label=True)
    '''

    # AL experiments: single class prediction, random sampling, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    evaluate_gridsearch_params(X_b, y_b, n_classes, 'Bonn', query_strategy=RandomSampling(), multi_label=False)
    evaluate_gridsearch_params(X_e, y_e, n_classes, 'Ehrenfeld', query_strategy=RandomSampling(), multi_label=False)
    evaluate_gridsearch_params(X_m, y_m, n_classes, 'Moers', query_strategy=RandomSampling(), multi_label=False)
    '''

    # AL experiments: multi class prediction, random sampling, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    evaluate_gridsearch_params(X_b, y_b_multi, n_classes, 'Bonn', query_strategy=RandomSampling(), multi_label=True)
    evaluate_gridsearch_params(X_e, y_e_multi, n_classes, 'Ehrenfeld', query_strategy=RandomSampling(), multi_label=True)
    evaluate_gridsearch_params(X_m, y_m_multi, n_classes, 'Moers', query_strategy=RandomSampling(), multi_label=True)
    '''

    # AL experiments: single class prediction, query by committee, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    evaluate_gridsearch_params(X_b, y_b, n_classes, 'Bonn', query_strategy=QBC(), multi_label=False)
    evaluate_gridsearch_params(X_e, y_e, n_classes, 'Ehrenfeld', query_strategy=QBC(), multi_label=False)
    evaluate_gridsearch_params(X_m, y_m, n_classes, 'Moers', query_strategy=QBC(), multi_label=False)
    '''

    # AL experiments: multi class prediction, query by committee, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    evaluate_gridsearch_params(X_b, y_b_multi, n_classes, 'Bonn', query_strategy=QBC(), multi_label=True)
    evaluate_gridsearch_params(X_e, y_e_multi, n_classes, 'Ehrenfeld', query_strategy=QBC(), multi_label=True)
    evaluate_gridsearch_params(X_m, y_m_multi, n_classes, 'Moers', query_strategy=QBC(), multi_label=True)
    '''

