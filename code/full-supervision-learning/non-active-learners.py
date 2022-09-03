import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import IterativeStratification
import pickle


# dummy function for tokenization and preprocessing
def dummy_fun(doc):
    return doc


def evaluation(clf, param_grid, X, y, filename, type):
    # train-test splits
    if type == "multi":
        skf = IterativeStratification(n_splits=5, order=1)  # (different stratification for multi-label case)
    if type == "single":
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)

    # scores
    accuracy = []
    f1_micro = []
    f1_macro = []
    f1_weighted = []

    best_settings = []
    results = []

    split = 1
    for train_index, test_index in skf.split(X, y):
        print(split)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # tune parameters with grid search
        grid = GridSearchCV(clf, param_grid, refit=True, verbose=0)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        # save the model to disk
        f = filename + str(split) + '.sav'
        pickle.dump(grid, open("../../results/"+f, 'wb'))

        # print best parameter after tuning
        print("Best settings:", grid.best_params_)
        best_settings.append(grid.best_params_)
        print(best_settings)

        # print classification report
        print(classification_report(y_test, y_pred))
        results.append(classification_report(y_test, y_pred, output_dict=True))
        print(results)

        accuracy.append(accuracy_score(y_test, y_pred))
        f1_micro.append(f1_score(y_test, y_pred, average='micro'))
        f1_macro.append(f1_score(y_test, y_pred, average='macro'))
        f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))

        split += 1

    print()
    print()
    print("Accuracy (avg, std)", np.average(accuracy), np.std(accuracy))
    print("F1 micro (avg, std)", np.average(f1_micro), np.std(f1_micro))
    print("F1 macro (avg, std)", np.average(f1_macro), np.std(f1_macro))
    print("F1 weighted (avg, std)", np.average(f1_weighted), np.std(f1_weighted))
    print()
    print()
    print(best_settings)
    print(results)
    print()
    print()


def prepare_multi(df_b, df_e, df_m):
    # tf-idf vectorize feature vectors
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,
                                 preprocessor=dummy_fun, token_pattern=None, max_df=0.8, min_df=2)

    X_b, X_e, X_m = vectorizer.fit_transform(df_b['preprocessed_text']), vectorizer.fit_transform(
        df_e['preprocessed_text']), vectorizer.fit_transform(df_m['preprocessed_text'])

    multi_cols = ['level1_traffic_lights', 'level1_lighting', 'level1_signage', 'level1_bicycle_parking',
                  'level1_obstacles', 'level1_cycling_traffic_management', 'level1_cycle_path_quality',
                  'level1_misc']

    y_b, y_e, y_m = df_b[multi_cols].to_numpy(), df_e[multi_cols].to_numpy(), df_m[multi_cols].to_numpy()

    return X_b, X_e, X_m, y_b, y_e, y_m


def prepare_single(df_b, df_e, df_m):
    # tf-idf vectorize feature vectors
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,
                                 preprocessor=dummy_fun, token_pattern=None, max_df=0.8, min_df=2)
    le = preprocessing.LabelEncoder()

    X_b, y_b = vectorizer.fit_transform(df_b['preprocessed_text']), le.fit_transform(df_b['main_category_level1'])
    print("vocabulary size B:", len(vectorizer.vocabulary_))
    X_e, y_e = vectorizer.fit_transform(df_e['preprocessed_text']), le.fit_transform(df_e['main_category_level1'])
    print("vocabulary size E:", len(vectorizer.vocabulary_))
    X_m, y_m = vectorizer.fit_transform(df_m['preprocessed_text']), le.fit_transform(df_m['main_category_level1'])
    print("vocabulary size M:", len(vectorizer.vocabulary_))

    return X_b, X_e, X_m, y_b, y_e, y_m


if __name__ == '__main__':
    df_raddialoge = pd.read_pickle("../../data/dataset-preprocessed.pkl")

    # get separated dataframes for each city
    df_b = df_raddialoge[df_raddialoge['dataset'] == 'B']
    df_e = df_raddialoge[df_raddialoge['dataset'] == 'E']
    df_m = df_raddialoge[df_raddialoge['dataset'] == 'M']

    X_b, X_e, X_m, y_b, y_e, y_m = prepare_single(df_b, df_e, df_m)

    # Full supervision baseline: single class prediction, support vector machine (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {'C': [0.1, 1, 10, 100], 'class_weight': ['balanced'], 'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['linear', 'rbf']}
    clf = SVC()

    evaluation(clf, param_grid, X_b, y_b, "SVM-Single-Bonn_split", "single")
    evaluation(clf, param_grid, X_e, y_e, "SVM-Single-Ehrenfeld_split", "single")
    evaluation(clf, param_grid, X_m, y_m, "SVM-Single-Moers_split", "single")
    '''

    # Full supervision baseline: single class prediction, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {'C': [10, 100, 1000], 'class_weight': ['balanced'], 'penalty': ['l1', 'l2'], 'solver': ['saga']}
    clf = LogisticRegression()

    evaluation(clf, param_grid, X_b, y_b, "LR-Single-Bonn_split", "single")
    evaluation(clf, param_grid, X_e, y_e, "LR-Single-Ehrenfeld_split", "single")
    evaluation(clf, param_grid, X_m, y_m, "LR-Single-Moers_split", "single")
    '''

    # Full supervision baseline: single class prediction, naive bayes (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {}
    clf = BernoulliNB()

    evaluation(clf, param_grid, X_b, y_b, "NB-Single-Bonn_split", "single")
    evaluation(clf, param_grid, X_e, y_e, "NB-Single-Ehrenfeld_split", "single")
    evaluation(clf, param_grid, X_m, y_m, "NB-Single-Moers_split", "single")
    '''

    # Full supervision baseline: single class prediction, ensemble method (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {'svm__C': [1, 10, 100],
                  'svm__gamma': [1, 0.1],
                  'svm__kernel': ['rbf', 'linear'],
                  'svm__class_weight': ['balanced'],
                  'lr__penalty': ['l1', 'l2'],
                  'lr__C': [10, 100, 1000],
                  'lr__solver': ['saga'],
                  'lr__class_weight': ['balanced']}
    eclf = VotingClassifier(estimators=[
        ('svm', SVC(probability=True)),
        ('nb', BernoulliNB()),
        ('lr', LogisticRegression())],
        voting='hard')

    evaluation(eclf, param_grid, X_b, y_b, "Ensemble-Single-Bonn_split", "single")
    evaluation(eclf, param_grid, X_e, y_e, "Ensemble-Single-Ehrenfeld_split", "single")
    evaluation(eclf, param_grid, X_m, y_m, "Ensemble-Single-Moers_split", "single")
    '''
    
    X_b, X_e, X_m, y_b, y_e, y_m = prepare_multi(df_b, df_e, df_m)

    # Full supervision baseline: multi class prediction, support vector machine (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {'estimator__C': [0.1, 1, 10, 100], 'estimator__class_weight': ['balanced'],
                  'estimator__gamma': [1, 0.1, 0.01, 0.001], 'estimator__kernel': ['linear', 'rbf']}
    clf = SVC()
    multi_target_clf = MultiOutputClassifier(clf, n_jobs=-1)

    evaluation(multi_target_clf, param_grid, X_b, y_b, "SVM-Multi-Bonn_split", "multi")
    evaluation(multi_target_clf, param_grid, X_e, y_e, "SVM-Multi-Ehrenfeld_split", "multi")
    evaluation(multi_target_clf, param_grid, X_m, y_m, "SVM-Multi-Moers_split", "multi")
    '''

    # Full supervision baseline: multi class prediction, logistic regression (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {'estimator__C': [10, 100, 1000], 'estimator__class_weight': ['balanced'],
                  'estimator__penalty': ['l1', 'l2'], 'estimator__solver': ['saga']}
    clf = LogisticRegression()
    multi_target_clf = MultiOutputClassifier(clf, n_jobs=-1)

    evaluation(multi_target_clf, param_grid, X_b, y_b, "LR-Multi-Bonn_split", "multi")
    evaluation(multi_target_clf, param_grid, X_e, y_e, "LR-Multi-Ehrenfeld_split", "multi")
    evaluation(multi_target_clf, param_grid, X_m, y_m, "LR-Multi-Moers_split", "multi")
    '''

    # Full supervision baseline: multi class prediction, naive bayes (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    param_grid = {}
    clf = BernoulliNB()
    multi_target_clf = MultiOutputClassifier(clf, n_jobs=-1)

    evaluation(multi_target_clf, param_grid, X_b, y_b, "NB-Multi-Bonn_split", "multi")
    evaluation(multi_target_clf, param_grid, X_e, y_e, "NB-Multi-Ehrenfeld_split", "multi")
    evaluation(multi_target_clf, param_grid, X_m, y_m, "NB-Multi-Moers_split", "multi")
    '''

    # Full supervision baseline: mutli class prediction, ensemble method (hyperparamaters with gridsearch)
    # uncomment to run
    '''
    eclf = VotingClassifier(estimators=[
        ('svm', SVC(probability=True)),
        ('nb', BernoulliNB()),
        ('lr', LogisticRegression())],
        voting='hard')
    param_grid = {'estimator__svm__C': [1, 10, 100],
                  'estimator__svm__gamma': [1, 0.1],
                  'estimator__svm__kernel': ['rbf', 'linear'],
                  'estimator__svm__class_weight': ['balanced'],
                  'estimator__lr__penalty': ['l1', 'l2'],
                  'estimator__lr__C': [10, 100, 1000],
                  'estimator__lr__solver': ['saga'],
                  'estimator__lr__class_weight': ['balanced']}
    multi_target_clf = MultiOutputClassifier(eclf, n_jobs=-1)

    evaluation(multi_target_clf, param_grid, X_b, y_b, "Ensemble-Multi-Bonn_split", "multi")
    evaluation(multi_target_clf, param_grid, X_e, y_e, "Ensemble-Multi-Ehrenfeld_split", "multi")
    evaluation(multi_target_clf, param_grid, X_m, y_m, "Ensemble-Multi-Moers_split", "mutli")
    '''
