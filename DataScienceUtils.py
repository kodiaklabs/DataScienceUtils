from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import time


def random_tuner(X, y, pipeline, hyper_params, n_iter_search,
                 scoring_metric):
    """
    This is a quick function to explore different base classifiers.
    X: Input data to predict off
    y: target vector or array (for multilabel problems)
    pipline: pipeline which preprocesses and classifies. Could be as simple as
        a classifier, or a longer series of transformations.
    hyper_params: hyperparameters for both the pipeline and the classifier.
    n_iter_search is the number of parameter sets to explore.
    scoring_metric: the metric for which to score pipeline.

    Output:
        Prints the total time taken, best scorebest hyperparameters

    Example Usage

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression()),
    ])

    hyper_params = {
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': [None, 'balanced'],
        'clf__C': [1e-3, 1e-2, 1e-1, 1.0, 10.]
    }
    n_iter = 20
    scoring_metric = roc_auc
    log_model = random_tuner(X, y, pipeline, hyper_params, n_iter, roc_auc)
    """
    model = RandomizedSearchCV(pipeline, hyper_params, n_iter=n_iter_search,
                               n_jobs=-1, verbose=1, scoring=scoring_metric)

    start_t = time.time()

    model.fit(X, y)

    del_t = time.time() - start_t
    print('Time taken (secs): ', del_t)

    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return model


def grid_tuner(X, y, pipeline, hyper_params, scoring_metric):
    """
    Grid search version of random_tuner. For parameter description, see
    documentation of random_tuner.

    NB: It is recommended that one uses random search of the hyper parameter
    space, as that is emperically more efficient [1].

    [1] http://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    """
    model = GridSearchCV(pipeline, hyper_params, n_jobs=-1, verbose=1,
                         scoring=scoring_metric)

    start_t = time.time()
    print('\nTraining multi-output model...')

    model.fit(X, y)

    del_t = time.time() - start_t
    print('Time taken (secs): ', del_t)

    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return model


# Classifiers
def logreg_hype():
#     'clf__penalty': ('l2', 'l1'),
#     'clf__class_weight': (None, 'balanced'),
#     'clf__C': (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
#     'clf__warm_start': (True, False),
#     'clf__fit_intercept': (True, False)
    hyper_params = {'penalty': ['l1', 'l2'],
                    'class_weight': [None, 'balanced'],
                    'C': [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0
                    }
    return hyper_params


def sgd_hype():
    hyper_params = {'alpha': [1e-2, 1e-3, 1e-4, 1e-5, 1e-7],
                    'loss': ['hinge', 'log', 'modified_huber',
                             'squared_hinge', 'perceptron'],
                    'penalty': ['none', 'l2', 'l1', 'elasticnet']}
    return hyper_params


def svc_hype():
    hyper_params = {"C": [1e-2, 1e-1, 1.0, 5, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    "gamma": ['auto', 1e-3, 1e-1, 1],
                    "class_weight": [None, 'balanced']}
    return hyper_params


def randforest_hype():
    hyper_params = {"max_depth": [2, 5, 10, None],
                    "n_estimators": [10, 20, 100, 150],
                    'max_features': [None, 'auto', 'sqrt', 'log2'],
                    'class_weight': [None, 'balanced']}
    return hyper_params
