from sklearn.grid_search import RandomizedSearchCV, GridSearchCV


def random_tuner(X, y, pipline, hyper_params, n_iter_search,
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
    model = RandomizedSearchCV(pipeline, parameters, n_iter=n_iter_search,
                               n_jobs=-1, verbose=1, scoring=scoring_metric)

    start_t = time.time()
    print('\nTraining multi-output model...')

    model.fit(train_x, train_y)

    del_t = time.time() - start_t
    print('Time taken (secs): ', del_t)

    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return model
