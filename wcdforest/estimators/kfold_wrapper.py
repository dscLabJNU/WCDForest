# -*- coding:utf-8 -*-

import os, os.path as osp
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path
LOGGER = get_logger("wcdforest.estimators.kfold_wrapper")

class KFoldWrapper(object):
    """
    K-Fold Wrapper
    """
    def __init__(self, name, n_folds, est_class, est_args, random_state=None):
        """
        Parameters
        ----------
        n_folds (int): 
            Number of folds.
            If n_folds=1, means no K-Fold
        est_class (class):
            Class of estimator
        est_args (dict):
            Arguments of estimator
        random_state (int):
            random_state used for KFolds split and Estimator
        """
        self.name = name
        self.n_folds = n_folds
        self.est_class = est_class
        self.est_args = est_args
        self.random_state = random_state
        self.estimator1d = [None for k in range(self.n_folds)]

    def _init_estimator(self, k):
        est_args = self.est_args.copy()
        est_name = "{}/{}".format(self.name, k)
        est_args["random_state"] = self.random_state
        return self.est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify, cache_dir=None,eval_metrics=None, keep_model_in_mem=True):

        if cache_dir is not None:
            cache_dir = osp.join(cache_dir, name2path(self.name))

        eval_metrics = eval_metrics if eval_metrics is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(len(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
        # Fit
        y_probas = []
        n_dims = X.shape[-1]
        n_datas = X.size / n_dims
        inverse = False
        for k in range(self.n_folds):
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]

            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, n_dims)), y[train_idx].reshape(-1), cache_dir=cache_dir)

            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, n_dims)), cache_dir=cache_dir)

            # merging result
            if k == 0:
                y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=np.float32)
                y_probas.append(y_proba_cv)
            y_probas[0][val_idx, :] += y_proba
            if keep_model_in_mem:
                self.estimator1d[k] = est

        if inverse and self.n_folds > 1:
            y_probas /= (self.n_folds - 1)

        # log
        acc=self.log_eval_metrics(self.name, y, y_probas[0], eval_metrics, "train_cv")
        return y_probas, acc

    def log_eval_metrics(self, est_name, y_true, y_proba, eval_metrics, y_name):
        # accuracy=0
        if eval_metrics is None:
            return
        accuracy = eval_metrics[0][1](y_true, y_proba)
        LOGGER.info("Accuracy({}.{}.{})={:.2f}%".format(est_name, y_name, eval_metrics[0][0], accuracy * 100.))
        return accuracy

    def predict_proba(self, X_test):
        # K-Fold split
        n_dims = X_test.shape[-1]
        n_datas = X_test.size / n_dims
        for k in range(self.n_folds):
            est = self.estimator1d[k]
            y_proba = est.predict_proba(X_test.reshape((-1, n_dims)), cache_dir=None)
            if k == 0:
                y_proba_kfolds = y_proba
            else:
                y_proba_kfolds += y_proba
        y_proba_kfolds /= self.n_folds
        return y_proba_kfolds
