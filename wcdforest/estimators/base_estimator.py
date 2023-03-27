# -*- coding:utf-8 -*-

import os, os.path as osp
import numpy as np

from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path

LOGGER = get_logger("wcdforest.estimators.base_estimator")

def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

class BaseClassifierWrapper(object):
    def __init__(self, name, est_class, est_args):
        """
        name: str)
            Used for debug and as the filename this model may be saved in the disk
        """
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        """
        You can re-implement this function when inherient this class
        """
        est = self.est_class(**self.est_args)
        return est

    def fit(self, X, y, cache_dir=None):
        """
        cache_dir(str):
            if not None
                then if there is something in cache_dir, dont have fit the thing all over again
                otherwise, fit it and save to model cache
        """
        LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if self._is_cache_exists(cache_path):
            LOGGER.info("Find estimator from {} . skip process".format(cache_path))
            return
        est = self._init_estimator()
        self._fit(est, X, y)
        if cache_path is not None:
            # saved in disk
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est

        y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and osp.exists(cache_path)

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _fit(self, est, X, y):
        est.fit(X, y)

    def _predict_proba(self, est, X):
        return est.predict_proba(X)
