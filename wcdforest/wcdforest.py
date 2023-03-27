import numpy as np
from .cascade.cascade_classifier import CascadeClassifier
from .config import GCTrainConfig
from .utils.log_utils import get_logger
from .fgnet import FGNet


LOGGER = get_logger("wcdforest.wcdforest")

class WCDForest(object):

    
    def __init__(self, config):
        self.config = config
        self.train_config = GCTrainConfig(config.get("train", {}))
        if "net" in self.config:
            self.fg = FGNet(self.config["net"])
        else:
            self.fg = None
        if "cascade" in self.config:
            self.ca = CascadeClassifier(self.config["cascade"])
        else:
            self.ca = None

    def fit_transform(self, X_train, y_train, X_test=None, y_test=None, train_config=None):
        X_train=np.hstack((X_train,X_train))
        train_config = train_config or self.train_config
        if X_test is None or y_test is None:
            if "test" in train_config.phases:
                train_config.phases.remove("test")
        if self.fg is not None:
            X_train= self.fg.fit_transform(X_train, y_train, train_config=train_config)
        if self.ca is not None:
            _, X_train, _ = self.ca.fit_transform(X_train, y_train, train_config=train_config)
        return X_train

    def predict_proba(self, X):
        if self.fg is not None:
            X = self.fg.transform(X)
        y_proba = self.ca.predict_proba(X)
        return y_proba

    def predict(self, X):
        X = np.hstack((X,X))
        y_proba = self.predict_proba(X)
        # y_pred = np.argmax(y_proba, axis=1)
        return y_proba