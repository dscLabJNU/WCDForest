import numpy as np
import os, os.path as osp
from .estimators import get_estimator_kfold
from .utils.config_utils import get_config_value
from .utils.metrics import accuracy_pb
from .utils.log_utils import get_logger

LOGGER = get_logger("wcdforest.gcnet")


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc

class FGNet(object):


    def __init__(self, ca_config):

        self.ca_config = ca_config

        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.windows = self.get_value("windows", None, list)
        self.random_state = self.get_value("random_state", None, int)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.win_each = []
        self.forest_each=[]

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="net")

    def _set_estimator(self, li, est):
        self.estimator2d[li] = est

    def _get_estimator(self, li):
        return self.estimator2d.get(li, None)

    def _init_estimators(self, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "estimator_{} - {}_folds".format(ei, est_args["n_folds"])
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)


    def _window_slicing_sequence(self, X, window, y=None, stride=1):

        len_iter = int(X.shape[1]/2)
        iter_array = np.arange(0, stride*len_iter, stride)

        ind_1X = np.arange(np.prod((1,X.shape[1])))
        inds_to_take = [ind_1X[i:i+window] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        elif y is None:
            sliced_target = None
        return sliced_sqce, sliced_target

    def fit_transform(self, X_groups_train, y_train, train_config=None):

        if train_config is None:
            from .config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir
        windows = self.windows
        opt_datas = [None, None]
        n_est=0

        n_trains = X_groups_train.shape[0]

        try:
            X_proba_win=[]
            y_proba_win = []
            train_acc_list=[]
            y_true=[]

            for i in range(0,len(windows)):
                x, y = self._window_slicing_sequence(X_groups_train, windows[i], y_train)
                y_true.append(y)
                est_1 = self._init_estimators(n_est)
                y_probas_1 ,acc_1 = est_1.fit_transform(x, y, y,eval_metrics=self.eval_metrics,
                            keep_model_in_mem=train_config.keep_model_in_mem)

                if train_config.keep_model_in_mem:
                    self._set_estimator(n_est, est_1)

                n_est = n_est+1
                est_2 = self._init_estimators(n_est)
                y_probas_2, acc_2 = est_2.fit_transform(x, y, y, eval_metrics=self.eval_metrics,
                                                        keep_model_in_mem=train_config.keep_model_in_mem)
                w1=acc_1/(acc_1+acc_2)
                w2 = acc_2 / (acc_1 + acc_2)
                self.forest_each.append(w1)
                self.forest_each.append(w2)

                y_probas=y_probas_1[0]*w1+y_probas_2[0]*w2
                y_proba_win.append(y_probas)

                x_probas=np.hstack((y_probas_1[0] * w1, y_probas_2[0] * w2))
                X_proba_win.append(x_probas)

                if train_config.keep_model_in_mem:
                    self._set_estimator(n_est, est_2)

                n_est = n_est + 1

            for i in range(0,len(y_proba_win)):
                train_avg_acc = np.sum(np.std(y_proba_win[i], ddof=1, axis=1))
#                 train_avg_acc = calc_accuracy(y_true[i], np.argmax(y_proba_win[i], axis=1), 'forest{} - train.forest'.format(i))
                train_acc_list.append(train_avg_acc)

            train_acc_w=[]
            for i in range(0, len(train_acc_list)):
                train_acc_w.append(train_acc_list[i]/sum(train_acc_list))

            for i in range(0, len(train_acc_w)):
                X_proba_win[i]=(X_proba_win[i]*train_acc_w[i]).reshape([n_trains, -1])

            self.win_each=train_acc_w
#####
            x_probas=X_proba_win[0]

            for i in range(0,len(X_proba_win)-1):
                x_probas = np.hstack((x_probas, X_proba_win[i+1]))

            X_proba_win=x_probas
####
            return X_proba_win
        except KeyboardInterrupt:
            pass

    def transform(self, X_groups_test):
        n_trains = X_groups_test.shape[0]

        windows = self.windows
        forest_each=self.forest_each
        win_each=self.win_each
        y_proba_win=[]
        X_proba_win=[]
        n_est=0
        for i in range(0, len(windows)):
            x, y = self._window_slicing_sequence(X_groups_test, windows[i])
            est_1 = self._get_estimator(n_est)
            y_probas_1 = est_1.predict_proba(x)
            w1=forest_each[n_est]

            n_est = n_est + 1
            est_2 = self._get_estimator(n_est)
            y_probas_2 = est_2.predict_proba(x)
            w2 = forest_each[n_est]

            y_probas = y_probas_1* w1 + y_probas_2* w2
            y_proba_win.append(y_probas)

            x_probas = np.hstack((y_probas_1 * w1, y_probas_2* w2))
            X_proba_win.append(x_probas)
            n_est = n_est + 1

        for i in range(0, len(win_each)):
            X_proba_win[i] = (X_proba_win[i] * win_each[i]).reshape([n_trains, -1])
####
        x_probas = X_proba_win[0]

        for i in range(0, len(X_proba_win) - 1):
            x_probas = np.hstack((x_probas, X_proba_win[i + 1]))

        X_proba_win = x_probas
####
        return X_proba_win