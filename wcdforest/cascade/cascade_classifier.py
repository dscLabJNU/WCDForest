import numpy as np
import os
import os.path as osp
import pickle
from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb

LOGGER = get_logger('wcdforest.cascade.cascade_classifier')


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class CascadeClassifier(object):


    def __init__(self, ca_config):

        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.estimator2d_w = {}
        self.estimator_acc = {}
        self.estimator_pre = {}

        self.estimator_test_pre = {}
        self.opt_layer_num = -1

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)


    def _set_estimator_w(self, li, ei, acc):
        if li not in self.estimator2d_w:
            self.estimator2d_w[li] = {}
        self.estimator2d_w[li][ei] = acc

    def _get_estimator_w(self, li, ei):
        return self.estimator2d_w.get(li, {}).get(ei, None)

    def _set_acc(self, li, acc):
        self.estimator_acc[li] = acc

    def _get_acc(self, li):
        return self.estimator_acc.get(li, None)

    def _set_pre(self, li, pro):
        self.estimator_pre[li] = pro

    def _get_pre(self, li):
        return self.estimator_pre.get(li, None)

    def _set_test_pre(self, li, pro):
        self.estimator_test_pre[li] = pro

    def _get_test_pre(self, li):
        return self.estimator_test_pre.get(li, None)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
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

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = list(range(n_groups))
        else:
            look_indexs_cycle = self.look_indexs_cycle
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1])
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X

        
    def fit_transform(self, X_groups_train, y_train, train_config=None):
        

        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        LOGGER.info("X_groups_train.shape={},y_train.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)

        # check groups dimension
        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)

        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("group_starts={}".format(group_starts))
        LOGGER.info("group_ends={}".format(group_ends))

        n_trains = X_groups_train[0].shape[0]

        n_classes = self.n_classes
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []

        opt_datas = [None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            X_cur_train = None
            layer_id = 0
            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                else:
                    X_cur_train = X_proba_train.copy()
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]

                X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[look_indexs]:group_ends[look_indexs]]))

                X_cur_pre=np.zeros((n_trains, n_classes), dtype=np.float32)
                
                #
                for i in range(0, layer_id):
                    X_cur_pre = X_cur_pre + self._get_pre(i) * (self._get_acc(i)/sum(self.estimator_acc.values()))


                if layer_id>0:
                    X_cur_pre=X_cur_pre/layer_id
                    X_cur_train = np.hstack((X_cur_train, X_cur_pre))

                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((n_trains, n_classes))

                y_probas_y=[]
                y_probas_acc=[]

                for ei, est_config in enumerate(self.est_configs):
                    est = self._init_estimators(layer_id, ei)
                    # fit_trainsform
                    y_probas, acc = est.fit_transform(X_cur_train, y_train, y_train,eval_metrics=self.eval_metrics,
                            keep_model_in_mem=train_config.keep_model_in_mem)
                    y_probas_y.append(y_probas[0])
                    y_probas_acc.append(acc)
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)

                for i in range(0,len(y_probas_acc)):
                    acc_w = y_probas_acc[i]/sum(y_probas_acc)
                    if train_config.keep_model_in_mem:
                        self._set_estimator_w(layer_id, i, acc_w)

                for i in range(0, len(y_probas_y)):
                    X_proba_train[:, i * n_classes: i * n_classes + n_classes] = y_probas_y[i]*self._get_estimator_w(layer_id, i)
                    y_train_proba_li += y_probas_y[i]*self._get_estimator_w(layer_id, i)

                y_train_proba_li /= len(self.est_configs)
                self._set_pre(layer_id, y_train_proba_li)
                train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                self._set_acc(layer_id, train_avg_acc)

                train_acc_list.append(train_avg_acc)
                opt_layer_id = get_opt_layer_id(train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id]))
                    if data_save_dir is not None:
                        self.save_data(data_save_dir, opt_layer_id, *opt_datas)
                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    return opt_layer_id, opt_datas[0], opt_datas[1]
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                layer_id += 1

            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id]))
            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            return self.max_layers, opt_datas[0], opt_datas[1]
        except KeyboardInterrupt:
            pass

    

    def transform(self, X_groups_test):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_test, False)
        # check group_dims
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes

        # probability of each cascades's estimators
        X_proba_test = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        X_cur_test = None


        for layer_id in range(self.opt_layer_num):
            # Copy previous cascades's probability into current X_cur
            if layer_id == 0:
                # first layer not have probability distribution
                X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
            else:
                X_cur_test = X_proba_test.copy()
            # Stack data that current layer needs in to X_cur
            look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]

            X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[look_indexs]:group_ends[look_indexs]]))

            y_train_proba_li = np.zeros((n_tests, n_classes))

            X_cur_pre = np.zeros((n_tests, n_classes), dtype=np.float32)

            for i in range(0, layer_id):
                X_cur_pre += self._get_test_pre(i)

            if layer_id>0:
                X_cur_pre=X_cur_pre/layer_id
                X_cur_test = np.hstack((X_cur_test, X_cur_pre))
                

            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(
                layer_id, look_indexs, X_cur_test.shape))
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                y_probas = est.predict_proba(X_cur_test)
                X_proba_test[:, ei * n_classes:ei * n_classes + n_classes] = y_probas*self._get_estimator_w(layer_id, ei)
                y_train_proba_li += y_probas * self._get_estimator_w(layer_id, ei)

            y_train_proba_li /= len(self.est_configs)

            self._set_test_pre(layer_id, y_train_proba_li)

        return X_proba_test

    def predict_proba(self, X):
        # n x (n_est*n_classes)
        y_proba = self.transform(X)
        # n x n_est x n_classes
        y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        return y_proba

    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
