from .sklearn_estimators import GCSGDClassifier,GCLR, GCExtraTreesClassifier, GCRandomForestClassifier, GCXGBClassifier
from .kfold_wrapper import KFoldWrapper

def get_estimator_class(est_type):
    if est_type == "ExtraTreesClassifier":
        return GCExtraTreesClassifier
    if est_type == "RandomForestClassifier":
        return GCRandomForestClassifier
    if est_type == "LogisticRegression":
        return GCLR
    if est_type == "SGDClassifier":
        return GCSGDClassifier
    if est_type == "XGBClassifier":
        return GCXGBClassifier
    raise ValueError('Unkown Estimator Type, est_type={}'.format(est_type))

def get_estimator(name, est_type, est_args):
    est_class = get_estimator_class(est_type)
    return est_class(name, est_args)

def get_estimator_kfold(name, n_splits, est_type, est_args, random_state=None):
    est_class = get_estimator_class(est_type)
    return KFoldWrapper(name, n_splits, est_class, est_args, random_state=random_state)
