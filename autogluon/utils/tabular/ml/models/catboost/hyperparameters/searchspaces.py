""" Default hyperparameter search spaces used in CatBoost Boosting model """
from ....constants import BINARY, MULTICLASS, REGRESSION
from .......core import Real, Int


def get_default_searchspace(problem_type, num_classes=None):
    if problem_type == BINARY or problem_type not in [MULTICLASS, REGRESSION]:
        return get_searchspace_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass_baseline(num_classes=num_classes)
    else:
        return get_searchspace_regression_baseline()


def get_searchspace_multiclass_baseline(num_classes):
    return {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.1, log=True),
        'depth': Int(lower=5, upper=8, default=6),
        'l2_leaf_reg': Real(lower=1, upper=5, default=3),
    }


def get_searchspace_binary_baseline():
    return {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.1, log=True),
        'depth': Int(lower=5, upper=8, default=6),
        'l2_leaf_reg': Real(lower=1, upper=5, default=3),
    }


def get_searchspace_regression_baseline():
    return {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.1, log=True),
        'depth': Int(lower=5, upper=8, default=6),
        'l2_leaf_reg': Real(lower=1, upper=5, default=3),
    }
