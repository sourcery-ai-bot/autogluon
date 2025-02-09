import copy
import logging

import numpy as np
from pandas import Series

from ..ml.constants import BINARY, MULTICLASS, REGRESSION

logger = logging.getLogger(__name__)


# LabelCleaner cleans labels prior to entering feature generation
class LabelCleaner:
    num_classes = None
    inv_map = None
    ordered_class_labels = None
    ordered_class_labels_transformed = None

    @staticmethod
    def construct(problem_type: str, y: Series, y_uncleaned: Series = None):
        if problem_type == BINARY:
            return LabelCleanerBinary(y)
        elif problem_type == MULTICLASS:
            if y_uncleaned is None:
                y_uncleaned = copy.deepcopy(y)
            if len(y.unique()) == 2:
                return LabelCleanerMulticlassToBinary(y, y_uncleaned)
            else:
                return LabelCleanerMulticlass(y, y_uncleaned)
        elif problem_type == REGRESSION:
            return LabelCleanerDummy()
        else:
            raise NotImplementedError

    def transform(self, y: Series) -> Series:
        raise NotImplementedError

    def inverse_transform(self, y: Series) -> Series:
        raise NotImplementedError

    def transform_proba(self, y):
        return y

    def inverse_transform_proba(self, y):
        return y


class LabelCleanerMulticlass(LabelCleaner):
    def __init__(self, y: Series, y_uncleaned: Series):
        self.cat_mappings_dependent_var: dict = self._generate_categorical_mapping(y)
        self.inv_map: dict = {v: k for k, v in self.cat_mappings_dependent_var.items()}

        self.cat_mappings_dependent_var_uncleaned: dict = self._generate_categorical_mapping(y_uncleaned)
        self.inv_map_uncleaned: dict = {v: k for k, v in self.cat_mappings_dependent_var_uncleaned.items()}

        self.num_classes = len(self.cat_mappings_dependent_var.keys())
        self.ordered_class_labels = list(y_uncleaned.astype('category').cat.categories)
        self.valid_ordered_class_labels = list(y.astype('category').cat.categories)
        self.ordered_class_labels_transformed = list(range(len(self.valid_ordered_class_labels)))
        self.invalid_class_count = len(self.ordered_class_labels) - len(self.valid_ordered_class_labels)
        self.labels_to_zero_fill = [1 if label not in self.valid_ordered_class_labels else 0 for label in self.ordered_class_labels]
        self.label_index_to_keep = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 0]
        self.label_index_to_remove = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 1]

    def transform(self, y: Series) -> Series:
        if isinstance(y, np.ndarray):
            y = Series(y)
        y = y.map(self.inv_map)
        return y

    def inverse_transform(self, y: Series) -> Series:
        if isinstance(y, list):
            y = Series(y)
        y = y.map(self.cat_mappings_dependent_var)
        return y

    # TODO: Unused?
    def transform_proba(self, y):
        if self.invalid_class_count > 0:
            # this assumes y has only 0's for any columns it is about to remove, if it does not, weird things may start to happen since rows will not sum to 1
            return np.delete(y, self.label_index_to_remove, axis=1)
        else:
            return y

    def inverse_transform_proba(self, y):
        if self.invalid_class_count <= 0:
            return y
        y_transformed = np.zeros([len(y), len(self.ordered_class_labels)])
        y_transformed[:, self.label_index_to_keep] = y
        return y_transformed

    @staticmethod
    def _generate_categorical_mapping(y: Series) -> dict:
        categories = y.astype('category')
        return dict(enumerate(categories.cat.categories))


# TODO: Expand print statement to multiclass as well
class LabelCleanerBinary(LabelCleaner):
    def __init__(self, y: Series):
        self.num_classes = 2
        self.unique_values = list(y.unique())
        if len(self.unique_values) != 2:
            raise AssertionError('y does not contain exactly 2 unique values:', self.unique_values)
        # TODO: Clean this code, for loop
        if (1 in self.unique_values) and (2 in self.unique_values):
            self.inv_map: dict = {1: 0, 2: 1}
        elif ('1' in self.unique_values) and ('2' in self.unique_values):
            self.inv_map: dict = {'1': 0, '2': 1}
        elif ((str(False) in [str(val) for val in self.unique_values]) and
              (str(True) in [str(val) for val in self.unique_values])):
            false_val = [val for val in self.unique_values if str(val) == str(False)][0]  # may be str or bool
            true_val = [val for val in self.unique_values if str(val) == str(True)][0]  # may be str or bool
            self.inv_map: dict = {false_val: 0, true_val: 1}
        elif (0 in self.unique_values) and (1 in self.unique_values):
            self.inv_map: dict = {0: 0, 1: 1}
        elif ('0' in self.unique_values) and ('1' in self.unique_values):
            self.inv_map: dict = {'0': 0, '1': 1}
        elif ('No' in self.unique_values) and ('Yes' in self.unique_values):
            self.inv_map: dict = {'No': 0, 'Yes': 1}
        elif ('N' in self.unique_values) and ('Y' in self.unique_values):
            self.inv_map: dict = {'N': 0, 'Y': 1}
        elif ('n' in self.unique_values) and ('y' in self.unique_values):
            self.inv_map: dict = {'n': 0, 'y': 1}
        elif ('F' in self.unique_values) and ('T' in self.unique_values):
            self.inv_map: dict = {'F': 0, 'T': 1}
        elif ('f' in self.unique_values) and ('t' in self.unique_values):
            self.inv_map: dict = {'f': 0, 't': 1}
        else:
            self.inv_map: dict = {self.unique_values[0]: 0, self.unique_values[1]: 1}
            logger.log(15, 'Note: For your binary classification, AutoGluon arbitrarily selects which label-value represents positive vs negative class')
        poslabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 1][0]
        neglabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 0][0]
        logger.log(
            20,
            f'Selected class <--> label mapping:  class 1 = {poslabel}, class 0 = {neglabel}',
        )

        self.cat_mappings_dependent_var: dict = {v: k for k, v in self.inv_map.items()}
        self.ordered_class_labels_transformed = [0, 1]
        self.ordered_class_labels = [self.cat_mappings_dependent_var[label_transformed] for label_transformed in self.ordered_class_labels_transformed]

    def transform(self, y: Series) -> Series:
        if isinstance(y, np.ndarray):
            y = Series(y)
        y = y.map(self.inv_map)
        return y

    def inverse_transform(self, y: Series) -> Series:
        if isinstance(y, list):
            y = Series(y)
        return y.map(self.cat_mappings_dependent_var)


class LabelCleanerMulticlassToBinary(LabelCleanerMulticlass):
    def __init__(self, y: Series, y_uncleaned: Series):
        super().__init__(y=y, y_uncleaned=y_uncleaned)
        self.label_cleaner_binary = LabelCleanerBinary(y=y.map(self.inv_map))

    def transform(self, y: Series) -> Series:
        y = super().transform(y)
        y = self.label_cleaner_binary.transform(y)
        return y

    def inverse_transform_proba(self, y):
        y = self.convert_binary_proba_to_multiclass_proba(y=y)
        return super().inverse_transform_proba(y)

    @staticmethod
    def convert_binary_proba_to_multiclass_proba(y):
        y_transformed = np.zeros([len(y), 2])
        y_transformed[:, 0] = 1 - y
        y_transformed[:, 1] = y
        return y_transformed


class LabelCleanerDummy(LabelCleaner):
    def transform(self, y: Series) -> Series:
        if isinstance(y, np.ndarray):
            y = Series(y)
        return y

    def inverse_transform(self, y: Series) -> Series:
        return y
