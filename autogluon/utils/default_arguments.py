from typing import Set, Tuple, Dict
import logging
import numbers

logger = logging.getLogger(__name__)


class CheckType(object):
    def assert_valid(self, key: str, value):
        pass


class Float(CheckType):
    def __init__(self, lower: float = None, upper: float = None):
        if lower and upper:
            assert lower < upper
        self.lower = lower
        self.upper = upper

    def assert_valid(self, key: str, value):
        assert isinstance(
            value, numbers.Real
        ), f"{key}: Value = {value} must be of type float"

        assert (
            not self.lower
        ) or value >= self.lower, f"{key}: Value = {value} must be >= {self.lower}"

        assert (
            not self.upper
        ) or value <= self.upper, f"{key}: Value = {value} must be <= {self.upper}"


class Integer(CheckType):
    def __init__(self, lower: int = None, upper: int = None):
        if lower and upper:
            assert lower < upper
        self.lower = lower
        self.upper = upper

    def assert_valid(self, key: str, value):
        assert isinstance(
            value, numbers.Integral
        ), f"{key}: Value = {value} must be of type int"

        assert (
            not self.lower
        ) or value >= self.lower, f"{key}: Value = {value} must be >= {self.lower}"

        assert (
            not self.upper
        ) or value <= self.upper, f"{key}: Value = {value} must be <= {self.upper}"


class Categorical(CheckType):
    def __init__(self, choices: Tuple[str, ...]):
        self.choices = set(choices)

    def assert_valid(self, key: str, value):
        assert (
            isinstance(value, str) and value in self.choices
        ), f"{key}: Value = {value} must be in {self.choices}"


class String(CheckType):
    def assert_valid(self, key: str, value):
        assert isinstance(value, str), "{}: Value = {} must be of type str"


class Boolean(CheckType):
    def assert_valid(self, key: str, value):
        assert isinstance(value, bool), f"{key}: Value = {value} must be boolean"


def check_and_merge_defaults(
        options: dict, mandatory: Set[str], default_options: dict,
        constraints: Dict[str, CheckType] = None, dict_name=None) -> dict:
    """
    First, check that all keys in mandatory appear in options. Second, create
    result_options by merging options and default_options, where entries in
    options have precedence. Finally, if constraints is given, this is used to
    check validity of values.

    :param options:
    :param mandatory:
    :param default_options:
    :param constraints:
    :param dict_name:
    :return: result_options
    """
    prefix = "" if dict_name is None else f"{dict_name}: "
    for key in mandatory:
        assert key in options, f"{prefix}Key '{key}' is missing (but is mandatory)"
    log_msg = ""
    result_options = {
        k: v for k, v in options.items() if v is not None}
    for key, value in default_options.items():
        if key not in result_options:
            log_msg += f"{prefix}Key '{key}': Imputing default value {value}\n"
            result_options[key] = value
    if log_msg:
        logger.info(log_msg)
    # Check constraints
    if constraints:
        for key, value in result_options.items():
            if check := constraints.get(key):
                check.assert_valid(f"{prefix}Key '{key}'", value)

    return result_options


def filter_by_key(options: dict, remove_keys: Set[str]) -> dict:
    """
    Filter options by removing entries whose keys are in remove_keys.
    Used to filter kwargs passed to a constructor, before passing it to
    the superclass constructor.

    :param options:
    :param remove_keys:
    :return: Filtered options
    """
    return {
        k: v for k, v in options.items() if k not in remove_keys}


def assert_no_invalid_options(options: dict, all_keys: Set[str], name: str):
    for k in options:
        assert k in all_keys, f"{name}: Invalid argument '{k}'"
