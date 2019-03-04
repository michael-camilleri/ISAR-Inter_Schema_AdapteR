"""
Other General Utilities
"""
import numpy as np
import copy


def lister(value):
    """
    Static (Module) Method for converting a scalar into a list if not already. Will also ignore None.

    :param value: The Value to convert
    :return: A List containing value if a scalar, or value if already a list/tuple/numpy array/none
    """
    return value if (type(value) in (tuple, list, np.ndarray) or value is None) else [value, ]


def dictextend(_d1, _d2, deep=False):
    """
    This function extends the elements of _d1 by those in _d2. The elements in either case must support the extend
    method (i.e. are typically lists) - scalars are however supported through the lister method. Note that if a key
    exists in _d2 and not in _d1, it is automatically created as a list.

    :param _d1: Dictionary to extend: will be modified
    :param _d2: Dictionary to copy data from.
    :param deep: If True, then the elements in _d2 are deep copied when extending/creating
    :return:    Updated _d1 for chaining etc...
    """
    # Extend/Copy values
    for key, value in _d2.items():
        if key in _d1:
            _d1[key].extend(lister(copy.deepcopy(value) if deep else value))
        else:
            _d1[key] = lister(copy.deepcopy(value) if deep else value)

    # Return _d1
    return _d1


class NullableSink:
    """
    Defines a wrapper class which supports a nullable initialisation
    """
    def __init__(self, obj=None):
        self.Obj = obj

    def write(self, str):
        if self.Obj is not None:
            self.Obj.write(str)

    def flush(self):
        if self.Obj is not None:
            self.Obj.flush()
