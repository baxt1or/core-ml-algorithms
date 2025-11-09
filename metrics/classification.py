import numpy as np


loss = lambda y, p: y*np.log(p) + (1-y)*np.log(1-p)


def log_loss(y_true, y_pred):
    """ 
    gives the log loss value
    :param y_true: true target values
    :param y_pred: values predicted by a model
    :returns: float value
    """

    if len(np.unique(y_pred)) <= 2:
        raise ValueError("log loss takes probabilistic vales not explicit values")
    N = len(y_true)

    return -(1/N) * np.sum(loss(y_true, y_pred))