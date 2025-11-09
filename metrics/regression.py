import numpy as np



loss = lambda y, p: (y - p)**2

def mse(y_true, y_pred): 
    """  
    calculates the mean squared error value
    :param y_true: true target values
    :param y_pred: predicted values by a model
    :returns: float value
    """
    N = len(y_true)

    return (1/N) * np.sum(loss(y_true, y_pred))