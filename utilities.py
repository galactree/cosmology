import numpy as np
import scipy

def deriv_central_diff(param, func, delta, *args):
    """
    Numerical derivative using the central difference method

    #########
    Parameters

    param (float,): value of the variable to compute the derivative at

    func (callable, ): the function to differentiate

    delta (float, ): dx to use

    #########
    Returns

    Numerical Derivative evaluated at param

    """
    return (func(param + delta, *args) - func(param - delta, *args)) / (2 * delta)

def j_1(x):
    '''
    
    First order spherical Bessel function of the first kind

    ##########
    Parameters

    x (arraylike or float,):  variable

    #########
    Returns

    J1 (arraylike or float,): J_1(x) First order spherical Bessel function of the first kind evaluated at x

    '''
    return (np.sin(x)/x**2) - (np.cos(x)/x)