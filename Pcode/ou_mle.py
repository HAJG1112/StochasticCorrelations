import math
from math import sqrt, exp, log, pi  # exp(n) == e^n, log(n) == ln(n)
import scipy.optimize as so
import numpy as np
from mpmath import mp

def __compute_log_likelihood(params, *args):
    '''
    Compute the average Log Likelihood, this function will by minimized by scipy.
    Find in (31) in linked paper

    returns: the average log likelihood from given parameters
    '''
    # functions passed into scipy's minimize() needs accept one parameter, a tuple of
    #   of values that we adjust to minimize the value we return.
    #   optionally, *args can be passed, which are values we don't change, but still want
    #   to use in our function (e.g. the measured heights in our sample or the value Pi)

    k, mu, sigma = params
    X, dt, pi = args
    n = len(X)
    term_1 = 0
    term_2 = 0
    term_3 = 0

    for i in range(n):
        term_1 += mp.log( mp.sqrt(  k /  (pi*sigma**2*(1-mp.exp(-2*k*dt)) ) ) )
        term_2 += mp.log( 1 / (1- X[i]**2) )
        term_3 += -k*(np.arctanh(X[i]) - np.arctanh(X[i - 1])*mp.exp(-k*dt) - mu*(1-mp.exp(-k*dt)))**2 / sigma**2*(1-mp.exp(-2*k*dt))
    
    log_likelihood = term_1 + term_2 + term_3

    return -log_likelihood

def estimate_coefficients_MLE(X, dt, tol=1e-4):
    '''
    Estimates Ornstein-Uhlenbeck coefficients (k, µ, σ) of the given array
    using the Maximum Likelihood Estimation method

    input: X - array-like time series data to be fit as an OU process, in our case X is the correlation we wish to fit
        dt - time increment (1 / days(start date - end date))
        tol - tolerance for determination (smaller tolerance means higher precision)
    returns: θ, µ, σ, Average Log Likelihood
    '''
    pi = math.pi
    bounds = ((None, None), (1e-5, None), (1e-5, None))  # k ∈ ℝ, mu > 0, sigma > 0
                                                        # we need 1e-10 b/c scipy bounds are inclusive of 0, 
                                                        # and sigma = 0 causes division by 0 error
    k_init = np.mean(X)   #mean correlation across a window of time.
    initial_guess = (k_init, 100, 100)  # initial guesses for k, mu, sigma
    result = so.minimize(__compute_log_likelihood, initial_guess, args=(X, dt, pi), bounds=bounds)
    k, mu, sigma = result.x 
    max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood
    # .x gets the optimized parameters, .fun gets the optimized value
    return k, mu, sigma, max_log_likelihood

