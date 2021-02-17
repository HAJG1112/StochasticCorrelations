import pandas as pd
import numpy as np
from collections import deque
import math
from math import sqrt, exp, log 
import scipy.optimize
import numpy as np
import ou_mle as ou
# Modelling Stochastic Correlation: Long Teng, Matthias Ehrhardt and Michael Gunther (2016)

class ModifiedOU:
    def get_coeff(self, dataset, rolling_window, dt):
        '''
        Finds the $ allocation ratio to stock B to maximize the log likelihood
        from the fit of portfolio values to an OU process

        input: subset: 2xN data series containing the rolling correlations of stock A and B
               dt - time increment (1 / days(start date - end date))
        returns: k*, µ*, σ*
        B* is the allocation ratio between two correlated brownian process that makes the residual stationary
        '''
        theta = mu = sigma = 0
        max_log_likelihood = 0

        def compute_coefficients(x):
            correlation_values = self.compute_correlation_values(dataset, rolling_window) #returns a matrix of width 1 and length = dataset-rolling_window
            return ou.estimate_coefficients_MLE(correlation_values, dt)  # mu, k, sigma, max_log_likelihood

        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.01, 1, 100)
        res = vectorized(linspace)
        index = res[3].argmax()
        return res[0][index], res[1][index], res[2][index], res[3][index]

    def compute_correlation_values(self, dataset, rolling_window):
        '''
        input: dataset: raw dataframe
                window: window size to calculate rolling correlations
        outputs: list of rolling correlations o
        '''
        corr_rolling = []
        for i in range(len(dataset) - rolling_window):
            _corr = dataset[i:i+rolling_window].corr().iloc[0,1]
            corr_rolling.append(_corr)
        #get it into a dateframe with datetime indexing
        #corr_rolling = pd.DataFrame(corr_rolling, columns = [f'Rolling_Correlation_{rolling_window}'], index = dataset[rolling_window:].index)
        return corr_rolling
