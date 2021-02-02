import pandas as pd
import numpy as np

# Modelling Stochastic Correlation: Long Tend, Matthias Ehrhardt and Michael Gunther (2016)

class ModifiedOU():
    
    def __init__(self):
        pass
    
    def ou_coeffs(self):
        # inputs k, miu and sigma from calibration of OU SDE 
        # returns k_star, a_star, sig_star 
        pass

    def _a(self):
        pass

    def _b(self):
        pass

    def _M(self):
        # takes params a nd b and using the gamma function and hypergeometric function for calculation
        pass

    def _f(self):
        # fit eq 39 to the historical correlations of window size w
        pass
    
    def calibrate_tdf_to_edf(self):
        pass