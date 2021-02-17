from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma
from numba import jit

class DCC_GARCH:

    def __init__(self, dist = 'norm'):
            if dist == 'norm' or dist == 't':
                self.dist = dist
            else: 
                print("Takes pdf name as param: 'norm' or 't'.")

    def garch_fit(self, returns):
        res = minimize( self.garch_loglike, (0.01, 0.01, 0.94), args = returns,
              bounds = ((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        LogL = np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns.A1**2)/(2*var_t))
        return -LogL


    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)     
        for i in range(T):
            if i==0:
                var_t[i] = returns[i]**2
            else: 
                var_t[i] = omega + alpha*(returns[i-1]**2) + beta*var_t[i-1]
        return var_t        

    def __compute_volatility_loglikelihood(self, params, *args):
        #file:///C:/Users/justi/Downloads/SSRN-id236998.pdf
        a = params[0]
        b = params[1]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        D_t = args

        summation_term = 0
        for i in range(1, self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))
            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))
            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))
            
            summation_term += self.N*np.log(2*np.pi) + \
                            np.log(np.linalg.det(D_t[i])**2) + \
                            np.matmul( np.linalg.inv(self.rt[i]) , np.matmul( np.linalg.inv(D_t[i])**2, self.rt[i]))
        
        loglike = -0.5*summation_term
        return -loglike   #we need to retrun as estimate of D_t which we pump into the second step estimation

    def __compute_correlation_loglikelihood(self, params, D_t):
        a = params[0]
        b = params[1]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        

        loglike = 0
        for i in range(1, self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))
            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))
            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))
            loglike = loglike + np.log(np.linalg.det(R_t[i])) + \
                                np.matmul(et.T, np.matmul(np.linalg.inv(R_t[i]), et.T)) - \
                                np.matmul(et.T, et)
            
        return -loglike  

    def fit(self, returns):
        self.rt = np.matrix(returns)
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            params = self.garch_fit(self.rt[:,i])
            D_t[:,i] = np.sqrt(self.garch_var(params, self.rt[:,i]))
        self.D_t = D_t
        
        if self.dist == 'norm':
            res = minimize(self.__compute_volatility_loglikelihood, (0.01, 0.94), args = D_t,
            bounds = ((1e-6, 1), (1e-6, 1)), 
            #options = {'maxiter':10000000, 'disp':True},
            )
            self.a = res.x[0]
            self.b = res.x[1]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b} 
      
    def predict(self, ndays = 1):
        if 'a' in dir(self):
            Q_bar = np.cov(self.rt.reshape(self.N, self.T))

            Q_t = np.zeros((self.T,self.N,self.N))
            R_t = np.zeros((self.T,self.N,self.N))
            H_t = np.zeros((self.T,self.N,self.N))

            Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

            for i in range(1,self.T):
                dts = np.diag(self.D_t[i])
                dtinv = np.linalg.inv(dts)
                et = dtinv*self.rt[i].T
                Q_t[i] = (1-self.a-self.b)*Q_bar + self.a*(et*et.T) + self.b*Q_t[i-1]
                qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))
                R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))
                H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))  

            if self.dist == 'norm':
                return {'dist': self.dist, 'cov': H_t[-1]*np.sqrt(ndays)}
            else:
                print('Model not fit')
                
