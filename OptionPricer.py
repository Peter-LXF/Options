import numpy as np
from scipy import stats

class VanillaOption(object):

    def __init__(self, S, K, tau, sigma, r = 0, optType = 'Call'):
        self.S = np.array(S, dtype='float64').reshape(-1, 1)
        self.K = np.array(K, dtype='float64').reshape(-1, 1)
        self.tau = np.array(tau, dtype='float64').reshape(-1, 1)
        self.sigma = np.array(sigma)
        self.d1 = (np.log(self.S/self.K)) + (r+0.5*self.sigma**2) * self.tau / self.sigma / self.tau**0.5
        self.d2 = self.d1 - sigma*self.tau**0.5
        self.optType = optType

    def get_delta(self):
        if self.optType == 'Call':
            return stats.norm.cdf(self.d1, 0., 1.)
        else:
            return stats.norm.cdf(self.d1, 0., 1.) - 1

    def get_gamma(self):
        return 1.0 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * self.d1 ** 2) / self.S / self.sigma / self.tau ** 0.5

    def get_vega(self):
        return self.S / (2 * np.pi) ** 0.5 * np.exp(-0.5 * self.d1 ** 2) * self.tau ** 0.5

    def get_theta(self):
        return -self.S / (2 * np.pi) ** 0.5 * np.exp(-0.5 * self.d1 ** 2) * self.sigma / 2 / self.tau ** 0.5

    def get_price(self):
        if self.optType == 'Call':
            return self.S * stats.norm.cdf(self.d1, 0., 1.) - self.K * stats.norm.cdf(self.d2, 0., 1.)
        else:
            return self.K * stats.norm.cdf(-self.d2, 0., 1.) - self.S * stats.norm.cdf(-self.d1, 0., 1.)