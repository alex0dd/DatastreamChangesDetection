import numpy as np

def approx_equal(a, b, tol=10**-6):
    return np.abs(a - b) < tol

def run_martingale_tester(tester, series):
    """
    Given a martingale tester and a time series, 
    apply the tester to detect changes.
    """
    test_values = []
    if type(series) == np.ndarray:
        series_iterator = enumerate(series)
    else:
        # pandas series
        series_iterator = series.items()
    for time, sample in series_iterator:
        sample_d = [sample] 
        test_value = tester.step(sample_d)
        test_values.append(test_value)
    test_values = np.array(test_values)
    return tester.changes_history

class MartingaleTest:
    
    def __init__(self, lmb, epsilon=0.98):
        """
        lambda: threshold value
        epsilon
        """
        self.lmb = lmb
        self.epsilon = epsilon
        self._initialize()
        
    def _initialize(self):
        self.M = [1]
        self.T = [] # online training set
        self.i = 1
        self.strangeness_total = []
        self.changes_history = []
        
    def strangeness(self, T, x_i):
        """
        T: np.array or list of shape [B, Dim]
        x_i: np.array of shape [1, Dim]
        """
        centroid = np.mean(T, axis=0) # compute cluster of T (unlabeled training set)
        strangeness_x_i = np.linalg.norm(x_i - centroid)
        return strangeness_x_i
    
    def compute_p_value(self, strangeness_total):
        strangeness_total = np.array(strangeness_total)
        theta_j = np.random.uniform()
        i = len(strangeness_total) - 1 # current time step
        p_val = (
            np.sum(strangeness_total[:i] > strangeness_total[i]) + \
            theta_j * np.sum(approx_equal(strangeness_total[:i], strangeness_total[i])) \
        ) / (i + 1)
        # NOTE: p_val can be <= only in case when we have a single strangeness value
        if p_val <= 0.0:
            p_val = 1. / (i + 1)
        return p_val
        
    def step(self, observation):
        """
        Performs a step of the test (one loop iteration)
        
        Idea:
        - Algorithm gathers a training set from a stream.
        - When it detects a change in stream distribution, it resets the training set.
        """
        # a new example x_i is observed
        x_i = observation
        if len(self.T) == 0:
            # set strangeness of x_i to 0
            strangeness_x_i = 0.0
        else:
            # compute strangeness of x_i and data points in T
            strangeness_x_i = self.strangeness(self.T, x_i)
        self.strangeness_total.append(strangeness_x_i)
        # compute P-values p_i using (7)
        p_i = self.compute_p_value(self.strangeness_total)
        M_i_1 = self.M[-1]
        # Compute M(i) using (6)
        M_i = self.epsilon * (p_i ** (self.epsilon - 1.0)) * M_i_1
        self.M.append(M_i)
        if self.M[-1] > self.lmb:
            # change detected
            # Set M(i) = 1
            self.M[-1] = 1.
            # reinitialize T to empty set
            self.T = []
            self.strangeness_total = []
            test_value = 1 # change
            self.changes_history.append(self.i)
        else:
            self.T.append(x_i)
            test_value = 0 # no change
        # increment i
        self.i += 1
        return test_value
