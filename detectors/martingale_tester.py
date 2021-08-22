import numpy as np
from utils.metrics import metrics_in_range

def approx_equal(a, b, tol=10**-6):
    return np.abs(a - b) < tol
    
def martingale_tester_grid_search(dataset, changes_gt, epsilon_range, m_range, interval, verbose=0):
    """
    Given a dataset and ranges of epsilon and M parameters, perform a grid search
    to find the configuration of MartingaleTester(e, M) that gives the best F1 score
    wrt the given interval.
    
    Params:
        - dataset (np.ndarray): dataset on which to perform the grid search
        - changes_gt (List[int]): list of indexes in which change in stream has occurred
        - epsilon_range (List[float]): range of epsilon parameters
        - m_range (List[float]): range of M parameters
        - interval (float): interval of prediction to use when computing F1 score
    Returns:
        - best_params: dictionary with best parameters
        - (precision, recall): precision and recall lists for all the tested parameters
    """
    # grid search
    #epsilon = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
    #M = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # interval window inside which we consider a predicted change to hold true
    # note that 1 unit of interval corresponds to 3 seconds of real time.
    #interval = 15
    best_params = {"F1": 0.0, "params": None}
    precision_list = []
    recall_list = []
    e_list = []
    m_list = []
    f1_list = []
    for e in epsilon_range:
        for m in m_range:
            martingale_tester = MartingaleTest(m, epsilon=e)
            changes_pred = run_martingale_tester(
                martingale_tester,
                dataset
            )
            metrics = metrics_in_range(gt=changes_gt, pred=changes_pred, interval=interval)
            if metrics["F1"] > best_params["F1"]:
                best_params["F1"] = metrics["F1"]
                best_params["recall"] = metrics["recall"]
                best_params["precision"] = metrics["precision"]
                best_params["params"] = (e, m)
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])
            e_list.append(e)
            m_list.append(m)
            f1_list.append(metrics["F1"])
            if verbose:
                print("e={}, m={}, F1={}, Recall={}".format(e, m, metrics["F1"], metrics["recall"]))
    return best_params, (precision_list, recall_list, e_list, m_list, f1_list)

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
