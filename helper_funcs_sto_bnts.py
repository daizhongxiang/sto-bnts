import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import pickle

def acq_max(ac, pred_model, pred_param, bounds):
    para_dict={"pred_model":pred_model, "pred_param":pred_param}
    
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(10000, bounds.shape[0]))

    ys = []
    for x in x_tries:
        ys.append(ac(x.reshape(1, -1), para_dict))
    ys = np.array(ys)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(100, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), para_dict),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    return x_max


class UtilityFunction(object):
    def __init__(self):
        pass
    def utility(self, x, para_dict):
        pred_model, pred_param = para_dict["pred_model"], para_dict["pred_param"]

        return self._ts(x, pred_model, pred_param)

    @staticmethod
    def _ts(x, pred_model, pred_param):
        test_pred = pred_model(pred_param, x)
        
        return np.squeeze(test_pred)
