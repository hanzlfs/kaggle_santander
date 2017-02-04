import numpy as np
import pandas as pd
import time
import lightgbm as lgb

class LightGBMTrans(object):
    """
    fit raw data with light gbm
    export leaf indices
    """
    def __init__(self):
        self.params = {"num_iteration": 10, 
                       "n_estimators" : 30, 
                       "max_depth" : 6, 
                       "eval_metrics":"multi_logloss"}
    
    def lgbm_params(self, params):
        self.params= params
        
    def lgbm_trans(self,X, y):
        params = self.params
        lgb_model = lgb.LGBMClassifier(n_estimators=params["n_estimators"], 
                                       max_depth = params["max_depth"])
        lgb_model.fit(X, y, eval_metric=param["eval_metrics"], 
                      verbose = False)
        features = lgb_model.apply(X, num_iteration= params["num_iteration"])
        return features

class LogistRegTrans(object):
    """
    fit raw data with logist regression
    export coefficients for random forest 
    """
    def __init___(self):
        self.params = {}
        

        