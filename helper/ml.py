#light GBM workflow
#Include new train-val split scheme 

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from feature_names import FEATNAME
from average_precision import apk, mapk

def train_model(y, X, model = None):    
    """
    y, X: data
    model: initialization of a ml model
    """

    unq_lb = sorted(np.unique(y).tolist())    
    model.fit(X, y, eval_metric="multi_logloss")   

    return model, unq_lb

def create_prediction(model, X, previous_products, unq_lb):
    """
    Makes a prediction using the given model and parameters
    
    model: trained model
    X: test set
    previous_products: previous product records
    unq_lb: unique labels
    """

    rank = model.predict_proba(X)
    # if some labels are missing, fill zeros in rank so that the shape matchs nsamp * 24
    if rank.shape[1] < 24:
        rank_copy = np.zeros((rank.shape[0], 24))
        rank_copy[:, unq_lb] = rank.copy()
        filtered_rank = np.equal(previous_products, 0) * rank_copy
    else:
        filtered_rank = np.equal(previous_products, 0) * rank
    predictions = np.argsort(filtered_rank, axis=1)
    predictions = predictions[:,::-1][:,0:7]

    return predictions 


def create_submission(predictions, test_id, filename):
    """
    make submission 
    
    """
    test_month = 17
    target_cols = np.array(FEATNAME.COLNAMES["product"])    
    final_preds = [" ".join(list(target_cols[pred])) for pred in predictions]   
    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_df = out_df[["ncodpers","added_products"]]
    out_df.to_csv(filename, index=False)

