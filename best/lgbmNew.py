#light GBM workflow
#Include new train-val split scheme 

import pandas as pd
import numpy as np
import time
import gzip
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from dataset import SantanderDataset
from average_precision import mapk
from genetic_search import genetic_search

import random
import lightgbm as lgb
from sklearn import datasets, metrics, model_selection
from common import *

dataset_root = '../'
dataset = SantanderDataset(dataset_root, isLag = True, lags = range(1, 12))


def train_model(msg, params, train_for_validation = True):
    """
    Trains a model using the given parameters
    
    train_for_validation: bool, it True, then use the train months aside from validation, 
    else use all train months for production
    """    
    if train_for_validation:
        # only use train month that not in eval month
        input_data, output_data = dataset.X_tr, dataset.y_tr
    else:
        # use all months in train month
        input_data, output_data = dataset.train_data, dataset.train_label
    
    print "training data size, ", input_data.shape, " training target size, ", output_data.shape
    #Get unique train labels in case it is incomplete
    unq_lb = sorted(np.unique(output_data).tolist())

    #Train lgbm model
    clf = lgb.LGBMClassifier(n_estimators = params.get('n_estimators', 100), nthread = 8) # specify nthread = 8 to speed up 
    clf.fit(input_data, output_data, eval_metric="multi_logloss")   

    # Save model
    saving_path = msg.get('model_path', None)
    if saving_path is not None:
        clf.save_model(dataset_root + 'saved_model/' + saving_path + '.txt')      

    return clf, unq_lb

def create_prediction(clf, input_data, previous_products, unq_lb):
    """
    Makes a prediction using the given model and parameters

    """
    print "predicting data size, ", input_data.shape
    #Get the prediction
    rank = clf.predict_proba(input_data)
    # if some labels are missing, fill zeros in rank so that the shape matchs nsamp * 24
    if rank.shape[1] < 24:
        print "rank,", rank.shape, "label, ", unq_lb
        assert rank.shape[1] == len(unq_lb)
        rank_copy = np.zeros((rank.shape[0], 24))
        rank_copy[:, unq_lb] = rank.copy()
        filtered_rank = np.equal(previous_products, 0) * rank_copy
    else:
        filtered_rank = np.equal(previous_products, 0) * rank
    predictions = np.argsort(filtered_rank, axis=1)
    predictions = predictions[:,::-1][:,0:7]
    return predictions 


def validation(clf, unq_lb):
    """
    make prediction on eval set output validation scores 

    """
    predictions = create_prediction(clf, dataset.X_val, \
                                    dataset.p_val, unq_lb)
    output_data = dataset.y_val
    #Get the score
    print output_data.shape
    print output_data[0:10]

    score = mapk(output_data, predictions)
    return score

def create_submission(filename, clf, unq_lb):
    """
    make submission 
    
    """
    #Get prediction on test dataset
    predictions = create_prediction(clf, dataset.test_data, \
                                    dataset.test_prev_prod, unq_lb)

    #Produce submission file
    test_month = 17
    target_cols = np.array(dataset.product_columns)    
    final_preds = [" ".join(list(target_cols[pred])) for pred in predictions]
    test_id = np.array(dataset.eval_current.loc[dataset.eval_current.fecha_dato == test_month, "ncodpers"])
    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_df = out_df[["ncodpers","added_products"]]
    out_df.to_csv(dataset_root + 'submissions/' + filename + '.csv', index=False)

def workflow(msg, params, isTest = True):
    """
    train, validation and submission workflow
    
    produce submission if istest == True, else only do validation
    """

    # Validation
    # train model with train months except eval months
    clf, unq_lb = train_model(msg, params, train_for_validation = True)
    score = validation(clf, unq_lb)
    
    # submission
    if isTest:
        # train the model with all train months
        clf, unq_lb = train_model(msg, params, train_for_validation = False)        
        create_submission(msg['filename'], clf, unq_lb)
    print "validation map@7 = ", score
    print('Calculate feature importances...')
        # feature importances
    print('Feature importances:', list(clf.feature_importance()))

    #### Write feature importance to file #####

    feature_names = pd.DataFrame({'position': range(0, len(dataset.feature_names)),\
                                    'feature_name': dataset.feature_names, \
                                    'fi': list(clf.feature_importance())}) 

    feature_names = feature_names.sort_values(['fi'], ascending = [False])
    feature_names.to_csv('../input/'+ msg['file_fi'] + '.csv', header = True, index = False)
    

def get_msg(filename = None, file_fi = None, model_path = None):
    """
    user specified 
    define input features, months, interactions and conditions here

    filename: submission file name
    model_path: 
    """
    ##### define base ########
    msg = {
        'train_month': [1,2,5,6,10,11,16], 
        'eval_month': None, 
        'input_columns': ['renta','age','indrel','indrel_1mes','indext','segmento','month'], 
        'use_product': True,
        'use_change': True,
        'use_product_lags': [2,3,4,5,6,7,8,9,10,11],
        'use_profile_lags': [1,2,3,4,5,6,7,8,9,10],
        'input_columns_lags': ['indrel', 'indext', 'indrel_1mes', 'segmento'], 
        'use_product_change_lags': {'lags':[2,3,4,5,6,7,8,9,10,11], 'use':'month-based'}, 
        'use_profile_change_lags': [0,1,2,3,4], 
        'input_columns_change': ['indrel', 'indext', 'indrel_1mes', 'segmento'], 
        'use_gbdt_feature': False
    }

    ##### path to save lgbm model #####
    if model_path is not None :
        msg.update({'model_path': model_path})

    ##### path to write submission files ####
    if filename is not None:
        msg.update({'filename': filename})

    #### csv file to keep feature name and feature importance####
    if file_fi is not None:
        msg.update({'file_fi': file_fi})

    ###### define interactions ########
    msg['input_columns_interactions'] = [['indrel','indrel_1mes','indext', 'segmento']]
    
    ###### define sequences ########
    sequence_columns = []
    sequence_lags = [1,3,5]
    for col in feat_prod:
        sequence_columns.append([col + '_L' + str(lag) for lag in sequence_lags])

    msg['sequence_columns'] = None

    return msg

def get_param():
    """
    user specified 
    define model parameters for light gbm
    """
    params = {}
    params['n_estimators'] = 100
    return params

if __name__ == "__main__":
    
    msg = get_msg(filename = 'lgbm-1212-04', file_fi = 'lgbmFI-1212-04', model_path = None)
    params = get_param()

    start_time = time.time()
    dataset.get_train_val_split_data(msg, r_val = 0.1)  
    dataset.get_test_data(msg)  
    workflow(msg, params, isTest = True)
    print(msg, params)  
    print(time.time()-start_time)


