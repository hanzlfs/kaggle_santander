#cell 0
# Naive Bayes Third Round

#cell 1
## I'm going to use the new splitted dataset to train a naive bayes model. I will be using a class for the dataset, and this will help me for later using a unified class for the model. 

#cell 2
## Load the dataset

#cell 3
#Imports
import numpy as np
from sklearn.naive_bayes import BernoulliNB
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


#cell 4
dataset_root = '../'
dataset = SantanderDataset(dataset_root, isLag = True)

#cell 5
## I have been testing the class and seems to be working fine. When loaded the dataset is using only 500MB of RAM.

#cell 6
## Testing with Naive Bayes

#cell 7
def train_lgbm_model(msg):
    """
    Trains a model using the given parameters
    
    month: int or list with the number of the month we want
        the data to be taken of
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    msg_copy = msg.copy()
    msg_copy['train'] = True
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['train_month']
    #Get the data for training
    ret = dataset.get_data(msg_copy)
    input_data, output_data = ret[0:2]
    unq_lb = sorted(np.unique(output_data).tolist())
    saving_path = msg_copy['model_path']
    
    clf = lgb.LGBMClassifier(n_estimators=100, nthread = 6) # specify nthread = 8 to speed up 
    print "training data size, ", input_data.shape, " training target size, ", output_data.shape
    #print "Column_6 unique vals: ", len(np.unique(input_data[:,6]))
    #print "Column_9 unique vals: ", len(np.unique(input_data[:,9]))
    #print "Column_20 unique vals: ", len(np.unique(input_data[:,20]))
    #print "Column_26 unique vals: ", len(np.unique(input_data[:,26]))
    clf.fit(input_data, output_data, eval_metric="multi_logloss")
    if saving_path is not None:
        clf.save_model(saving_path)
        
    return clf, unq_lb

#cell 8
def create_prediction(clf, msg, unq_lb):
    """
    Makes a prediction using the given model and parameters
    
    month: int or list with the number of the month we want
        the data to be taken of
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    msg_copy = msg.copy()
    msg_copy['train'] = False
    if not 'month' in msg_copy.keys():
        msg_copy['month'] = msg_copy['eval_month']
    #Get the data for making a prediction
    ret = dataset.get_data(msg_copy)
    input_data, output_data, previous_products = ret
    print "predicting data size, ", input_data.shape, " predicting target size, ", output_data.shape
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
    return predictions, output_data

#cell 9
def prediction_workflow(msg):
    """
    Implements all the steps of training and evaluating a naive bayes classifier
    Returns the score and the trained model
    
    train_month: int or list with the number of the month we want
        the data to be taken of for training 
    eval_month: int or list with the number of the month we want
        the data to be taken of for testing
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    if type(msg['eval_month']) is not list:
        msg['eval_month'] = [msg['eval_month']]
    #Train the model
    clf, unq_lb = train_lgbm_model(msg)
    scores = []
    for month in msg['eval_month']:
        msg_copy = msg.copy()
        msg_copy['month'] = month
        #Create prediction
        predictions, output_data = create_prediction(clf, msg_copy, unq_lb)
        #Get the score
        score = mapk(output_data, predictions)
        scores.append(score)
    
    return scores, clf, unq_lb



#cell 37
def create_submission(filename, msg, 
                        verbose=False):
    """
    Implements all the steps of training and evaluating a naive bayes classifier
    Returns the score and the trained model
    
    train_month: int or list with the number of the month we want
        the data to be taken of for training 
    eval_month: int or list with the number of the month we want
        the data to be taken of for testing
    input_columns: a list with the name of the columns we are going to use
        in the task
    use_product: bool, if true adds the product columns of the month before
    use_change: bool, if true adds the change columns of the month before
    """
    test_month = 17
    #Train the model and get validation scores
    ret = prediction_workflow(msg)
    scores = ret[0]
    bnb = ret[1]
    unq_lb = ret[2]
    #Create a prediction
    msg['month'] = test_month
    predictions, output_data = create_prediction(bnb, msg, unq_lb)
    #Create the submission text
    if verbose:
        print('Creating text...')
    text='ncodpers,added_products\n'
    for i, ncodpers in enumerate(dataset.eval_current[dataset.eval_current.fecha_dato == test_month].ncodpers):
        text += '%i,' % ncodpers
        for j in predictions[i]:
            text += '%s ' % dataset.product_columns[j]
        text += '\n'
    #Write to file
    if verbose:
        print('Writing to file...')
    #with gzip.open(dataset_root + 'submissions/%s.csv.gz' % filename, 'w') as f:
    #    f.write(bytes(text, 'utf-8'))
    with open(dataset_root + 'submissions/%s.csv' % filename, 'w') as f:
        f.write(text)
    return scores

def get_msg(model_path = None):
    """
    user specified 
    define input features, months, interactions and conditions here
    """

    #####0. define base ########
    msg = {
    #'train_month': [1,2,5,6,10,11,16],
    #'eval_month': [5, 16],
    'train_month': [5],
    'eval_month': [16],
    'input_columns': ['renta', 'pais_residencia','age','indrel','indrel_1mes','indext','segmento','month'], # the input columns not lag
    'use_product': True,
    'use_change': True,
    'use_product_lags': [2,3,4,5],
    'use_profile_lags': [1,2,3,4,5],
    'input_columns_lags': ['indrel', 'indext', 'indrel_1mes', 'segmento'], # profile features for which we include in lags as well
    'input_columns_interactions': [], # groups of interactions we include in training
    'use_product_change_lags': [], # lags for which we use product change features
    'use_profile_change_lags': [], # lags for which we use profile change features
    'input_columns_change': [], # profile features for which we collect change info
    'use_gbdt_feature': False,
    'model_path':None
    }
    if model_path is not None :
        msg.update({'model_path': model_path})
    ######1. define interactions ########
    profile_feature = ['renta', 'age', 'indrel','indrel_1mes','indext', 'segmento']
    is_prod_feature = False
    profile_lag = [0]
    prod_lag = [1]
    interact_order = 2
    interact_option = 'individual'
    msg['input_columns_interactions'] = create_interaction_list(profile_feature = profile_feature, \
                                            is_prod_feature = is_prod_feature, \
                                            profile_lag = profile_lag, prod_lag = prod_lag, \
                                            interact_order = interact_order, interact_option = interact_option)
    return msg

if __name__ == "__main__":
    submission_file_name = 'lgbm_1204_02'
    msg = get_msg()
    start_time = time.time()
    print(create_submission(submission_file_name,msg))
    print(time.time()-start_time)

