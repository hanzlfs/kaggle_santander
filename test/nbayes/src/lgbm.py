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

#cell 4
dataset_root = '../'
dataset = SantanderDataset(dataset_root, isLag = True)

#cell 5
## I have been testing the class and seems to be working fine. When loaded the dataset is using only 500MB of RAM.

#cell 6
## Testing with Naive Bayes

#cell 7
def train_bnb_model(msg):
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
    #Fit the model
    #bnb = LogisticRegression()
    bnb = lgb.LGBMClassifier(n_estimators=100)
    print "training data size, ", input_data.shape, " training target size, ", output_data.shape
    #print "Column_6 unique vals: ", len(np.unique(input_data[:,6]))
    #print "Column_9 unique vals: ", len(np.unique(input_data[:,9]))
    #print "Column_20 unique vals: ", len(np.unique(input_data[:,20]))
    #print "Column_26 unique vals: ", len(np.unique(input_data[:,26]))
    #return 
    bnb.fit(input_data, output_data, eval_metric="multi_logloss")
    #bnb = BernoulliNB(alpha=1e-2)
    #bnb.partial_fit(input_data, output_data, classes = range(24))
    return bnb

#cell 8
def create_prediction(bnb, msg):
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
    rank = bnb.predict_proba(input_data)
    # if some labels are missing, fill zeros in rank so that the shape matchs nsamp * 24


    filtered_rank = np.equal(previous_products, 0) * rank
    predictions = np.argsort(filtered_rank, axis=1)
    predictions = predictions[:,::-1][:,0:7]
    return predictions, output_data

#cell 9
def naive_bayes_workflow(msg):
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
    bnb = train_bnb_model(msg)
    scores = []
    for month in msg['eval_month']:
        msg_copy = msg.copy()
        msg_copy['month'] = month
        #Create prediction
        predictions, output_data = create_prediction(bnb, msg_copy)
        #Get the score
        score = mapk(output_data, predictions)
        scores.append(score)
    
    return scores, bnb


#cell 17
## I'm going to launch a new search to see if I can get the same results or better.

#cell 19
#Define evaluation function
def eval_function_1(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_6',individual,verbose=False)
    return ret[0:1]

#cell 24
#Define evaluation function
def eval_function_2(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_8',individual,verbose=False)
    return [np.sum(ret)/2]

#cell 28
#Define evaluation function
def eval_function_3(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_9',individual,verbose=False)
    return ret[0:1]


#cell 31
def get_genomic_score(test_month, filename, genome, verbose=False):
    """
    Receives only test month and the genome
    Returns the score and saves the configuration and results in a file
    It's the same function as above but without training with month 16
    
    If the genome size is 35 then use_product is set to True
    If the genome size is 36 then all the parameters are in the search
    len(categorical_columns) = 18
    So we need a genome of 18+2+1 = 21
    """
    if verbose:
        print(genome)
    #Decide which train months to use, from 1 to 16
    if np.sum(genome[0:16]) > 0:
        used_months = np.array(range(1,17))[np.array(genome[0:16]) == 1]
        train_month = used_months
    else:
        #Select a random month
        used_months = np.random.randint(1,17,1)[0]
        train_month = [used_months]
    if verbose:
        print('train_month', train_month)
    #Decide wich category input columns to use
    categorical_columns = dataset.categorical_columns
    used_index = np.arange(len(categorical_columns))[
        np.array(genome[16:34]) == 1]
    input_columns = [categorical_columns[i] for i in used_index]
    if verbose:
        print('input_columns', input_columns)
    #Decide on using change columns and product as input
    use_change = genome[34] == 1
    #This allows to use a shorter genome to fix some properties
    if len(genome) >= 36: 
        use_product = genome[35] == 1
    else:
        use_product = True
    #Build message for training 
    msg ={'train_month':list(train_month),
          'eval_month':test_month,
          'input_columns':input_columns,
          'use_product':use_product,
          'use_change':use_change,
        
    }
    if verbose:
        print(msg)
    ret = naive_bayes_workflow(msg)
    #Print and save to file 
    text = '\t'.join([str(a) for a in ret[0]]) + '\t'
    text += '%s\t%s\t' % ( use_change, use_product)
    if verbose:
        print(text)
    text += "','".join(input_columns)
    text += "\t" + ",".join([str(a) for a in train_month])
    text += '\n'
    with open(dataset_root+'logs/%s.log' % filename, 'a') as f:
        f.write(text)
        
    return ret[0]

#cell 32
#Define evaluation function
def eval_function_4(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_10',individual,verbose=False)
    return [np.sum(ret)/2]


#cell 35
## Submission

#cell 36
## I have to create a submission function, I will reuse the one from the previous notebook.

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
    ret = naive_bayes_workflow(msg)
    scores = ret[0]
    bnb = ret[1]
    #Create a prediction
    msg['month'] = test_month
    predictions, output_data = create_prediction(bnb, msg)
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

def get_msg():
    """
    user specified 
    define input features, months, interactions and conditions here
    """

    #####0. define base ########
    msg = {
    'train_month': [1,2,5,6,10,11,16],
    'eval_month': [5, 16],
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
    'use_gbdt_feature': False}

    ######1. define interactions ########
    profile_feature = ['renta', 'age', 'indrel','indrel_1mes','indext', 'segmento']
    is_prod_feature = True
    profile_lag = [0]
    prod_lag = [1]
    interact_order = 2
    interact_option = 'individual'
    msg['input_columns_interactions'] = create_interaction_list(profile_feature = profile_feature, \
                                            is_prod_feature = is_prod_feature, \
                                            profile_lag = profile_lag, prod_lag = prod_lag, \
                                            interact_order = interact_order, interact_option = interact_option)

if __name__ == "__main__":
    submission_file_name = 'lgbm_1204_02'
    msg = get_msg()
    start_time = time.time()
    print(create_submission(submission_file_name,msg))
    print(time.time()-start_time)

