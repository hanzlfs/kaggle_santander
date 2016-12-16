# -*- coding: utf-8 -*-
"""
Definition of the class dataset for handling the Santander competition dataset
"""

import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from ast import literal_eval
from common import *
#import lightgbm as lgb

class SantanderDataset(object):
    """
    Class for storing the dataset of Santander competition
    It will give the data as it is requested
    """

    def __init__(self, dataset_root, isLag = False, lags = None):
        """
        Loads the dataset
        """
        if lags is not None:
            self.lags = lags

        self.dataset_root = dataset_root
        self.__load_datasets(dataset_root, isLag)
        self.__prepare_datasets()

        #self.transform_dataset_for_training(self.df)

    def __load_datasets(self, dataset_root, isLag):
        """
        Loads all the datasets
        """        
        limit_rows   = 20000000
        if isLag:
            dic_used = get_dict_type_w_lag()
            #path_tr = "input/train_current_lag_5_1209.csv"
            #path_val = "input/eval_current_lag_5_1209.csv"
            path_tr = "input/train_current_month_dataset_w_lag5_clean.csv"
            path_val = "input/eval_current_month_dataset_w_lag5_clean.csv"
        else:
            dic_used = dictionary_types
            path_tr = "input/train_current_month_dataset.csv"
            path_val = "input/eval_current_month_dataset.csv"

        """
        Read train current and val current w. lags
        """
        start_time = time.time()
        self.train_current = pd.read_csv(dataset_root + path_tr,
                                   dtype=dic_used,
                                   nrows=limit_rows)
        print('It took %i seconds to load the dataset' % (time.time()-start_time))

        start_time = time.time()
        self.eval_current = pd.read_csv(dataset_root + path_val,
                                   dtype=dic_used,
                                   nrows=limit_rows)
        print('It took %i seconds to load the dataset' % (time.time()-start_time))

        """
        Read train and eval previous
        """
        start_time = time.time()
        self.eval_previous = pd.read_csv(dataset_root + "input/eval_previous_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print('It took %i seconds to load the dataset' % (time.time()-start_time))
       
        start_time = time.time()
        self.train_previous = pd.read_csv(dataset_root + "input/train_previous_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print('It took %i seconds to load the dataset' % (time.time()-start_time))

        print(len(self.eval_current), len(self.eval_previous))
        print(len(self.train_current), len(self.train_previous))
        return


    def __prepare_datasets(self, verbose=False):
        """
        Private function for modifying the datasets before training
        """
        for df in [self.train_current, self.eval_current]:
            #Discretize the data
            renta_ranges = [0]+list(range(20000, 200001, 10000))
            renta_ranges += list(range(300000, 1000001, 100000))+[2000000, 100000000]
            df.renta = pd.cut(df.renta, renta_ranges, right=True)
            # I'm going to use periods of one year
            antiguedad_ranges = [-10]+list(range(365, 7301, 365))+[8000]
            df.antiguedad = pd.cut(df.antiguedad, antiguedad_ranges, right=True)
            #age
            age_ranges = list(range(0, 101, 10))+[200]
            df.age = pd.cut(df.age, age_ranges, right=True)
            #Create month column
            df['month'] = (df.fecha_dato)%12 + 1
            df.month = df.month.astype('category')
        #Get column groups
        df = self.eval_previous
        change_columns = [name for name in df.columns if 'change' in name]
        product_columns = [name for name in df.columns
            if 'ult1' in name and 'change' not in name]
        df = self.eval_current
        categorical_columns = list( df.select_dtypes(
                                include=['category']).columns)
        print categorical_columns
        #Create translation dictionary
        text = '{\n'
        df = self.eval_current
        for key in categorical_columns:
            text += '\t"%s":{' % key
            for i, category in enumerate(df[key].unique()):
                text += '"%s": %s,' % (category, i)
            text += ' },\n'
        df = self.eval_previous
        for key in change_columns:
            text += '\t"%s":{' % key
            for i, category in enumerate(df[key].unique()):
                text += '"%s": %s,' % (category, i)
            text += ' },\n'
        text += '}\n'
        translation_dict = eval(text)
        #Use the dictionary for translation
        for df in [self.train_current, self.eval_current]:
            for key in categorical_columns:
                if verbose:
                    print(key)
                df[key].cat.categories = [translation_dict[key][str(category)]
                                        for category in  df[key].cat.categories]
        for df in [self.train_previous, self.eval_previous]:
            for key in change_columns:
                if verbose:
                    print(key)
                df[key].cat.categories = [translation_dict[key][str(category)]
                                        for category in  df[key].cat.categories]
        #Transform new_products column to a list
        df = self.eval_current
        df.new_products = df.new_products.apply(literal_eval)
        #Save some data for later
        self.change_columns = change_columns # does not include lag
        self.product_columns = product_columns # does not include lag
        self.categorical_columns = categorical_columns # includes lag
        self.translation_dict = translation_dict # include change_columns and categorical_columns


    def __gbm_encoded_data(self, X, y = None, params = None, config = "train", 
                           path = "../gbm_model/model.txt", onehot = True):
        """
        Private method that uses gbm encoder for
        transforming the required data

        df: pandas dataframe
        input_columns: list with the names of the columns to use
        """
        self._gbm_params =  {"num_iteration": 8, 
                             "n_estimators" : 30, 
                             "max_depth" : 5, 
                             "eval_metrics":"multi_logloss"}
        
        if config == "train":
            if params is None: 
                params = self._gbm_params
            lgb_model = lgb.LGBMClassifier(n_estimators=params["n_estimators"], max_depth = params["max_depth"])
            lgb_model.fit(X, y, eval_metric=params["eval_metrics"], 
                                verbose = False)
            lgb_model.booster().save_model(path)
            features = lgb_model.apply(X, num_iteration= params["num_iteration"])
            #return features
            
        if config == "eval" :
            booster = lgb.Booster(model_file=path)
            print "X shape", X.shape
            features = booster.predict(X, pred_leaf=True, num_iteration=8)
            #return features

        if onehot:
            print "max val in features is ", np.max(features)
            #n_values = 2 ** self._gbm_params['max_depth'] 
            n_values = np.max(features) + 1
            enc = preprocessing.OneHotEncoder(n_values=n_values,\
                                          sparse=False, dtype=np.uint8)
            enc.fit(features)
            encoded_features = enc.transform(features)
            return encoded_features
        else:
            return features

    def __get_encoded_data(self, df, input_columns, option = "default"):
        """
        Private method that uses one hot encoder for
        transforming the required data

        df: pandas dataframe
        input_columns: list with the names of the columns to use
        """
        #Get parameters for the encoder
        n_values = [len(self.translation_dict[key].values())
                    for key in input_columns]
        #Create the encoder
        enc = preprocessing.OneHotEncoder(n_values=n_values,
                                          sparse=False, dtype=np.uint8)
        #Fit the encoder
        enc.fit(df[input_columns].values)
        #Transform the data
        encoded_data = enc.transform(df[input_columns].values)
        return encoded_data

    def __gen_levels(self, feat_group):
    	"""
		return number of levels in feature defined in feat_group
	
    	"""

    	value_list = []
    	for key in feat_group:
    		if 'ult1' in key:
    			if '_change' in key:
    				value = 3
    			else:
    				value = 2
    		else:
    			value = len(self.translation_dict[key].values())
    		value_list.append(value)
    	return value_list

    def __get_interact_data(self, df, interact_columns):
        """
        Private method that includes all interactions to expand feature space
    
        df: pandas dataframe
        interact_columns: list[list]: groups of features that have local interaction, for example [[A,B],[A,C,E]] give two
        interact groups, the two-way interaction of [A,B] and three-way interaction among [A,C,E]

        Note: this method must come BEFORE one-hot encoding as we need to use raw feature values
        """
        #feat_interact_names = [] # the names of all interact feature groups
        n_values = []
        arr = None
        n = df.shape[0]
        for feat_group in interact_columns:
            # Append the total possible levels of feat_group for one-hot encoding

            #for key in feat_group:
            #	print key, self.translation_dict[key], df.loc[:,key].unique()      
            value_list = self.__gen_levels(feat_group)
            n_values.append(np.prod(value_list))
            # Think as a free-base number representation, idx gives the coordinates (n by 1) of each combined feature in one-hot matrix
            nbases = np.cumprod([1] + value_list)[:-1]
            idx = df[feat_group].apply(lambda row: np.inner(nbases, \
                                [int(row[key]) for key in feat_group]).astype(int), axis = 1)
            if arr is None:
                arr = idx.copy().values.reshape(n,1)
            else:
                arr = np.concatenate((arr, idx.values.reshape(n,1)), axis = 1)
        
        enc = preprocessing.OneHotEncoder(n_values=n_values,\
                                            sparse=False, dtype=np.uint8)
        enc.fit(arr)        
        interact_data = enc.transform(arr)
        return interact_data
            
    def __get_sequence_data(self, df, sequence_columns):
        """
        Private method to include sequence features

        A special case of interaction features, for example, [ft, ft_L2, ft_L4] interaction gives a sequence
        feature ft for time 0, -2, -4
        """
        seq_data = self.__get_interact_data(df, sequence_columns)
        return seq_data

    def check_data_sanity(self):
        """
        Public method that check if the product features we are using are corrected generated

        Basically, compare the product feature values in self.train_previous and product_L1 feature values in self.train_current
        For example, the product features (ind_ahor_fin_ult1, for example) in self.train_previous[with fecha_dato == 3] 
        should match those of Lag 1 (ind_ahor_fin_ult1_L1) in self.train_current[with fecha_dato == 4]

        Check for all months (fecha_dato) in these dataset
        Raise an alert if for any month these two sets of product features does not match
        """
        for mon in range(16):
            # Product feature data match check
            prod_lag1 = [col + '_L' + str(1) for col in feat_prod]
            print self.train_current.loc[self.train_current.fecha_dato == mon + 1, prod_lag1].values.shape
            print self.train_previous.loc[self.train_previous.fecha_dato == mon, feat_prod].values.shape
            count_not_eq = np.sum(np.not_equal(self.train_current.loc[self.train_current.fecha_dato == mon + 1, prod_lag1].values.astype(int), \
                self.train_previous.loc[self.train_previous.fecha_dato == mon, feat_prod].values.astype(int)))
            print "count_not_eq for month: ", mon, " is, ", count_not_eq

            # Status change data match check
            col_current = [x + '_L' + str(1) for x in feat_prod]
            col_prev = [x + '_L' + str(2) for x in feat_prod]
            status_change_data_L1 = np.not_equal(self.train_current.loc[self.train_current.fecha_dato == mon + 1, col_current].values.astype(int), \
                                    self.train_current.loc[self.train_current.fecha_dato == mon + 1, col_prev].values.astype(int)).astype(int)

            col_change = [x + '_change' for x in feat_prod]
            status_change_data = self.train_previous.loc[self.train_previous.fecha_dato == mon, col_change].values.astype(int)
            print "mismatch for prod change for month: ", mon, "is, ", np.sum(np.greater(status_change_data,status_change_data_L1))

    def __get_feature_status_change_data(self, df, feature_columns, lags):
        """
        Private method that generate status change data for non-product user profile features
        
        df: pandas dataframe
        feature_columns: the selected profile feature to be used that we should consider status change
        if lag == 0, then stand on current month feature

        Note: this function must be applied after digitalized categorical features and before one-hot encoding
        """
        status_change_data = None
        for lag in lags:
            print lag
            if lag == 0:
                col_current = feature_columns
            else:
                col_current = [x + '_L' + str(lag) for x in feature_columns]
            col_prev = [x + '_L' + str(lag + 1) for x in feature_columns]
            if status_change_data is None:
                # This actually defines the OPPOSITE to change! I tried switch to np.not_equal... but it seems np.equal gives higher LB, very interesting
                status_change_data = np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)
            else:
                status_change_data = np.concatenate((status_change_data, \
                    np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)), axis = 1)
                
        return status_change_data

    def __get_product_status_change_data(self, df, lags):
        """
        Private method that includes product feature status change comparing cur vs. prev month

        df: pandas dataframe    
        lags: a list of lags that we use to compute status change

        prod_change 1/0, prod_add 1/0 prod_drop 1/0 prod_maintain 1/0
        """

        status_change_data = None
        for lag in lags:
            if lag == 0:
                continue
            col_current = [x + '_L' + str(lag) for x in self.product_columns]
            col_prev = [x + '_L' + str(lag + 1) for x in self.product_columns]  
            if status_change_data is None:
                # This defines the OPPOSITE to change! I tried switch to np.not_equal... but it seems np.equal gives higher LB, very interesting
                status_change_data = np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)
                status_change_data = np.concatenate((status_change_data, \
                    np.greater(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)), axis = 1)
                status_change_data = np.concatenate((status_change_data, \
                    np.greater(df[col_prev].values.astype(int), df[col_current].values.astype(int)).astype(int)), axis = 1)
            else:
                status_change_data = np.concatenate((status_change_data, \
                    np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)), axis = 1) # change
                status_change_data = np.concatenate((status_change_data, \
                    np.greater(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)), axis = 1) # add 
                status_change_data = np.concatenate((status_change_data, \
                    np.greater(df[col_prev].values.astype(int), df[col_current].values.astype(int)).astype(int)), axis = 1) # drop

        return status_change_data

    def __get_data_aux(self, msg):
        """
        Auxiliary method for get_data
        It handles when there is more than one month requested
        """
        #Loop over the required months
        data = [None, None, None, None] # input_data, output_data, previous_products, user_ids
        for month in msg['month']:
            print "collecting month ... ... ", month
            msg_copy = msg.copy()
            msg_copy['month'] = month
            ret = self.get_data(msg_copy)
            for i in range(4): 
                #Aggregate with the data of other months if necessary
                if data[i] is None:
                    data[i] = ret[i]
                else:
                    #print "data[i], ", data[i].shape, " ret[i], ", ret[i].shape
                    data[i] = np.concatenate((data[i], ret[i]), axis=0)
        return data 

    def get_data(self, msg, verbose=False):
        """
        Returns the data needed for training given the specified parameters

        The input is a message with the given fields

        month: int or list with the number of the month we want
            the data to be taken of
        train: bool, if true uses training dataset otherwise uses eval dataset
        input_columns: a list with the name of the columns we are going to use
            in the task
        use_product: bool, if true adds the product columns of the month before
        use_change: bool, if true adds the change columns of the month before
        """
        #TODO: I'm not filtering by month
        #TODO: The function for eval data will be very similar, try to reuse
        if verbose:
            print(msg)
        #If we have more than one month return aux function
        if type(msg['month']) is list:
            return self.__get_data_aux(msg)
        #Select the datasets we will be using
        if msg['train']:
            df_current = self.train_current[
                self.train_current.fecha_dato == msg['month']]
            df_previous = self.train_previous[
                self.train_previous.fecha_dato == msg['month']-1]            
        else:
            #Then eval datasets are used
            df_current = self.eval_current[
                self.eval_current.fecha_dato == msg['month']]
            df_previous = self.eval_previous[
                self.eval_previous.fecha_dato == msg['month']-1]
        

        #Collect the required categorical data from current dataset
        input_data = None
        input_columns = msg['input_columns']
        if len(input_columns) > 0:
            #Get parameters for the encoder
            input_data = self.__get_encoded_data(df_current,
                                                 input_columns)
        #Collect the required data from previous dataset
        #Add product columns if necesary, product columns are binary
        if msg['use_product']:
            product_data = df_previous[self.product_columns].values
            if input_data is None:
                input_data = product_data
            else:
                #Join the matrixes
                if verbose:
                    print(input_data.shape, product_data.shape)
                input_data = np.concatenate((input_data, product_data),
                                            axis=1)
        #Add change columns if necesary
        if msg['use_change']:
            change_data = self.__get_encoded_data(df_previous,
                                                     self.change_columns)
            if input_data is None:
                input_data = change_data
            else:
                #Join the matrixes
                if verbose:
                    print(input_data.shape, change_data.shape)
                input_data = np.concatenate((input_data, change_data),
                                            axis=1)
        #Add interaction data if necessary
        if msg['input_columns_interactions']:
            print "processing feature interactions ...... "
            interact_data = self.__get_interact_data(df_current, msg['input_columns_interactions'])

            if input_data is None:
                input_data = interact_data
            else:
                if verbose:
                    print(input_data.shape, interact_data.shape)
                input_data = np.concatenate((input_data, interact_data),
                                            axis=1)
        #Add time series sequence data 
        if msg['sequence_columns']:
            print "add sequence data ....... "
            sequence_data = self.__get_sequence_data(df_current, msg['sequence_columns'])
            if input_data is None:
                input_data = sequence_data
            else:
                if verbose:
                    print(input_data.shape, sequence_data.shape)
                input_data = np.concatenate((input_data, sequence_data),
                                            axis=1)

        # add lagged product features if necessary
        if msg['use_product_lags']:
            product_columns_lag = get_feat_prod_lag(msg['use_product_lags'])
            product_data_lag = df_current[product_columns_lag].values       
            if input_data is None:
                input_data = product_data_lag
            else:
                #Join the matrixes
                if verbose:
                    print(input_data.shape, product_data_lag.shape)
                input_data = np.concatenate((input_data, product_data_lag),
                                            axis=1)

        # add lagged profile features if necessary
        if msg['use_profile_lags']:
            profile_columns_lag = get_feat_lag(msg['input_columns_lags'], msg['use_profile_lags'])
            profile_data_lag = self.__get_encoded_data(df_current,\
                                                        profile_columns_lag)
            if input_data is None:
                input_data = profile_data_lag
            else:
                if verbose:
                    print(input_data.shape, profile_data_lag.shape)
                input_data = np.concatenate((input_data, profile_data_lag),
                                            axis=1)

        # add status change for customer profile features
        if msg['input_columns_change']:
            print "adding profile status change features................"
            profile_data_change_lag = self.__get_feature_status_change_data(df_current, \
                                            msg['input_columns_change'], msg['use_profile_change_lags'])
            if input_data is None:
                input_data = profile_data_change_lag
            else:
                if verbose:
                    print(input_data.shape, profile_data_change_lag.shape)
                input_data = np.concatenate((input_data, profile_data_change_lag), axis = 1)

        # add status change for customer product buyings
        if msg['use_product_change_lags']:
            print "adding product status change features................"
            product_data_change_lag = self.__get_product_status_change_data(df_current, msg['use_product_change_lags'])
            if input_data is None:
                input_data = product_data_change_lag
            else:
                if verbose:
                    print(input_data.shape, product_data_change_lag.shape)
                input_data = np.concatenate((input_data, product_data_change_lag), axis = 1)
                    
        #Now collect the output data
        if msg['train']:
            output_data = df_current.buy_class.values
        else:
            output_data = df_current.new_products.values
        
        #Collect previous products data        
        if msg['train']:
            previous_products = None
        else:
            previous_products = df_previous[self.product_columns].values

        # Keep track of user_ids with np array data for train/val split
        user_ids = df_current.ncodpers.values

        return input_data, output_data, previous_products, user_ids

    def __get_train_val_test_data_aux(self, msg, istrain, months):
        """
        Private methods that takes msg and request, return input data, output data and previous_products, if any

        """
        msg_copy = msg.copy()
        msg_copy['train'] = istrain 
        msg_copy['month'] = months
        ret = self.get_data(msg_copy)
        return ret

    def get_train_val_split_data(self, msg, r_val = 0.1):
        """
        Split train and val by user id

        r_val: the percentage for validation
        """

        #Get entire train data 
        print("Read train data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = True, \
                months = msg['train_month'])
        self.train_data, self.train_label = ret[0:2]
        # get train user id including duplicate
        user_ids_train = ret[3] 

        #Get entire eval data, Note we still use train_month due to different validation logic
        print("Read val data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = False, \
                months = msg['train_month'])
        self.val_data, self.val_label, self.val_prev_prod, user_ids_val = ret

        #Get unique user ids
        unique_ids = np.unique(user_ids_val)
        num_unique_ids = len(unique_ids)
        
        #Generate mapping table for unique user ids, to 1(val) or 0(train)
        np.random.seed(seed = 2016) # Fix seed 
        sample = np.random.binomial(1, r_val, size=num_unique_ids) 
        table_ref = np.concatenate((unique_ids.reshape(num_unique_ids, 1), \
                        sample.reshape(num_unique_ids, 1)), axis = 1)
        df_ref = pd.DataFrame(table_ref, columns = ['ncodpers', 'isval'])

        #Left join to create train and val split
        df_user_ids_train = pd.DataFrame(user_ids_train, columns = ['ncodpers'])
        df_user_ids_train = df_user_ids_train.merge(df_ref, how = 'left', left_on = 'ncodpers',\
                                                        right_on = 'ncodpers', suffixes = ['', '_ref'])

        df_user_ids_val = pd.DataFrame(user_ids_val, columns = ['ncodpers'])
        df_user_ids_val = df_user_ids_val.merge(df_ref, how = 'left', left_on = 'ncodpers',\
                                                        right_on = 'ncodpers', suffixes = ['', '_ref'])

        #Get train/val indices and split train/val 
        idx_train = np.where(df_user_ids_train.isval.values == 0)[0] # idx_train must be within train set
        idx_val = np.where(df_user_ids_val.isval.values == 1)[0] # idx_val must be within val set

        self.X_tr = self.train_data[idx_train ,:]
        self.y_tr = self.train_label[idx_train]

        self.X_val = self.val_data[idx_val,:]
        self.y_val = self.val_label[idx_val]
        self.p_val = self.val_prev_prod[idx_val,:] # remember p_val is used to keep track of previous_product, when evaluation


    def get_test_data(self, msg):
        """
        Read test data

        """
        print("Read test data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = False, \
                months = 17)
        self.test_data, self.test_label, self.test_prev_prod = ret[:3]
        print "test data size, ", self.test_data.shape      

    def get_train_val_test_data(self, msg):
        """
        Get train and val and test data

        train data contains months in msg['train_month'] 
        val data contains months in msg['eval_month'] 
        test data contains month 17 (2016-06-28) 
        """
        start_time = time.time()

        print("Read train data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = True, \
                months = [x for x in msg['train_month'] if x not in msg['eval_month']])
        self.train_data_tr, self.train_label_tr = ret[0:2]
        print "train data size, ", self.train_data_tr.shape, self.train_label_tr.shape

        print("Read validation part of train data, for production use both train and val for test")
        ret = self.__get_train_val_test_data_aux(msg, istrain = True, \
                months = [x for x in msg['train_month'] if x in msg['eval_month']])
        self.train_data_val, self.train_label_val = ret[0:2]
        print "train data size, ", self.train_data_val.shape, self.train_label_val.shape

        print("Read val data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = False, \
                months = msg['eval_month'])
        self.val_data, self.val_label, self.val_prev_prod = ret
        print "val data size, ", self.val_data.shape

        print("Read test data")
        ret = self.__get_train_val_test_data_aux(msg, istrain = False, \
                months = 17)
        self.test_data, self.test_label, self.test_prev_prod = ret
        print "test data size, ", self.test_data.shape
        

        print('It took %i seconds to process the dataset' % (time.time()-start_time))

if __name__ == "__main__":
    dataset_root = '../'
    dataset = SantanderDataset(dataset_root, isLag = True, lags = [1,2,3,4,5])
    #dataset.check_data_sanity()




