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

    def __init__(self, dataset_root, isLag = False):
        """
        Loads the dataset
        """
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

    def __get_interact_data(self, df, interact_columns):
        """
        Private method that includes all interactions to expand feature space
    
        df: pandas dataframe
        interact_columns: list[list]: groups of features that have local interaction

        Note: this method must come BEFORE one-hot encoding
        """
        feat_interact_names = [] # the names of all interact feature groups
        n_values = []
        for feat_group in interact_columns:
            feat_group_name = '-'.join(feat_group) # the interaction feature name 
            feat_interact_names.append(feat_group_name)
            n_values.append(np.prod([len(self.translation_dict[key].values())\
                                for key in feat_group]))
            # concatenate feature into 'A-B-C'
            df[feat_group_name] = \
                df.apply(lambda row: '-'.join([str(int(row[x])) if isInt(row[x]) else \
                            str(row[x]) for x in feat_group ]), axis = 1)
            df[feat_group_name] = pd.factorize(df[feat_group_name])[0]

        # one-hot encoding 

        enc = preprocessing.OneHotEncoder(n_values=n_values,\
                                            sparse=False, dtype=np.uint8)
        enc.fit(df[feat_interact_names].values)        
        interact_data = enc.transform(df[feat_interact_names].values)

        return interact_data

    def __get_sequence_data(self, df, feature_seqs):
        """
        Private method that generate all sequence features and one-hot. The procedure should be similar to __get_interact_data() 

        df: pandas dataframe
        feature_seqs : dict(), key : feature name, value : sequence 
        feature_seqs: dict(), feature_seqs[feature_name] = [[lag group 1], [lag group 2], ... ]
            for example, feature_seqs[renta] = [[0,2,4],[3,5,9]] means we want to include features "renta-renta_L2-renta_L4" 
            and "renta_L3-renta_L5-renta_L9". The feature values are first combined using '-' and then one-hotted, similar to 
            __get_interact_data() 

        return: sequence_data numpy.array with shape[df.shape[0], number of added sequence features after one-hot]
        """
        feat_seqs_name = []
        for name, seqs in feature_seqs.iteritems():
            seqs_name_tmp = map(lambda x : name + "_L" +str(x) if x > 0 else "", seqs) # map [0, 2, 4] -> [renta_L0, renta_L2, renta_L4]
            seqs_name = filter(lambda x : x != "", seqs_name_tmp) # filter out renta_L0
            feat_seqs_name.append("-".join(seqs_name))

        df[feat_seqs_name]

        return 0


    def check_data_sanity(self):
        """
        Public method that check if the product features we are using are corrected generated

        Basically, compare the product feature values in self.train_previous and product_L1 feature values in self.train_current
        For example, the product features (ind_ahor_fin_ult1, for example) in self.train_previous[with fecha_dato == 3] 
        should match those of Lag 1 (ind_ahor_fin_ult1_L1) in self.train_current[with fecha_dato == 4]

        Check for all months (fecha_dato) in these dataset
        Raise an alert if for any month these two sets of product features does not match
        """

        return 0

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
        data = [None, None, None]
        for month in msg['month']:
            print "collecting month ... ... ", month
            msg_copy = msg.copy()
            msg_copy['month'] = month
            ret = self.get_data(msg_copy)
            for i in range(3):
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

        return input_data, output_data, previous_products

    def __get_train_val_test_data_aux(self, msg, istrain, months):
        """
        Private methods that takes msg and request, return input data, output data and previous_products, if any

        """
        msg_copy = msg.copy()
        msg_copy['train'] = istrain 
        msg_copy['month'] = months
        ret = self.get_data(msg_copy)
        return ret

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

