"""feature_extraction.py
Compute requested features output dataset
"""

import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from numpy import dot, array

def create_features_nn(data_query, data_train, month, nn_used, nn_distance_lag, nn_summary_lag, nn_search_type, \
										istrain = True):
	"""
	Compute nearest neighbor summary features 
	data_train data_test: train/test data
	month: current month
	nn_used: the selected nn number for feature summary
	nn_distance_var: features selected to compute distance in knn
	nn_summary_var: features that are to be summarized from nn
	nn_search_type: where to search nearest neighbors
	istrain: does data_query belong to train set or test set

	return: numpy array [num rows in data_query] - by - [len(nn_summary_var) * len(nn_used)]
	"""				
	nn_distance_var = ['renta','age','indrel','indrel_1mes','indext','segmento']

	################### prepare database and configs #######################
	# determine search space
	if istrain:			
		if nn_search_type == 'prev-mon':
			if month > 1:
				month_lag = 1					
			elif month == 1:
				month_lag = 0					
		elif nn_search_type == 'cur-mon':
			month_lag = 0				
	else:
		if nn_search_type == 'prev-mon':
			month_lag = 1				
		else:
			month_lag = None	
			
	if month_lag == 1:
		nn_distance_var_query = nn_distance_var + [col + '_L' + str(lag) for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
		nn_distance_var_db = nn_distance_var + [col + '_L' + str(lag-1) if lag > 1 else col for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
		nn_summary_var_db = [col + '_L' + str(lag-1) if lag > 1 else col for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
	elif month_lag == 0:
		nn_distance_var_query = nn_distance_var + [col + '_L' + str(lag) for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
		nn_distance_var_db = nn_distance_var + [col + '_L' + str(lag) for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
		nn_summary_var_db = [col + '_L' + str(lag) for col in FEATNAME.COLUMNS["product"] for lag in nn_distance_lag]
	else:
		nn_distance_var_query = None
		nn_distance_var_db = None
		nn_summary_var_db = None


	# read candidate database
	df_data_train = data_train.drop_duplicates(['fecha_dato','ncodpers']).copy()

	# Database where we find neighbors of Query dataset
	df_db = df_data_train.loc[df_data_train.fecha_dato == month - month_lag, :].reset_index()	
	X_db = df_db.loc[:, nn_distance_var_db].values

	# initialize fit knn object, get nn ids
	knn = NearestNeighbors(n_neighbors=max(nn_used), metric = 'hamming')	
	knn.fit(X_db) 
	X_query = data_query.loc[:, nn_distance_var_query].values			
	# Return n_neighbors row ids in X_db which are nn's of the rows in X_query
	neigh_ids = knn.kneighbors(X_query, return_distance=False) # size df row -by- max(nn_used)

	################################# start collecting summary features ####################							
	nn_data = None # the return object
	encoded_features = [] # the return feature name
	# matrix multiplication
	n_values = df_db.shape[0]
	enc = preprocessing.OneHotEncoder(n_values=n_values,\
										  sparse=False, dtype=np.uint8)
	for nn_num in nn_used:
		neigh_ids_selected = neigh_ids[:, :nn_num].copy() # [nrow data_query] -by- [nn_num]
		# trainsform neigh_ids into binary index
		neigh_ids_transpose = np.transpose(neigh_ids_selected) # [nn_num] -by- [nrow data_query]					
		enc.fit(neigh_ids_transpose)
		encoded_neigh_ids = enc.transform(neigh_ids_transpose) # [nrow df_db] -by- [nrow data_query]
		neigh_ids_selected = np.transpose(encoded_neigh_ids) # [nrow data_query] -by- [nrow df_db]
		# iterate through each nn num option	
		mean_feature_vals = np.dot(neigh_ids_selected, df_db.loc[:, nn_summary_var_db].values) #[nrow data_query] -by- [len nn_summary_var]
		if nn_data is None:
			nn_data = mean_feature_vals
		else:
			nn_data = np.concatenate((nn_data, mean_feature_vals), axis = 1)
		
		encoded_features.extend([col + '_nn_mean_' + str(nn_num) for col in nn_summary_var_db])

	return nn_data, encoded_features

def create_features_gbm_hash(X, y = None, params = None, config = "train", 
					   path = "../gbm_model/model.txt", onehot = True):
	"""
	Private method that uses gbm encoder for
	transforming the required data

	df: pandas dataframe
	input_columns: list with the names of the columns to use
	"""
	gbm_params =  {"num_iteration": 8, 
						 "n_estimators" : 30, 
						 "max_depth" : 5, 
						 "eval_metrics":"multi_logloss"}
	
	if config == "train":
		if params is None: 
			params = gbm_params
		lgb_model = lgb.LGBMClassifier(n_estimators=params["n_estimators"], max_depth = params["max_depth"])
		lgb_model.fit(X, y, eval_metric=params["eval_metrics"], 
							verbose = False)
		lgb_model.booster().save_model(path)
		features = lgb_model.apply(X, num_iteration= params["num_iteration"])
		#return features
		
	if config == "eval" :
		booster = lgb.Booster(model_file=path)
		features = booster.predict(X, pred_leaf=True, num_iteration=8)

	if onehot:
		n_values = np.max(features) + 1
		enc = preprocessing.OneHotEncoder(n_values=n_values,\
									  sparse=False, dtype=np.uint8)
		enc.fit(features)
		encoded_features = enc.transform(features)
		return encoded_features
	else:
		return features

def gen_levels(feat_group, translation_dict):
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
			value = len(translation_dict[key].values())
		value_list.append(value)
	return value_list

def gen_levels_max(feat_group, translation_dict):
	"""
	return max val of levels in feature defined in feat_group

	"""
	value_list = []
	for key in feat_group:
		if 'ult1' in key:
			if '_change' in key:
				value = 3
			else:
				value = 2
		else:
			value = max(translation_dict[key].values()) + 1
		value_list.append(value)
	return value_list

def create_features_onehot_encode(df, columns, translation_dict):
	"""
	Private method that uses one hot encoder for
	transforming the required data

	df: pandas dataframe
	columns: list with the names of the columns to use
	"""
	#Get parameters for the encoder

	#n_values = [len(self.translation_dict[key].values())
	#			for key in columns]
	n_values = gen_levels(columns, translation_dict)
	
	encoded_features = [] # field-value pair corresponding to one-hot encoding
	for key in columns:
		encoded_features.extend([key + '-' + str(x) for x in sorted(translation_dict[key].values())])

	#Create the encoder
	enc = preprocessing.OneHotEncoder(n_values=n_values,
									  sparse=False, dtype=np.uint8)
	#Fit the encoder
	enc.fit(df[columns].values.astype(int))
	#Transform the data
	encoded_data = enc.transform(df[columns].values.astype(int))
	return encoded_data, encoded_features	

def create_features_products(df, product_lags, product_lags_cummax):
	"""
	create product related features, lags, combined lags, and cummax lags

	product_lags: [1, 2, [3 14], ...]
	if individual, then include individual column, if list, then include bitwise OR (maximum) combined feature		
	
	product_lags_cummax: compute cumulative max product from far before to current month
	return product_data, encoded_features [p1, p2, p3-p14] ...
	"""
	encoded_features = []
	arr = None

	lags_individual = [x for x in product_lags if type(x) is not list]
	lags_combine = [x for x in product_lags if type(x) is list]
	n = df.shape[0]
	if lags_individual:
		# the lags are like [1,2,3,4, ... ]
		cols = [str(x) + '_L' + str(lag) for x in FEATNAME.COLUMNS["product"] for lag in lags_individual]
		encoded_features.extend(cols)
		arr = df[cols].values  

	if lags_combine:
		# the lags are like [[1,2],[3,4], ... ] now we assume each combine group only has 2 lags
		cols_current = [str(x) + '_L' + str(y[0]) for x in FEATNAME.COLUMNS["product"] for y in lags_combine] 
		cols_previous = [str(x) + '_L' + str(y[1]) for x in FEATNAME.COLUMNS["product"] for y in lags_combine] 	
		encoded_features.extend([str(x) + '-' + str(y) for x, y in zip(cols_current, cols_previous)])

		if arr is None:
			arr = np.maximum(df[cols_current].values, df[cols_previous].values).astype(int)
		else:
			arr = np.concatenate( (arr, \
					np.maximum(df[cols_current].values, df[cols_previous].values).astype(int)), axis = 1)

	if product_lags_cummax:
		# add cum-max product features
		data_current = None
		for lag in product_lags_cummax[::-1]:
			cols_current = [str(x) + '_L' + str(lag) for x in FEATNAME.COLUMNS["product"]] 
			if data_current is None:
				data_current = df[cols_current].values
			else:
				data_current = np.maximum(data_current, df[cols_current].values).astype(int)
			if arr is None:
				arr = data_current
			else:
				arr = np.concatenate((arr, data_current), axis = 1)
			encoded_features.extend([col + '_cummax' for col in cols_current])			

	return arr, encoded_features


def create_features_tensor_combine(df, interact_columns, translation_dict):
	"""
	Create combined features using tensor products

	df: pandas dataframe
	interact_columns: list[list]: groups of features that have local interaction, for example [[A,B],[A,C,E]] give two
	interact groups, the two-way interaction of [A,B] and three-way interaction among [A,C,E]

	Note: this method must come BEFORE one-hot encoding as we need to use raw feature values
	"""
	#feat_interact_names = [] # the names of all interact feature groups
	n_values = []
	arr = None
	n = df.shape[0]
	encoded_features = []
	for feat_group in interact_columns:
		# Append the total possible levels of feat_group for one-hot encoding
   
		value_list = gen_levels(feat_group, translation_dict)
		n_values.append(np.prod(value_list))

		encoded_features.extend(['-'.join(feat_group + [str(x)]) for x in range(np.prod(value_list))])
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

	return interact_data, encoded_features

def create_features_interaction(df, mult_interact_columns, translation_dict):
	"""
	Create Feature Interactions on by multiplication 
	Take the product of interaction groups BEFORE one-hot encoding, then do one-hot encoding

	"""
	n_values = []
	arr = None
	n = df.shape[0]
	encoded_features = []
	for feat_group in mult_interact_columns:
		# Append the total possible levels of feat_group for one-hot encoding
     
		value_list = gen_levels_max(feat_group, translation_dict)
		n_values.append(np.prod(value_list))

		encoded_features.extend(['-mult-'.join(feat_group + [str(x)]) for x in range(np.prod(value_list))])
		# Think as a free-base number representation, idx gives the coordinates (n by 1) of each combined feature in one-hot matrix
		idx = df[feat_group].apply(lambda row: np.prod([int(row[key]) for key in feat_group]).astype(int), axis = 1)

		if arr is None:
			arr = idx.copy().values.reshape(n,1)
		else:
			arr = np.concatenate((arr, idx.values.reshape(n,1)), axis = 1)
	
	enc = preprocessing.OneHotEncoder(n_values=n_values,\
										sparse=False, dtype=np.uint8)
	enc.fit(arr)        
	interact_data = enc.transform(arr)

	return interact_data, encoded_features

def create_features_sequence(df, sequence_columns, translation_dict):
	"""
	Create sequence features in time 

	A special case of interaction features, for example, [ft, ft_L2, ft_L4] interaction gives a sequence
	feature ft for time 0, -2, -4
	"""
	seq_data, encoded_features = create_features_interaction(df, sequence_columns, translation_dict)
	return seq_data, encoded_features

def create_features_profile_change(df, feature_columns, lags):
	"""
	Private method that generate status change data for non-product user profile features
	
	df: pandas dataframe
	feature_columns: the selected profile feature to be used that we should consider status change
	if lag == 0, then stand on current month feature

	Note: this function must be applied after digitalized categorical features and before one-hot encoding
	"""
	status_change_data = None
	encoded_features = []
	for lag in lags:
		if lag == 0:
			col_current = feature_columns
		else:
			col_current = [x + '_L' + str(lag) for x in feature_columns]
		col_prev = [x + '_L' + str(lag + 1) for x in feature_columns]

		encoded_features.extend([x + '_change' for x in col_current])
		if status_change_data is None:
			# This actually defines the OPPOSITE to change! I tried switch to np.not_equal... but it seems np.equal gives higher LB, very interesting
			status_change_data = np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)
		else:
			status_change_data = np.concatenate((status_change_data, \
				np.equal(df[col_current].values.astype(int), df[col_prev].values.astype(int)).astype(int)), axis = 1)
			
	return status_change_data, encoded_features

def create_features_product_change(df, product_columns, lags, use = 'user-based'):
	"""
	Private method that includes product feature status change comparing cur vs. prev month

	df: pandas dataframe    
	lags: a list of lags that we use to compute status change
	user_based: bool. If True, then use the _change feature directly from df, else means "month based", calculate
	consecutive month difference of product features

	prod_change 1/0, prod_add 1/0 prod_drop 1/0 prod_maintain 1/0
	"""
	if use == 'user-based':
		col_change = [x + '_change_L' + str(lag) for x in product_columns for lag in lags]
		status_change_data, encoded_features = create_features_onehot_encode(df, input_columns = col_change)
		return status_change_data, encoded_features
	elif use == 'month-based':
		status_change_data = None
		encoded_features = []
		for lag in lags:
			if lag == 0:
				continue
			col_current = [x + '_L' + str(lag) for x in product_columns]
			col_prev = [x + '_L' + str(lag + 1) for x in product_columns]  

			encoded_features.extend([x + '_maintained' for x in col_current])
			encoded_features.extend([x + '_added' for x in col_current])
			encoded_features.extend([x + '_dropped' for x in col_current])

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
		return status_change_data, encoded_features

	else: # use both 
		status_change_data_1, encoded_features_1 = create_features_product_change(df, columns, lags, use = 'user-based')
		status_change_data_2, encoded_features_2 = create_features_product_change(df, columns, lags, use = 'month-based')
		encoded_features = encoded_features_1 + encoded_features_2
		status_change_data = np.concatenate((status_change_data_1, status_change_data_2), axis = 1)

		return status_change_data, encoded_features









