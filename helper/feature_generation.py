

import numpy as np
from feature_names import FEATNAME

from feature_engineer import *
from data_process import *


def append_feature_data(input_data, new_data):
	input_data = np.concatenate((input_data, data_feature_current), axis=1) \
		if input_data != None else data_feature_current
	return input_data

def create_data_one_month(feature_msg, translation_dict, \
					train_current, train_previous, test_current, test_previous, \
                    return_labels = True, return_prev_prod = True, return_user_id = True, return_new_products = True, \
                    month = None, is_train = True):
	
	"""
	create feature dataset for train OR test for only one month

	input:
		feature_msg: specified feature set and selected months
		translation_dict: the mapping dict for categorical features
		data_current_one_month, data_previous_one_month: data frames, could be train OR test for only month == month
			Note that 
			data_current_one_month = data_current[month == month]
			but
			data_previous_one_month = data_previous[month == month-1]
		return_labels: bool, whether to return label
		return_prev_prod: bool, whether to return previous product
		return_user_id: bool, whether to return user ids
		return_new_products: bool, whether to return list of new products
		month: the current month, must be >= 1

	output:
		label, data, previous_products, user_ids, new_products
	"""

	if is_train:
		data_current_one_month = train_current[train_current.fecha_dato == month]
		data_previous_one_month = train_previous[train_previous.fecha_dato == month-1]            
	else:		
		data_current_one_month = test_current[test_current.fecha_dato == month]
		data_previous_one_month = test_previous[test_previous.fecha_dato == month-1]

	# Start collecting features into input_data
	input_data = None

	# profile categorical features 
	if feature_msg['features_profile_cate']:
		new_data, _ = create_features_onehot_encode(data_current_one_month, \
				feature_msg['features_profile'], translation_dict)
		input_data = append_feature_data(input_data, new_data)		

	# profile numerical features
	if feature_msg['features_profile_num']:
		new_data = data_current_one_month.loc[:, feature_msg['numerical_features']].values
		input_data = append_feature_data(input_data, new_data)	
	
	# product feature
	if feature_msg['features_product_yn']:
		new_data = data_previous_one_month[FEATNAME.COLUMNS["product"]].values
		input_data = append_feature_data(input_data, new_data)	

	#product change at this month vs. previous month
	if feature_msg['features_product_change_yn']:
		new_data, _ = create_features_onehot_encode(data_current_one_month, \
				FEATNAME.COLUMNS["product_change"]], translation_dict)
		input_data = append_feature_data(input_data, new_data)	

	#feature combination by tensor product
	if feature_msg['features_combination_tensor']:
		new_data, _ = create_features_tensor_combine(data_current_one_month, \
			feature_msg['features_combination_tensor'], translation_dict):
		input_data = append_feature_data(input_data, new_data)

	#Add multiplicative interaction
	if feature_msg['features_interaction_mult']:
		new_data, _ = create_features_interaction(data_current_one_month, \
			feature_msg['features_interaction_mult'], translation_dict)
		input_data = append_feature_data(input_data, new_data)

	#Add time series sequence data 
	if feature_msg['features_time_sequence']:
		new_data, _ = create_features_sequence(data_current_one_month, \
			feature_msg['features_time_sequence'], translation_dict)
		input_data = append_feature_data(input_data, new_data)

	# add lagged product features if necessary
	if feature_msg['features_product_lags']:
		new_data, _ = create_features_products(data_current_one_month, \
			feature_msg['features_product_lags'], feature_msg['features_product_lags_cummax'])
		input_data = append_feature_data(input_data, new_data)		

	# add lagged profile features if necessary
	if feature_msg['features_profile_name_for_lags']:
		profile_columns_lag = [str(x) + '_L' + str(lag) for x in feature_msg['features_profile_name_for_lags'] \
			for lag in feature_msg['features_profile_lags']]
			
		new_data, _ = create_features_onehot_encode(data_current_one_month, profile_columns_lag, translation_dict)
		input_data = append_feature_data(input_data, new_data)

	# add status change for customer profile features
	if feature_msg['features_profile_change']:		
		new_data, _ = create_features_profile_change(data_current_one_month, \
			feature_msg['features_profile_change'], feature_msg['features_profile_change_lags'])
		input_data = append_feature_data(input_data, new_data)

	# add status change for customer product buyings
	if feature_msg['features_product_change_lags']:
		new_data, _ = create_features_product_change(data_current_one_month, FEATNAME.COLUMNS["product"], \
			feature_msg['features_product_change_lags']['lags'], feature_msg['features_product_change_lags']['use'])
		input_data = append_feature_data(input_data, new_data)
			
	# add nearest neighbor product features by simply appending
	if feature_msg['features_nn_used']:
		if is_train:
			nn_search_type = feature_msg['features_nn_search_type_train']
		else:
			nn_search_type = feature_msg['features_nn_search_type_test']
		new_data, _ = create_features_nn(data_query = data_current_one_month, data_train = data_train, month = month, \
			nn_used = feature_msg['features_nn_used'], nn_distance_lag = feature_msg['features_nn_distance_lags'], \
				nn_summary_lag = feature_msg['features_nn_summary_var_lags'], nn_search_type = nn_search_type, \
					istrain = feature_msg['train'])
		input_data = append_feature_data(input_data, new_data)

	#Now collect the output data
	label = data_current_one_month.buy_class.values if return_labels else None
	new_products = data_current_one_month.new_products.values if return_new_products else None
	user_ids = data_current_one_month.ncodpers.values if return_user_id else None
	previous_products = None
	if return_prev_prod:	  
		if is_train:
			previous_products = data_current_one_month[[col + '_L' + str(1) \
				for col in FEATNAME.COLUMNS["product"]]].values
		else:
			previous_products = data_previous_one_month[FEATNAME.COLUMNS["product"]].values
	
	return label, input_data, previous_products, user_ids, new_products

def create_data(feature_msg, translation_dict, \
					train_current, train_previous, test_current, test_previous, \
                    return_labels = True, return_prev_prod = True, return_user_id = True, return_new_products = True,\
                    MONTHS = None, is_train = True):
	"""
	create feature dataset for train or test
	
	input:
		feature_msg: specified feature set and selected months
		data_current, data_previous: data frames, could be train OR test
		return_labels: bool, whether to return label
		return_prev_prod: bool, whether to return previous product
		return_user_id: bool, whether to return user ids

	output:
		label, data, previous_products, user_ids, new_products
	"""	

	result = [None] * 5  #label, data, previous_products, user_ids, new_products
	for month in MONTHS:
		result_current_month = create_data_one_month(feature_msg, translation_dict, \
					train_current, train_previous, test_current, test_previous, \
                    return_labels = return_labels, return_prev_prod = return_prev_prod, \
                    return_user_id = return_user_id, return_new_products = return_new_products, \
                    month = month, is_train = is_train)
		for i in xrange(5):
			result[i] = np.concatenate((result[i], result_current_month[i]), axis=0) \
				if result[i] != None else result_current_month[i]

	return result


