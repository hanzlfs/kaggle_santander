#############################################################################
####Join nbayes train_current data with original data and create lag values##
#############################################################################
import pandas as pd
import numpy as np
from common import *
from ast import literal_eval
import datetime
import random
import time
from sklearn import preprocessing
from quickcheck import * # quick check if the computed new product matches train_current and eval_current
######################
##### Helpers ########
######################

def diff_month(d1, d2):
	return (d1.year - d2.year)*12 + d1.month - d2.month

def conv_to_mon(date_str):
	#d0 = datetime.datetime.strptime('2015-01-28', "%Y-%m-%d")
	#d1 = datetime.datetime.strptime(date_str, "%Y-%m-%d")
	#return diff_month(d1, d0)
	yr = int(date_str.split('-')[0])
	mon = int(date_str.split('-')[1])
	if yr == 2015:
		return mon - 1
	else:
		return mon + 11


def read_original(months = None):
	"""
	read original data from kaggle

	return: pandas data frame
	"""
	filename = "../input/train_ver2.csv"
	df = pd.read_csv(filename, dtype={"sexo":str,\
									"ind_nuevo":str,\
									"ult_fec_cli_1t":str,\
									"indext":str}, header = 0)

	# Create total month variable 
	df["total_month"] = df["fecha_dato"].apply(conv_to_mon)

	# return selected months
	if months is not None:
		df = df.loc[df.total_month.isin(months), :]

	# Minimum process of the data, do not fill missing values, just correct strange value ranges
	df["age"]   = pd.to_numeric(df["age"], errors="coerce") 
	df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce") 
	df["indrel_1mes"]   = pd.to_numeric(df["indrel_1mes"], errors="coerce") 
	df.loc[df.antiguedad<0, "antiguedad"] = 0
	df.loc[df.age < 0,"age"]  = 0
	df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"

	
	return df	

def read_our_data(filename = None, months = None):
	"""
	read train current data which contains profile feature and target value
	"""
	limit_rows   = 20000000
	df = pd.read_csv(filename, dtype = dictionary_types, header = 0, nrows = limit_rows)

	# return selected months
	if months is not None:		
		df = df.loc[df.fecha_dato.isin(months), :]
	return df

def join_and_clean(tr_current, tr_original, lags = None):
	"""
	For each month, find in original data which (month == current_month - lag AND user_id == current_user_id)

	If not found, then fill NA with:
		if NA is in product features, just fill 0
		if NA is in profile features, for each lag, fill the previous month with the current month, starting from lag 0
	
	return: the joined df[fecha_dato, ncodpers, target_values, profile_feature(for current and lags), \
	product feature(for current and lags)]

	months: the months used to filter train_current first
	lags: the lags used to join
	"""

	# Trim df_train_original to have only other features in df_train_current and product features
	cols_keep = ['fecha_dato','ncodpers','ind_empleado','pais_residencia','sexo','age',\
								'fecha_alta','ind_nuevo', 'antiguedad','indrel','indrel_1mes','tiprel_1mes',\
								'indresi','indext','canal_entrada','indfall','nomprov','ind_actividad_cliente','renta','segmento'] \
								+ feat_prod
	tr_original = tr_original[cols_keep + ['total_month']]

	# Create keys to join in df_train_current
	if lags	is None:
		lags = [0] 
	lags = sorted(lags) 

	join_ley_right = 'join_key_right'
	tr_original.loc[:,join_ley_right] = tr_original[['ncodpers','total_month']].apply(lambda row: str(row['ncodpers']) + '-'\
								+ str(row['total_month']), axis = 1)
	for lag in lags:
		#join_key = 'join_key_L' + str(lag)
		join_key_left = 'join_key_left_L' + str(lag)
		tr_current.loc[:,join_key_left] = tr_current[['ncodpers','fecha_dato']].apply(lambda row: str(row['ncodpers']) + '-'\
												+ str(row['fecha_dato'] - lag), axis = 1)
	

	# Left join on df_train_current with df_train_original by join_keys
	for lag in lags:	   
		join_key_left = 'join_key_left_L' + str(lag)	   
		tr_current = tr_current.merge(tr_original, how = 'left', left_on = join_key_left, \
										right_on = join_ley_right, suffixes = ['','_L' + str(lag)])

	# Impute missing values
	# Fill Product missing values with 0
	feat_product = [col + '_L' + str(lag) if lag > 0 else col for col in feat_prod for lag in lags]
	for col in feat_product:
		tr_current.loc[:,col].fillna(0, inplace = True)
	
	print "Columns with missing values in df_train_current: "
	print tr_current[feat_product].isnull().any()

	# Propogate from lag small to large profile missing values
	feat_profile = ['ind_empleado','pais_residencia','sexo','age',\
					'fecha_alta','ind_nuevo', 'antiguedad','indrel',\
					'indrel_1mes','tiprel_1mes','indresi','indext','canal_entrada',\
					'indfall','nomprov','ind_actividad_cliente','renta','segmento']
	
	for col in feat_profile:
		for lag in lags:
			if lag == 0:
				tr_current.loc[:, col + '_L' + str(lag)].fillna(tr_current.loc[:, col], inplace = True)    
			else:
				tr_current.loc[:, col + '_L' + str(lag)].fillna(tr_current.loc[:, col + '_L' + str(lag - 1)], inplace = True)   

	# Check and raise alert if there is any missing value remains in df
	print "Columns with missing values in df_train_current: "
	print tr_current.columns[tr_current.isnull().any()].values.tolist()

	return tr_current

def data_sanity_check(df_train_current, df_train_previous, months = None, lags = None):
	"""
	Check with nbayes data to see if our joins are correct
	1. Compare the added features based on target_values and the difference of current product feature and lag 1 product feature
	2. Compare the lag 1 product feature with train_previous product feature with mon-1
	3. Compare the lag 1 product change feature (comparing lag 1 and lag 2) with train_previous product feature with mon-1
	4. If 3 does not match, figure out how they computed the product change feature, basically select in train_previous user ids and \
	join on itself with mon-1, compare the difference of product features, to see if this give their change features. If so, then we should use our \
	own product feature and make our own changes as their approach is incorrect. 

	return None
	"""	
	# trim months
	if months is not None:
		df_train_current = df_train_current.loc[df_train_current.fecha_dato.isin(months), :]
		df_train_previous = df_train_previous.loc[df_train_previous.fecha_dato.isin([mon - 1 for mon in months]), :]

	# In train_current, check if the "new_products" feature matches the comparison of L0 prod and L1 prod
	print "...Compare Target Equality in current"
	_ = quick_check(df_train_current)

	# In train_current and train_previous, compare wither the previous month products match 
	print "...Compare Product Equality in current vs. previous"	
	for mon in months:
		# Product feature data match check
		count_not_eq = np.sum(np.not_equal(df_train_current.loc[df_train_current.fecha_dato == mon, products_L1].values.astype(int), \
			df_train_previous.loc[df_train_previous.fecha_dato == mon - 1, products_L0].values.astype(int)))
		print "Num Mismatch for month: ", mon, " is, ", count_not_eq


	# Compare wither the L1 vs. L2 difference in current matches the product "change" in previous data
	if max(lags) < 2:
		# If there is no lag-2, then skip the next quality check
		return

	print "...Compare Product Change Equality in current vs. previous"
	for mon in months:
		# Status change data match check		
		change_in_current = np.not_equal(df_train_current.loc[df_train_current.fecha_dato == mon, products_L1].values.astype(int), \
								df_train_current.loc[df_train_current.fecha_dato == mon, products_L2].values.astype(int)).astype(int)

		change_in_previous = df_train_previous.loc[df_train_previous.fecha_dato == mon - 1, products_change].values.astype(int)
		print "Num Mismatch for month: ", mon, "is, ", np.sum(np.not_equal(change_in_current, change_in_previous))
	
def data_sanity_check_prod_change(months = None):
	# Check the data sanity specially for product change
	if len(months) < 2:
		print "We must have at least two months data"
		return

	path_train_previous = "../input/train_previous_month_dataset.csv"
	tr_previous = read_our_data(filename = path_train_previous, months = months)

	# Create key for join on lag 1	
	tr_previous.loc[:,'join_key_left'] = tr_previous[['ncodpers','fecha_dato']].apply(lambda row: str(row['ncodpers']) + '-'\
											+ str(row['fecha_dato'] - 1), axis = 1)
	tr_previous.loc[:,'join_key_right'] = tr_previous[['ncodpers','fecha_dato']].apply(lambda row: str(row['ncodpers']) + '-'\
											+ str(row['fecha_dato']), axis = 1)

   	# Left join tr_previous on itself with lag 1
	tr_previous = tr_previous.merge(tr_previous, how = 'left', left_on = ['join_key_left'],\
					right_on = ['join_key_right'], suffixes = ['','_L' + str(1)])

	# Fill in missing values
	feat_product = products_L1
	for col in feat_product:
		tr_previous.loc[:,col].fillna(0, inplace = True)

	# Check if there is still missing values
	print "Columns with missing values in df_train_current: "
	print tr_previous.columns[tr_previous.isnull().any()].values.tolist()

	# compute product change and compare computed with presented
	change_computed = np.not_equal(tr_previous.loc[:, products_L0].values.astype(int), \
								tr_previous.loc[:, products_L1].values.astype(int)).astype(int)

	change_given = tr_previous.loc[:, products_change].values.astype(int)
	print "Num Mismatch is, ", np.sum(np.not_equal(change_computed, change_given))

def prod_new(row):
	"""
	return new product indices 

	"""
	#feat_current_products = [col + '_L' + str(0) for col in feat_prod]
	feat_current_products = feat_prod
	feat_prev_products = [col + '_L' + str(1) for col in feat_prod]
	added_products = np.equal(row[feat_prev_products], 0) * row[feat_current_products]

	return np.nonzero(added_products)[0]

if __name__ == "__main__":

	path_train_current = "../input/train_current_month_dataset.csv"
	path_train_previous = "../input/train_previous_month_dataset.csv"
	path_eval_current = "../input/eval_current_month_dataset.csv"	
	path_eval_previous = "../input/eval_previous_month_dataset.csv"

	
	# Read original data
	start_time = time.time()
	df_train_original = read_original(months = [14, 15, 16])	
	print('It took %i seconds to load original' % (time.time()-start_time))

	# Read our data
	start_time = time.time()
	df_train_current = read_our_data(filename = path_train_current, months = [16])
	df_train_previous = read_our_data(filename = path_train_previous, months = [15])
	#df_eval_current = read_our_data(filename = path_eval_current, months = [16])
	#df_eval_previous = read_our_data(filename = path_eval_previous, months = [15])
	print('It took %i seconds to load our data' % (time.time()-start_time))

	# Join original data into our data	
	start_time = time.time()
	df_train_current = join_and_clean(df_train_current, df_train_original, lags = [0,1,2])
	#df_eval_current = join_and_clean(df_eval_current, df_train_original, lags = [0,1,2])	
	print('It took %i seconds to join' % (time.time()-start_time))

	# Data quality check 
	start_time = time.time()
	data_sanity_check(df_train_current, df_train_previous, months = [16], lags = [0,1,2])
	#data_sanity_check(df_eval_current, df_eval_previous, months = [16])
	print('It took %i seconds for quality check' % (time.time()-start_time))
	

	"""
	start_time = time.time()
	data_sanity_check_prod_change(months = [14, 15])
	print('It took %i seconds for quality check for product change' % (time.time()-start_time))
	"""

