# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from ast import literal_eval
from featureNames import FEATNAME

def get_dtype():
	"""
	get variable types to read from csv

	"""
	dictionary_types = {}
	# Numerical profile variable
	for col in FEATNAME.COLNAMES["profile_num"] + :
		dictionary_types.update({col: float})

	# Categorical profile variable and product change variable
	for col in FEATNAME.COLNAMES["profile_cate"] + FEATNAME.COLNAMES["profile_cate_lag"] + \
		FEATNAME.COLNAMES["product_change"] + FEATNAME.COLNAMES["product_change_lag"]:
		dictionary_types.update({col: "category"})

	# Product variable
	for col in FEATNAME.COLNAMES["product"] + FEATNAME.COLNAMES["product_lag"]:
		dictionary_types.update({col: np.int8})

	return dictionary_types

def load_data(path_current, path_previous):
	"""
	Loads the raw datasets in csv files
	return pandas.DataFrame data_current, data_prev 
	"""
	dtype = get_dtype()
	data_current = pd.read_csv(path_current, dtype=dtype, header = 0)	
	data_previous = pd.read_csv(path_previous, dtype=dtype, header = 0)
	return data_current, data_previous

def generate_map_dict(data_train, data_test):
	"""
	get map dictionary from both train and test data and map categorical feature into index 
	data_train: dataframe raw train_current
	data_test: dataframe raw test_current
	"""

	for df in [data_train, data_test]:
		
		#Discretize the data			
		renta_ranges = [0]+list(range(20000, 200001, 10000))
		renta_ranges += list(range(300000, 1000001, 100000))+[2000000, 100000000]
		df.renta = pd.cut(df.renta, renta_ranges, right=True)

		antiguedad_ranges = [-10]+list(range(365, 7301, 365))+[8000]
		df.antiguedad = pd.cut(df.antiguedad, antiguedad_ranges, right=True)
		
		age_ranges = list(range(0, 101, 10))+[200] 		
		df.age = pd.cut(df.age, age_ranges, right=True)

		#Create month column
		df['month'] = (df.fecha_dato)%12 + 1
		df.month = df.month.astype('category')

	for df in [data_train, data_test]:
		for key in ['month'] + FEATNAME.COLNAMES["profile_cate"] + FEATNAME.COLNAMES["profile_cate_lag"] +\
				 FEATNAME.COLNAMES["product_change"] + FEATNAME.COLNAMES["product_change_lag"]:
			df[key] = df[key].astype('category')

	translation_dict = {}
	translation_dict['month'] = {}
	for mon in range(1, 13):
		translation_dict['month'].update({mon: mon-1})		

	for key in FEATNAME.COLNAMES["product_change"] + FEATNAME.COLNAMES["product_change_lag"]:
		if key not in translation_dict:
			translation_dict[key] = {'-1':1, '0':0, '1':2}		

	for key in FEATNAME.COLNAMES["profile_cate"] + FEATNAME.COLNAMES["profile_cate_lag"] +\
			FEATNAME.COLNAMES["profile_num"]:

		if key not in translation_dict:
			translation_dict[key] = {}
			count_level = 0

		for category in data_train[key].cat.categories:
			translation_dict[key].update({category:count_level})
			count_level += 1
		
		for category in data_test[key].cat.categories:
			if category not in translation_dict[key]:
				translation_dict[key].update({category:count_level})
				count_level += 1

	return translation_dict

def preprocess(data, translation_dict):
	"""
	map categorical values into unique integer index specified in columns
	change target variables into accessible format
	"""	
	
	for key in set(translation_dict.keys()) & set(data.columns.values.tolist()):
		data[key].cat.categories = [translation_dict[key][category] for category in data[key].cat.categories]		
	data.new_products = data.new_products.apply(literal_eval)

	return data

def get_processed_data(path_current_train, path_previous_train, path_current_test, path_previous_test,
						return_mapping_dict = True):
	"""
	input:
		return_mapping_dict: bool, whether to return mapping dictionary for categorical features

	return processed train_current, train_previous, test_current, test_previous
	"""
	#load raw data
	train_current, train_previous = load_data(path_current_train, path_previous_train)
	test_current, test_previous = load_data(path_current_train, path_previous_train)

	#generate mapping dictionary
	translation_dict = generate_map_dict(train_current, test_current)

	#map categorical to its dictionary values
	for data in [train_current, train_previous, test_current, test_previous]:
		data = preprocess(data, translation_dict)

	mapping_dict = translation_dict if return_mapping_dict else None

	return train_current, train_previous, test_current, test_previous, mapping_dict



