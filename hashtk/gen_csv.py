# generate sample csv
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble

from common import *

def sampleData_for_val(in_file_name, out_file_name, cur_mon = 5, sample_per=0.01):
	# sample data into file for hash feature test
	np.random.seed(69069)
	
	header = ['ncodpers'] + ['fecha_dato'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + TARGET

	prev_mon = cur_mon - 1
	
	prev_mon_tr = '2015-' + str(prev_mon).zfill(2) + '-28'
	cur_mon_tr = '2015-' + str(cur_mon).zfill(2) + '-28'
	prev_mon_val = '2016-' + str(prev_mon).zfill(2) + '-28'	
	cur_mon_val	= '2016-' + str(cur_mon).zfill(2) + '-28'

	lst_mon_use = [prev_mon_tr, cur_mon_tr, prev_mon_val, cur_mon_val]

	with open(out_file_name, 'w') as fp:
		wr = csv.writer(fp, delimiter=',')
		# write header
		wr.writerow(header)		
		count_line = 0
		for row in csv.DictReader(in_file_name):
			if not count_line % 1000000:
				print "processed line: ", count_line
			count_line += 1
			sample = []
			if row['fecha_dato'] not in lst_mon_use:
				continue
			r = np.random.rand()
			if r > sample_per:
				continue
			cust_id = row['ncodpers']
			sample.append(cust_id)
			sample.append(row['fecha_dato'])

			for col in FEAT_CAT:
				sample.append( row[col] ) # first categorical features added then numerical stacked after
			sample.append( row['age'] ) # any numerical value was rescaled into 0 to 1 
			sample.append( row['antiguedad'] )
			sample.append( row['renta'] )
			target_list = getTarget(row)
			sample.extend( target_list)
			wr.writerow(sample)	

def processDataVal_by_month(in_file_name, cust_dict_tr, cust_dict_val, cur_mon = 5, \
								tr_out_file = None, val_out_file = None):
	x_vars_list_tr = []
	y_vars_list_tr = []
	x_vars_list_val_ = [] 
	y_vars_list_val_ = [] # 1-by-n_products
	val_id = []

	count = 0
	prev_mon = cur_mon - 1
	
	prev_mon_tr = '2015-' + str(prev_mon).zfill(2) + '-28'
	cur_mon_tr = '2015-' + str(cur_mon).zfill(2) + '-28'
	prev_mon_val = '2016-' + str(prev_mon).zfill(2) + '-28'	
	cur_mon_val	= '2016-' + str(cur_mon).zfill(2) + '-28'

	lst_mon_use = [prev_mon_tr, cur_mon_tr, prev_mon_val, cur_mon_val]

	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in lst_mon_use:
			continue

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in [prev_mon_tr]:	
			target_list = getTarget(row)
			cust_dict_tr[cust_id] =  target_list[:]
			continue

		if row['fecha_dato'] in [prev_mon_val]:	
			target_list = getTarget(row)
			cust_dict_val[cust_id] =  target_list[:]
			continue

		x_vars = []
		for col in FEAT_CAT:
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == cur_mon_val:
			prev_target_list = cust_dict_val.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]

			x_vars_list_val_.append(x_vars + prev_target_list)
			y_vars_list_val_.append(new_products)
			val_id.append(cust_id)

			# Prev product has been used as a feature with all profile features for prediction! 
		elif row['fecha_dato'] == cur_mon_tr:
			prev_target_list = cust_dict_tr.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == n_products
						x_vars_list_tr.append(x_vars+prev_target_list)
						y_vars_list_tr.append(ind)

	if tr_out_file:
		header = ["Label"] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(tr_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_tr)):
				wr.writerow([y_vars_list_tr[i]] + x_vars_list_tr[i])

	if val_out_file:
		header = TARGET + ['Label'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(val_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_val_)):
				wr.writerow(y_vars_list_val_[i] + [0] + x_vars_list_val_[i])

	return x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id	

if __name__ == "__main__":
	
	"""
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")
	out_file_name = "../input/train_ver2_sample_001.csv" 
	sampleData_for_val(in_file_name, out_file_name, sample_per=0.01)
	"""

	tr_out_file = "../input/tr_sample_feature_v2.csv"
	val_out_file = "../input/val_sample_feature_v2.csv"
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2_sample.csv")
	cust_dict_tr = {}
	cust_dict_val = {}
	cur_mon = 5
	x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id =\
			processDataVal_by_month(in_file_name, cust_dict_tr, cust_dict_val, cur_mon = cur_mon, \
								tr_out_file = tr_out_file, val_out_file = val_out_file)
	



