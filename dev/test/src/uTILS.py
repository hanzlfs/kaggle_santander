import csv
import datetime
from operator import sub
from sklearn import preprocessing, ensemble
import pickle
import os
from common_fn import *
import pandas as pd 
import numpy as np


def clean_val_data(X_val, y_val):
	X, y = [],[]
	for i in range(X_val.shape[0]):
		num_new_prod = sum(list(y_val[i, :]))
		if num_new_prod > 0:
			X.append(list(X_val[i, :]))
			y.append(list(y_val[i, :]))
	X = np.array(X)
	y = np.array(y)
	return X, y

def save_cust_id_dic(in_file_name):
	# for each month, give the PREVIOUS month's cust-id and product history
	cur_mon = '2015-01-28'
	count = 0
	cust_dict = {}
	for row in csv.DictReader(in_file_name):		
		if not count % 1000000:
			print "processed line: ", count
		count += 1
		
		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] > cur_mon:
			with open('../input/cust_dict_' + cur_mon + '.pickle', 'wb') as handle:
  				pickle.dump(cust_dict, handle)
			cust_dict.clear()
			cur_mon = row['fecha_dato']
		
		if row['fecha_dato'] == cur_mon:
			target_list = getTarget(row)
			cust_dict[cust_id] =  target_list[:]
	# save last month
	with open('../input/cust_dict_' + cur_mon + '.pickle', 'wb') as handle:
  				pickle.dump(cust_dict, handle)					
	
def processData_fn_all_year(in_file_name, renta_mean, mon_val = 5, \
								tr_out_file = None, val_out_file = None):
	# feat_dict: store the names of the feature 
	# collect train set up to prev_mon_val
	x_vars_list_tr = []
	y_vars_list_tr = []
	tr_id = []

	x_vars_list_val_ = [] 
	y_vars_list_val_ = [] # 1-by-n_products
	val_id = []

	cust_dict = {}
	count = 0
	date_val = '2016-' + str(mon_val).zfill(2) + '-28' # month for validation

	date_cur = '2015-01-28' # current month 
	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		## temp 
		#r = np.random.rand()
		#if r > 0.001:
		#	continue
		if (row['fecha_dato'] == '2015-01-28') or (row['fecha_dato'] > date_val):
  			continue

		cust_id = int(row['ncodpers'])
		# read cust_dict
		if row['fecha_dato'] > date_cur:
			cust_dict.clear()			
			with open('../input/cust_dict_'+ date_cur + '.pickle', 'rb') as handle:
  				cust_dict = pickle.load(handle)
  			print "cust_dict_" + date_cur + " loaded"
  			date_cur = row['fecha_dato']

		#### collect feature values 
		x_vars = []			
		# added features 
		# fecha_alta till current num of months
		cur_mon = int(row['fecha_dato'].split('-')[1])
		x_vars.append(cur_mon - 1)	
		first_join_date = row['fecha_alta'].strip()
		if first_join_date in ['', 'NA']:
			first_join_date = '2011-09-12'
		joined_mons = 12 * ( int(row['fecha_dato'].split('-')[0]) - int(first_join_date.split('-')[0]) ) \
						+ int(row['fecha_dato'].split('-')[1]) - int(first_join_date.split('-')[1])
		x_vars.append(joined_mons)	

		# categorical features that do not need further process
		for col in FEAT_CAT:
			if col == 'tipodom': # drop this feature 
				continue
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )		
		x_vars.append( getRent(row)) # keep missing values to -99 and impute later
		#x_vars.append( getRent(row) )

		# added: bin numerical features
		x_vars.append(getAge_bin(row['age']))
		x_vars.append(getCustSeniority_bin(row['antiguedad']))

		# fill missing renta and get renta bin
		nomprov = getIndex(row, 'nomprov')
		sex = getIndex(row, 'sexo')
		age_bin = getAge_bin(row['age'])		
		renta = getRent_keep_missing(row)
		if renta == -99:
			renta = renta_mean[nomprov][sex][age_bin]
		x_vars.append(getRent_bin(renta))
		
		prev_target_list = cust_dict.get(cust_id, [0]*n_products)
		if row['fecha_dato'] == date_val: # if date equal validation, then collect to val 			
			x_vars_list_val_.append(x_vars + prev_target_list)
			val_id.append(cust_id)

			if date_val != '2016-06-28':	# test set do not have target_list
				target_list = getTarget(row)
				new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
				y_vars_list_val_.append(new_products)								

		else:
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				assert len(prev_target_list) == n_products
				x_vars_list_tr.append(x_vars+prev_target_list)
				y_vars_list_tr.append(new_products)
				tr_id.append(cust_id)

	""" #the write into file part need to be modified accordingly 
	if tr_out_file:
		header = TARGET + ["cust_id"] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(tr_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_tr)):
				wr.writerow(y_vars_list_tr[i] + [tr_id[i]] + x_vars_list_tr[i])

	if val_out_file:
		header = TARGET + ['cust_id'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(val_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_val_)):
				wr.writerow(y_vars_list_val_[i] + [val_id[i]] + x_vars_list_val_[i])
	""" 
	# return cust_dict_val because it will be used in generating test set
	# cust_dict_val is the customer prod list in 2016-05-28 when mon = 6
	return x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id	

def processData_fn_test(in_file_name, renta_mean):
	# feat_dict: store the names of the feature 
	# collect train set up to prev_mon_val
	# cust_id_prev: the cust id prod list in 2016-05-28 
	x_vars_list_val_ = [] 
	y_vars_list_val_ = [] # 1-by-n_products
	val_id = []

	### load cust_dict to add prev-prod as feature
	with open('../input/cust_dict_2016-05-28.pickle', 'rb') as handle:
		cust_dict = pickle.load(handle)
		print "../input/cust_dict_2016-05-28.pickle loaded"

	count = 0
	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		x_vars = []			
		# added features 
		# fecha_alta till current num of months
		cur_mon = int(row['fecha_dato'].split('-')[1])
		x_vars.append(cur_mon - 1)	
		first_join_date = row['fecha_alta'].strip()
		if first_join_date in ['', 'NA']:
			first_join_date = '2011-09-12'
		joined_mons = 12 * ( int(row['fecha_dato'].split('-')[0]) - int(first_join_date.split('-')[0]) ) \
						+ int(row['fecha_dato'].split('-')[1]) - int(first_join_date.split('-')[1])
		x_vars.append(joined_mons)	

		# categorical features that do not need further process
		for col in FEAT_CAT:
			if col == 'tipodom': # drop this feature 
				continue
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )		
		x_vars.append( getRent(row)) # keep missing values to -99 and impute later
		#x_vars.append( getRent(row) )

		# added: bin numerical features
		x_vars.append(getAge_bin(row['age']))
		x_vars.append(getCustSeniority_bin(row['antiguedad']))

		# fill missing renta and get renta bin
		nomprov = getIndex(row, 'nomprov')
		sex = getIndex(row, 'sexo')
		age_bin = getAge_bin(row['age'])		
		renta = getRent_keep_missing(row)
		if renta == -99:
			renta = renta_mean[nomprov][sex][age_bin]
		x_vars.append(getRent_bin(renta))
		
		### aggregate data 	
		cust_id = int(row['ncodpers'])			
		prev_target_list = cust_dict.get(cust_id, [0]*n_products)
		x_vars_list_val_.append(x_vars + prev_target_list)
		val_id.append(cust_id)
	
	return x_vars_list_val_, y_vars_list_val_, val_id								


def processData_fn(in_file_name, cust_dict_tr, cust_dict_val, renta_mean, cur_mon = 5, \
								tr_out_file = None, val_out_file = None):
	# feat_dict: store the names of the feature 
	x_vars_list_tr = []
	y_vars_list_tr = []
	tr_id = []
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

		## temp 
		#r = np.random.rand()
		#if r > 0.01:
		#	continue

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
		# added features 
		# fecha_alta till current num of months			
		first_join_date = row['fecha_alta'].strip()
		if first_join_date in ['', 'NA']:
			first_join_date = '2011-09-12'
		joined_mons = 12 * ( int(row['fecha_dato'].split('-')[0]) - int(first_join_date.split('-')[0]) ) \
						+ int(row['fecha_dato'].split('-')[1]) - int(first_join_date.split('-')[1])
		x_vars.append(joined_mons)	

		# categorical features that do not need further process
		for col in FEAT_CAT:
			if col == 'tipodom': # drop this feature 
				continue
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )		
		x_vars.append( getRent(row)) # keep missing values to -99 and impute later
		#x_vars.append( getRent(row) )

		# added: bin numerical features
		x_vars.append(getAge_bin(row['age']))
		x_vars.append(getCustSeniority_bin(row['antiguedad']))

		# fill missing renta and get renta bin
		nomprov = getIndex(row, 'nomprov')
		sex = getIndex(row, 'sexo')
		age_bin = getAge_bin(row['age'])		
		renta = getRent_keep_missing(row)
		if renta == -99:
			renta = renta_mean[nomprov][sex][age_bin]
		x_vars.append(getRent_bin(renta))

		if row['fecha_dato'] == cur_mon_val:
			prev_target_list = cust_dict_val.get(cust_id, [0]*n_products)
			x_vars_list_val_.append(x_vars + prev_target_list)
			val_id.append(cust_id)

			if cur_mon_val != '2016-06-28':	# test set do not have target_list
				target_list = getTarget(row)
				new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
				y_vars_list_val_.append(new_products)								

			# Prev product has been used as a feature with all profile features for prediction! 
		elif row['fecha_dato'] == cur_mon_tr:
			prev_target_list = cust_dict_tr.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				assert len(prev_target_list) == n_products
				x_vars_list_tr.append(x_vars+prev_target_list)
				y_vars_list_tr.append(new_products)
				tr_id.append(cust_id)
				"""
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == n_products
						x_vars_list_tr.append(x_vars+prev_target_list)
						y_vars_list_tr.append(ind)
				"""

	""" the write into file part need to be modified accordingly 
	if tr_out_file:
		header = TARGET + ["cust_id"] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(tr_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_tr)):
				wr.writerow(y_vars_list_tr[i] + [tr_id[i]] + x_vars_list_tr[i])

	if val_out_file:
		header = TARGET + ['cust_id'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + FEAT_TARGET
		with open(val_out_file, 'w') as fp:
			wr = csv.writer(fp, delimiter=',')
			# write header
			wr.writerow(header)	
			for i in range(len(y_vars_list_val_)):
				wr.writerow(y_vars_list_val_[i] + [val_id[i]] + x_vars_list_val_[i])
	"""

	return x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id

def test_baseline_binary(nlimit = 50, md = 'logreg'):
	# use binary classification, baseline feature, baseline model
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")
	cust_dict_tr = {}
	cust_dict_val = {}
	x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id =\
	 	processData_fn(in_file_name, cust_dict_tr, cust_dict_val, cur_mon = 5, \
								tr_out_file = None, val_out_file = None)
	X_tr = np.array(x_vars_list_tr)
	y_tr = np.array(y_vars_list_tr)
	X_val = np.array(x_vars_list_val_)
	y_val = np.array(y_vars_list_val_)
	print X_tr.shape, y_tr.shape, X_val.shape, y_val.shape
	print X_tr[0:1, :]
	print X_val[0:1, :]

	xgtest = xgb.DMatrix(X_val)

	in_file_name.close()
	
	len_feat = X_tr.shape[1]
	prev_prods_tr = X_tr[:, len_feat - 24: len_feat] # remove previous existed products from prediction 

	len_feat = X_val.shape[1]
	prev_prods_val = X_val[:, len_feat - 24: len_feat] # remove previous existed products from prediction 
	#print "prev_prods ", prev_prods.shape

	pred_val = [] # pred on val set 
	pred_tr = [] # pred on tr set
	for i in range(24):
		#if i!=11:
		#	continue
		l_tr = y_tr[:,i] # label on train 
		l_val = y_val[:,i]
		if len(np.unique(l_tr)) == 1: # if all label for prod i is zero, then do not train, output all zero instead 
			pred_tr.append(list(l_tr))
			pred_val.append(np.unique(l_tr).tolist() * X_val.shape[0])
			continue		

		if md == 'logreg':
			clf = logreg(X_tr, l_tr)
			pred_tr_i = clf.predict_proba(X_tr)[:, 1]
			#print "product ", i, " logloss on train is ", log_loss(label, y_pred)
			pred_val_i = clf.predict_proba(X_val)[:, 1]
		elif md == 'knn':
			clf = knn(X_tr, label)
			pred_tr_i = clf.predict_proba(X_tr)[:, 1]
			#print "product ", i, " logloss on train is ", log_loss(label, y_pred)
			pred_val_i = clf.predict_proba(X_val)[:, 1]
		elif md == 'xgb':
			#print X_tr.shape, label.shape, np.unique(label)
			#clf = xgb.XGBClassifier()
			clf = runXGB_binary(X_tr, l_tr, seed_val=0)	
			#clf.fit(X, label)		
			#return 		
			pred_tr_i = clf.predict(xgb.DMatrix(X_tr))
			pred_val_i = clf.predict(xgtest)	
			print pred_val_i.shape	
			print "product ", i, " logloss on train is ", log_loss(l_tr, pred_tr_i)
			if len(np.unique(l_val)) > 1:
				print "product ", i, " logloss on val is ", log_loss(l_val, pred_val_i)
			
		#print pred_i.shape
		pred_tr.append(list(pred_tr_i))
		pred_val.append(list(pred_val_i))
		#print len(list(label_val)), len(list(pred_i))
	pred_tr = np.array(pred_tr)
	pred_tr = np.transpose(pred_tr)
	pred_val = np.array(pred_val)
	pred_val = np.transpose(pred_val)
	# gen submission

	score = eval(y_tr, pred_tr, prev_prods_tr)
	print "train map@7 = ", score
	score = eval(y_val, pred_val, prev_prods_val)
	print "val map@7 = ", score

def test_hash_binary(nlimit = 50):
	# use binary classification, hash feature, baseline model
	# use binary classification, baseline feature, baseline model
	
	pred = []
	actual = []
	for i in range(24):
		X, y = onehot_binary(file_in_path = None, i_prod = i, ndigit = 1e3, tr_or_val = 'train', nlimit = nlimit)
		X_val, y_val = onehot_binary(file_in_path = None, i_prod = i, ndigit = 1e3, tr_or_val = 'val', nlimit = nlimit)
		clf = logreg(X, y)
		print "baseline train score for prod ",i, "=", clf.score(X, y)
		pred_i = clf.predict_proba(X_val)
		pred.append(list(pred_i))
		actual.append(list(y_val))
	pred = np.array(pred)
	pred = np.transpose(pred)
	actual = np.array(actual)
	actual = np.transpose(actual)
	score = eval(actual, pred)
	print "map@7 = ", score

def test_ffm_binary(nlimit = 50):
	# use binary classification, hash feature, ffm model
	pred = collect_ffm_pred(tr_or_val = 'val')
	file_in_path = "../input/val_sample_feature_v3.csv"
	X_val, actual = olddata_binary(file_in_path = file_in_path, tr_or_val = 'val', nlimit = nlimit)
	
	len_feat = X_val.shape[1]
	prev_prods = X_val[:, len_feat - 24: len_feat] # remove previous existed products from prediction 
	print "prev_prods ", prev_prods.shape

	score = eval(actual, pred, prev_prods)
	print "map@7 = ", score



def test_hash(nlimit = 50):
	file_in_path = "../input/tr.hashtk"
	X, y = onehot(file_in_path, ndigit = 1e3, tr_or_val = 'train', nlimit = nlimit)
	clf = logreg(X, y)
	print "hashtrick train score  ", clf.score(X, y)

	file_in_path = "../input/val.hashtk"
	X_val, y_val = onehot(file_in_path, ndigit = 1e3, tr_or_val = 'val', nlimit = nlimit)
	pred = clf.predict_proba(X_val)
	score = eval(y_val, pred)

	print "hashtrick map@7 = ", score


def sub_baseline_binary(md = 'xgb'):
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")
	cust_dict_tr = {}
	cust_dict_val = {} # test actually 
	x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id =\
	 	processData_fn(in_file_name, cust_dict_tr, cust_dict_val, cur_mon = 6, \
								tr_out_file = None, val_out_file = None)

	X_tr = np.array(x_vars_list_tr)
	y_tr = np.array(y_vars_list_tr)

	in_file_name.close()
	in_file_name =  open(data_path + "test_ver2.csv")
	x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id =\
	 	processData_fn(in_file_name, cust_dict_tr, cust_dict_val, cur_mon = 6, \
								tr_out_file = None, val_out_file = None)
	in_file_name.close()
	
	X_test = np.array(x_vars_list_val_)
	#y_test = np.actual(y_test)
	print X_tr.shape, y_tr.shape, X_test.shape
	xgtest = xgb.DMatrix(X_test)

	len_feat = X_test.shape[1]
	prev_prods = X_test[:, len_feat - 24: len_feat] # remove previous existed products from prediction 

	pred = []
	for i in range(24):	
		label = y_tr[:,i]
		#label_val = y_val[:,i]
		if len(np.unique(label)) == 1:
			pred.append(np.unique(label).tolist() * X_test.shape[0])
			continue		

		if md == 'logreg':
			clf = logreg(X_tr, label)
			y_pred = clf.predict_proba(X_tr)[:, 1]
			#print "product ", i, " logloss on train is ", log_loss(label, y_pred)
			pred_i = clf.predict_proba(X_test)[:, 1]
			#print "product ", i, " logloss on val is ", log_loss(label_val, pred_i)
		elif md == 'knn':
			clf = knn(X_tr, label)
			y_pred = clf.predict_proba(X_tr)[:, 1]
			#print "product ", i, " logloss on train is ", log_loss(label, y_pred)
			pred_i = clf.predict_proba(X_test)[:, 1]
			#print "product ", i, " logloss on val is ", log_loss(label_val, pred_i)
		elif md == 'xgb':
			#print X_tr.shape, label.shape, np.unique(label)
			#clf = xgb.XGBClassifier()
			clf = runXGB_binary(X_tr, label, seed_val=0)	
			#clf.fit(X, label)		
			#return 		
			pred_i = clf.predict(xgtest)	
			print pred_i.shape	
			#print "product ", i, " logloss on val is ", log_loss(label_val, pred_i)
			
		#print pred_i.shape
		pred.append(list(pred_i))
		#print len(list(label_val)), len(list(pred_i))
	pred = np.array(pred)
	pred = np.transpose(pred)
	print pred.shape
	#print pred

	gen_submission(pred, prev_prods)
	#score = eval(y_val, pred, prev_prods)
	#print "map@7 = ", score


def olddata_binary(file_in_path, tr_or_val = 'train', nlimit = 50):
	# return X = [nsamp, nfeat]
	# return y = [nsamp, nprod] 0 or 1
	# tr_sample_feature_v3.csv and val_sample_feature_v3.csv
	X = []
	y = []
	count = 0
	
	count_pos = 0
	count_neg = 0
	count_total = 0
	i_prod = 11

	if tr_or_val == 'train':
		assert file_in_path == '../input/tr_sample_feature_v3.csv'
	else:
		assert file_in_path == '../input/val_sample_feature_v3.csv'
	f = open_with_first_line_skipped(file_in_path, skip=True)
	for line in f:
		if count % 10000 == 0:
			print "count line ", count
		if nlimit:
			if count > nlimit:
				break
		count += 1

		X_cur = []
		line = line.strip().rstrip('\n')			
		lst = line.split(',')
		#print lst
		actual = map(int, lst[0:24])

		
		if actual[i_prod] == 1:
			count_pos += 1
		else:
			count_neg += 1
		count_total += 1

		y.append(actual)
		lst_feat = lst[25:]		
		X_cur.extend(map(float,lst_feat))
		X.append(X_cur)

	#print "for prod: ", i_prod, " npos = ", count_pos, " nneg = ", count_neg, " ntotal = ", count_total

	X = np.array(X)
	y = np.array(y)
	#print X.shape, y.shape
	#print X[0:2, 0:100]
	#print np.sum(X[0:2,:])
	return X, y

def onehot_binary(file_in_path, i_prod = 0, ndigit = 1e6, tr_or_val = 'train', nlimit = 50):
	# return X = [nsamp, nfeat from hash -> onehot]
	# return y = [nsamp, nprod] 0 or 1
	# input from ../input/tr[i].ffm and ../input/val[i].ffm
	X = []
	y = []
	count = 0
	if tr_or_val == 'train':
		file_in_path == '../input/tr' + str(i_prod) + '.ffm'
	else:
		file_in_path == '../input/val' + str(i_prod) + '.ffm'
	with open(file_in_path,'r') as f:
		for line in f:
			if count % 10000 == 0:
				print "count line ", count
			if nlimit:
				if count > nlimit:
					break
			count += 1

			X_cur = []
			line = line.strip().rstrip('\n')			
			lst = line.split()
			#print lst
			label = int(lst[0])			
			y.append(label)
			lst_feat = lst[1:]

			for astr in lst_feat:
				feat = [0] * int(ndigit)
				hash_val = int(astr.split(':')[1]) % (int(ndigit) - 1)        		
				feat[hash_val] = 1
				X_cur.extend(feat)
			X.append(X_cur)
	X = np.array(X)
	y = np.array(y)
	#print X.shape, y.shape
	#print X[0:2, 0:100]
	#print np.sum(X[0:2,:])
	return X, y

if __name__ == "__main__":
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")
	save_cust_id_dic(in_file_name)
	in_file_name.close()






