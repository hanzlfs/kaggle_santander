## the model script allows direct feature engineer tests

from sklearn.linear_model import LogisticRegression
import numpy as np
from common_fn import *
from uTILS import *
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import pandas as pd 

import csv
import datetime
from operator import sub
from sklearn import preprocessing, ensemble
import pickle
import os



def logreg(X_train, y_train):
	clf = LogisticRegression(penalty='l1', C=0.1)
	clf.fit(X_train, y_train)
	#print "class order ", clf.classes_
	return clf
#clf.predict_proba(X)

def knn(X_train, y_train):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train, y_train)
	#print "class order ", clf.classes_
	return clf

def runXGB_binary(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'binary:logistic'
	param['eta'] = 0.04
	param['max_depth'] = 6
	param['silent'] = 1
	#param['num_class'] = 2
	param['eval_metric'] = "logloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.95
	param['colsample_bytree'] = 0.95
	param['seed'] = seed_val
	num_rounds = 90

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

def runXGB(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.04
	param['max_depth'] = 6
	param['silent'] = 1
	param['num_class'] = n_products
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.95
	param['colsample_bytree'] = 0.95
	param['seed'] = seed_val
	num_rounds = 90

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

def make_multi_lb(X_tr, y_tr):
	# stack 24 * 1 binary labels into multi class labels 0 - 23, simply as when_less_is_more scheme
	# input X: [nsamp * nfeat] y [24 * 1] 0-1
	# output X [new nsamp(includes duplication) * nfeat] y [1] 0 - 23
	X = []
	y = []
	for i in range(X_tr.shape[0]): # nsamp
		for j in range(y_tr.shape[1]): # 0 - 23
			if y_tr[i, j] == 1:
				X.append(list(X_tr[i,:]))
				y.append(j)
	X = np.array(X)
	y = np.array(y)
	return X, y
def gen_submission(preds, prev_prods):

	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)
	test_id = np.array(pd.read_csv("../input/test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	final = []
	for i in range(preds.shape[0]):
		final_pred = []
		pred_ = [TARGET[int(idx)] for idx in list(preds[i, :])]
		prev_ = [TARGET[int(idx)] for idx in range(24) if int(prev_prods[i, idx]) == 1] # !!!!! 
		#final_pred = [x for x in pred_ if x not in prev_]
		#prev_ = []
		count = 0
		for p in pred_:
			if p in prev_:
				continue
			count += 1
			if count > 7:
				break
			final_pred.append(p)
		final.append(' '.join(final_pred))
	
	out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final})
	out_df = out_df[["ncodpers","added_products"]]
	out_df.to_csv('../submission/sub_11302.csv', index=False)

def eval(y_true, preds, prev_prods):
# actual: nsample by nclass binary
# pred: nsample by nclass 0-1 probability
# prev_prods: [nsample by nclass] prev buying indicators

	print preds.shape
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)
	#preds = np.fliplr(preds)[:,:7]
	#final_preds = [list(TARGET[pred]) for pred in preds]
	score = 0.0 
	for i in range(y_true.shape[0]):		
		final_pred = []
		actual = [TARGET[idx] for idx in range(n_products) if y_true[i, idx] > 0]
		#print list(preds[i, :])
		pred_ = [TARGET[int(idx)] for idx in list(preds[i, :])]
		prev_ = [TARGET[int(idx)] for idx in list(prev_prods[i, :])]
		#final_pred = [x for x in pred_ if x not in prev_]
		#prev_ = []
		count = 0
		for p in pred_:
			if p in prev_:
				continue
			count += 1
			if count > 7:
				break
			final_pred.append(p)			
													
		score += apk(actual, final_pred)
		"""
		if not i%100:
			print "actual: ", actual
			#print "target_", target_
			print "val_Y: ", y_true[i, :]
			print "final_pred: ", final_pred
			print "score: ", apk(actual, final_pred)
		"""
			
	score /= y_true.shape[0]
	#print("      MAP@7 score on val set is " + str(score))
	return score

def collect_ffm_pred(tr_or_val = 'train'):
	# collect ffm prediction on tr and val
	# input tr[i].out from /home/ubuntu/Projects/kaggle-2014-criteo/santander/output for all i from 0 to 23! 
	# output [n_sample, n_prod] each have probabilities 
	X = []
	
	for i in range(24):		
		tr_path = "../output/tr" + str(i) + '.out'
		val_path = '../output/val' + str(i) + '.out'
		if tr_or_val == 'train':
			file_in_path = tr_path
		else:
			file_in_path = val_path
		x_cur = []
		with open(file_in_path,'r') as f:		
			for line in f:
				line = line.strip().rstrip('\n')
				prob = float(line)
				x_cur.append(prob)
		X.append(x_cur)

	X = np.array(X)
	X = np.transpose(X)
	return X

""" =========================featurize pipeline=============================== """
def get_feature_index():	
	# feature added in order same as processData_fn
	feature_list = ['cur_mon', 'joined_mons']
	feat_dict = {}
	target_dict = {}
	# added features 
	feat_dict['cur_mon'] = 0
	feat_dict['joined_mons'] = 1
	idx = 2
	# categorical features 
	for col in FEAT_CAT:
		if col == 'tipodom':
			continue
		feat_dict[col] = idx
		feature_list.append(col)
		idx += 1
	# numerical features
	for col in FEAT_NUM:
		feat_dict[col] = idx
		idx += 1
		feature_list.append(col)
	# numerical binwise features
	for col in ['age_bin', 'antiguedad_bin', 'renta_bin']:
		feat_dict[col] = idx
		idx += 1
		feature_list.append(col)
	# target features in X
	for col in FEAT_TARGET:
		feat_dict[col] = idx
		idx += 1
		feature_list.append(col)
	# target index for y
	idx = 0
	for col in target_cols:
		target_dict[col] = idx
		idx += 1
	return feat_dict, target_dict

def prepare_feature_summary(in_file_name):
	# prepare feature summary: 	
	#. median renta grouped by sex, age bin, and nomprov NA ignored
	renta_dict = {}
	renta_mean = {}
	count = 0
	for nomprov in range(len(ALL_PROVS)):
		renta_dict[nomprov] = {}
		renta_mean[nomprov] = {}
		for sex in [0,1,2]:
			renta_dict[nomprov][sex] = {}
			renta_mean[nomprov][sex] = {}
			for age_bin in range(5):
				renta_dict[nomprov][sex][age_bin] = []
				renta_mean[nomprov][sex][age_bin] = 101402.55

	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		renta = row['renta'].strip()
		if renta in ['', 'NA']:
			continue # skip missing values

		nomprov = getIndex(row, 'nomprov')					
		sex = getIndex(row, 'sexo') # first categorical features added then numerical stacked after
		age_bin =  getAge_bin(row['age'])  # any numerical value was rescaled into 0 to 1 
		
		renta = float(renta)
		renta_dict[nomprov][sex][age_bin].append(renta)

	for nomprov in renta_dict:
		for sex in renta_dict[nomprov]:
			for age_bin in renta_dict[nomprov][sex]:
				med = np.median(renta_dict[nomprov][sex][age_bin])
				renta_mean[nomprov][sex][age_bin] = med

	with open('../input/renta_mean.pickle', 'wb') as handle:
  		pickle.dump(renta_mean, handle)
  	return renta_mean

	
def onehot(X, feat_list_onehot, feat_list_value, feat_dict):
	# input must be np.array
		
	feat = {} 
	for ft_ in feat_list_onehot:
		if ft_ in FEAT_CAT:
			n_levels = len(mapping_dict[ft_]) #unique levels of categorical variable	
		elif ft_ == 'cur_mon':
			n_levels = 12
		elif ft_ in FEAT_TARGET:
			n_levels = 2 # product history 0 or 1 
		else:
			n_levels = 5 # numerical bin
		feat[ft_] = [0] * n_levels

	X_v = [] # return 
	for i in range(X.shape[0]):
		x_cur_ = []
		for ft_ in feat_list_onehot:
			feat_ = [0] * len(feat[ft_])
			idx = int(X[i, feat_dict[ft_]])
			feat_[idx] = 1
			x_cur_.extend(feat_)
		for ft_ in feat_list_value:
			x_cur_.append(X[i, feat_dict[ft_]])
		X_v.append(x_cur_)
	X_v = np.array(X_v)
	return X_v

def test_baseline(nlimit = 50, md = 'logreg'):
	# get feature indices
	feat_dict, target_dict = get_feature_index()

	# validation for models, feature engineers, etc
	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")

	#### prepare renta_mean used to impute missing renta ####
	if os.path.isfile("../input/renta_mean.pickle"):
		print "file exist"
		with open("../input/renta_mean.pickle", 'rb') as handle:
  			renta_mean = pickle.load(handle)
	else:
		renta_mean = prepare_feature_summary(in_file_name)
	in_file_name.close()
	print len(renta_mean)
	#### read and impute missing, train and val data from file	
	in_file_name =  open(data_path + "train_ver2.csv")
	#cust_dict_tr = {}
	#cust_dict_val = {}
	#x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id =\
	# 	processData_fn(in_file_name, cust_dict_tr, cust_dict_val, renta_mean, cur_mon = 5, \
	#							tr_out_file = None, val_out_file = None)
	
	if not os.path.isfile("../input/X_tr.dat"):
		x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id =\
			processData_fn_all_year(in_file_name, renta_mean, mon_val = 5, \
								tr_out_file = None, val_out_file = None)
		
		X_tr = np.array(x_vars_list_tr)
		y_tr = np.array(y_vars_list_tr)
		X_val = np.array(x_vars_list_val_)
		y_val = np.array(y_vars_list_val_)

		X_tr.dump("../input/X_tr.dat")
		y_tr.dump("../input/y_tr.dat")
		X_val.dump("../input/X_val.dat")
		y_val.dump("../input/y_val.dat")
	else:
		X_tr = np.load("../input/X_tr.dat")
		y_tr = np.load("../input/y_tr.dat")
		X_val = np.load("../input/X_val.dat")
		y_val = np.load("../input/y_val.dat")
	in_file_name.close()

	# save train and val matrix 		
	print X_tr.shape, X_val.shape
	print X_tr[0:1, :]
	len_feat = X_tr.shape[1]
	prev_prods_tr = X_tr[:, len_feat - 24: len_feat] # remove previous existed products from prediction 

	len_feat = X_val.shape[1]
	prev_prods_val = X_val[:, len_feat - 24: len_feat] # remove previous existed products from prediction 

	#### OPTIONAL: add/change into, one-hot features in X
	#feat_list_onehot = ['age_bin', 'antiguedad_bin', 'renta_bin'] + \
	#	[x for x in FEAT_CAT if x not in ['tipodom', 'pais_residencia', 'canal_entrada','nomprov']] # onehot the numerical bin feature 
	#feat_list_value = ['pais_residencia','canal_entrada','nomprov'] + FEAT_NUM + FEAT_TARGET

	feat_list_onehot = ['cur_mon']
	feat_list_value = [x for x in FEAT_CAT if x not in ['tipodom']]\
						+ FEAT_NUM + FEAT_TARGET

	X_tr = onehot(X_tr, feat_list_onehot, feat_list_value, feat_dict)
	X_val = onehot(X_val, feat_list_onehot, feat_list_value, feat_dict)

	print X_tr.shape, X_val.shape
	print X_tr[0:1, :]
	#### feed into model 
	#print X_val.shape, len(val_id)	
	xgtest = xgb.DMatrix(X_val)
		
	#print "prev_prods ", prev_prods.shape

	pred_val = [] # pred on val set 
	pred_tr = [] # pred on tr set

	if md == 'xgb':
		X, l_tr = make_multi_lb(X_tr, y_tr)
		#print X.shape
		clf = runXGB(X, l_tr, seed_val=0)
		pred_tr = clf.predict(xgb.DMatrix(X_tr))
		pred_val = clf.predict(xgtest)	

		score = eval(y_tr, pred_tr, prev_prods_tr)
		print "train map@7 = ", score
		score = eval(y_val, pred_val, prev_prods_val)
		print "val map@7 = ", score

		return
	"""
	file_in_path = "../input/tr_sample_feature_v2.csv"
	X, y = olddata(file_in_path, tr_or_val = 'train', nlimit = nlimit)
	clf = logreg(X, y)
	print "baseline train score  ", clf.score(X, y)

	file_in_path = "../input/val_sample_feature_v2.csv"
	X_val, y_val = olddata(file_in_path, tr_or_val = 'val', nlimit = nlimit)
	pred = clf.predict_proba(X_val)
	score = eval(y_val, pred)

	print "baseline map@7 = ", score
	"""
def sub_baseline(md = 'xgb'):
	data_path = "../input/"

	###### prepare data summary #######
	feat_dict, target_dict = get_feature_index()		
	in_file_name =  open(data_path + "train_ver2.csv")
	#### prepare renta_mean used to impute missing renta ####
	if os.path.isfile("../input/renta_mean.pickle"):
		print "file exist"
		with open("../input/renta_mean.pickle", 'rb') as handle:
  			renta_mean = pickle.load(handle)
	else:
		renta_mean = prepare_feature_summary(in_file_name)
	print len(renta_mean)
	in_file_name.close()
	#################
	###READ TEST DATA
	#################
	in_file_name =  open(data_path + "test_ver2.csv")	
	x_vars_list_val_, y_vars_list_val_, val_id =\
		processData_fn_test(in_file_name, renta_mean)	
	X_test = np.array(x_vars_list_val_)
	print "test data: ", X_test.shape
	in_file_name.close()	

	#################
	###READ TRAIN DATA
	#################
	in_file_name =  open(data_path + "train_ver2.csv")
	if not os.path.isfile("../input/X_tr.dat"):
		x_vars_list_tr, y_vars_list_tr, tr_id, x_vars_list_val_, y_vars_list_val_, val_id, cust_dict_ts_prev =\
			processData_fn_all_year(in_file_name, cust_dict_val, renta_mean, cur_mon = 5, \
									tr_out_file = None, val_out_file = None)
		X_tr = np.array(x_vars_list_tr)
		y_tr = np.array(y_vars_list_tr)
		X_val = np.array(x_vars_list_val_)
		y_val = np.array(y_vars_list_val_)

		X_tr.dump("../input/X_tr.dat")
		y_tr.dump("../input/y_tr.dat")
		X_val.dump("../input/X_val.dat")
		y_val.dump("../input/y_val.dat")
	else:
		X_tr = np.load("../input/X_tr.dat")
		y_tr = np.load("../input/y_tr.dat")
		X_val = np.load("../input/X_val.dat")
		y_val = np.load("../input/y_val.dat")
	in_file_name.close()
	###################
    ####process val data so useless rows are removed 
	###################
	X_val, y_val = clean_val_data(X_val, y_val)

	##### combine train and val 
	X_tr = np.concatenate((X_tr, X_val), axis=0)
	y_tr = np.concatenate((y_tr, y_val), axis=0)	
	print "train data: ", X_tr.shape, y_tr.shape	
	
	##### get last-mon product list for each sample 
	len_feat = X_tr.shape[1]
	prev_prods_tr = X_tr[:, len_feat - 24: len_feat] # remove previous existed products from prediction 
	len_feat = X_test.shape[1]
	prev_prods_test = X_test[:, len_feat - 24: len_feat] # remove previous existed products from prediction 

	#### OPTIONAL: add/change into, one-hot features in X
	#feat_list_onehot = ['age_bin', 'antiguedad_bin', 'renta_bin'] + \
	#	[x for x in FEAT_CAT if x not in ['tipodom', 'pais_residencia', 'canal_entrada','nomprov']] # onehot the numerical bin feature 
	#feat_list_value = ['pais_residencia','canal_entrada','nomprov'] + FEAT_NUM + FEAT_TARGET

	feat_list_onehot = ['cur_mon']
	feat_list_value = [x for x in FEAT_CAT if x not in ['tipodom']]\
						+ FEAT_NUM + FEAT_TARGET

	X_tr = onehot(X_tr, feat_list_onehot, feat_list_value, feat_dict)
	X_test = onehot(X_test, feat_list_onehot, feat_list_value, feat_dict)
	xgtest = xgb.DMatrix(X_test)
	print "after one hot train, test, ", X_tr.shape, X_test.shape

	if md == 'xgb':
		X, l_tr = make_multi_lb(X_tr, y_tr) # stack data to have scalar labels 
		clf = runXGB(X, l_tr, seed_val=0)
		pred_tr = clf.predict(xgb.DMatrix(X_tr))
		score = eval(y_tr, pred_tr, prev_prods_tr)
		print "train map@7 = ", score		
		pred_test = clf.predict(xgtest)	
				
	print "pred_test.shape: ", pred_test.shape
	gen_submission(pred_test, prev_prods_test)
	#score = eval(y_val, pred, prev_prods)
	#print "map@7 = ", score

""" ==================================================================== """

if __name__ == "__main__":
	sub_baseline(md = 'xgb')




