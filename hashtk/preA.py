from xgboost import XGBClassifier, Booster
import xgboost as xgb
import numpy as np
from numpy import loadtxt
import argparse, csv, sys
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from common import *

# combination of pre-a.py and xgbt.cpp
csv_path = "../input/tr_sample_feature_v2.csv"
gbdt_out_path = "../input/tr.gbdt.out"

csv_path_te = "../input/val_sample_feature_v2.csv"
gbdt_out_path_te = "../input/val.gbdt.out"

#These features are dense enough (they appear in the dataset more than 8000 times), so we include them in GBDT
target_cat_feats = read_dense_feats(threshold = 300) 

#process input and collect data
n_cat = len(target_cat_feats)

"""
cat_idx = {}
for i, key in enumerate(target_cat_feats):
	cat_idx[key] = i
"""
def runXGB_feature(train_X, train_y):
	#model = XGBClassifier(objective='multi:softprob', n_estimators=30, max_depth=8)
	model = XGBClassifier(n_estimators=30, max_depth=8)
	model.fit(train_X, train_y)
	return model

def gen_xgb_gbdt_out(file_in_path, file_out_path, tr_or_val = 'train', model = None):
	# we assumed the input file already processed 
	X = []
	y = []
	#header = []
	#generate indices
	count = 0
	for row in csv.DictReader(open(file_in_path)):
		if count % 1000 == 0:
			print count
		count += 1
		X_cur = []	
		# numerical features
		feats = []
		for j in range(len(FEAT_NUM)):
			field = FEAT_NUM[j]
			val = row[FEAT_NUM[j]]
			#feats.append('{0}'.format(val))
			feats.append(val)
		X_cur.extend(feats)
		if tr_or_val == 'train':
			y.append(int(row['Label'])) # label is from 0 to 23! 
		else:
			y.append(map(int,getTarget(row) + [0])) # for val label is just 1 * 24 + dummy label, not used in model training! 

		# categorical features: only include dense ones and need one-hot encoded
		cat_feats = set()
		for field in FEAT_CAT:
			key = field + '-' + row[field]
			cat_feats.add(key)

		feats = [0] * n_cat 
		for j, feat in enumerate(target_cat_feats):
			if feat in cat_feats:
				feats[j] = 1

		X_cur.extend(feats)
		X.append(X_cur)

	#feed into xgb
	X = np.array(X)
	y = np.array(y)
	print X.shape
	#print X[0:5, :]
	print y.shape
	print np.unique(y)
	if not model:
		model = runXGB_feature(X, y)
	# get leaf indices
	xg = xgb.DMatrix(X)
	pred_arr = model.booster().predict(xg, output_margin=False, ntree_limit=0, pred_leaf=True)
	print model.booster().get_dump()[0]
	# generate tr_xgb.gbdt.out
	print pred_arr.shape
	#print pred_arr[0:5, :]
        if tr_or_val == 'train':
		tr_xgb_gbdt = np.concatenate([np.reshape(y, (len(y),1)), pred_arr], axis = 1)
	else:
		tr_xgb_gbdt = np.concatenate([y, pred_arr], axis = 1)
	#print tr_xgb_gbdt.shape
	#print tr_xgb_gbdt[0:5, :]
	#print np.max(tr_xgb_gbdt)
	with open(file_out_path, 'w') as f_gbdt:
		for i in range(tr_xgb_gbdt.shape[0]):
			f_gbdt.write(' '.join(map(str, list(tr_xgb_gbdt[i, :]))) + '\n')
	return model

if __name__ == "__main__":
	# combination of pre-a.py and gbdt.cpp
	file_in_path = csv_path
	file_out_path = gbdt_out_path
	model = gen_xgb_gbdt_out(file_in_path = file_in_path, file_out_path = file_out_path, \
								tr_or_val = 'train', model = None)
	
	file_in_path = csv_path_te
	file_out_path = gbdt_out_path_te
	model = gen_xgb_gbdt_out(file_in_path = file_in_path, file_out_path = file_out_path, \
								tr_or_val = 'val', model = model)


	



