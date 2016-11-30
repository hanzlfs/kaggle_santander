"""
Code based on BreakfastPirate Forum post
__author__ : SRK
"""
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
#import ml_metrics as metrics

mapping_dict = {
'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'ind_nuevo'     : {'0':0, '1':1, -99:2},
'indrel'        : {'1':0, '99':1, -99:2},
'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi'       : {-99:0, 'S':1, 'N':2},
'indext'        : {-99:0, 'S':1, 'N':2},
'conyuemp'      : {-99:0, 'S':1, 'N':2},
'indfall'       : {-99:0, 'S':1, 'N':2},
'tipodom'       : {-99:0, '1':1},
'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}
cat_cols = list(mapping_dict.keys())

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[0:]
n_products = len(target_cols)
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def getTarget(row):
	tlist = []
	for col in target_cols:
		if row[col].strip() in ['', 'NA']:
			target = 0
		else:
			target = int(float(row[col]))
		tlist.append(target)
	return tlist

def getIndex(row, col):
	val = row[col].strip()
	if val not in ['','NA']:
		ind = mapping_dict[col][val]
	else:
		ind = mapping_dict[col][-99]
	return ind

def getAge(row):
	mean_age = 40.
	min_age = 20.
	max_age = 90.
	range_age = max_age - min_age
	age = row['age'].strip()
	if age == 'NA' or age == '':
		age = mean_age
	else:
		age = float(age)
		if age < min_age:
			age = min_age
		elif age > max_age:
			age = max_age
	return round( (age - min_age) / range_age, 4) # normalize age into 0 to 1

def getCustSeniority(row):
	min_value = 0.
	max_value = 256.
	range_value = max_value - min_value
	missing_value = 0.
	cust_seniority = row['antiguedad'].strip()
	if cust_seniority == 'NA' or cust_seniority == '':
		cust_seniority = missing_value
	else:
		cust_seniority = float(cust_seniority)
		if cust_seniority < min_value:
			cust_seniority = min_value
		elif cust_seniority > max_value:
			cust_seniority = max_value
	return round((cust_seniority-min_value) / range_value, 4)

def getRent(row):
	min_value = 0.
	max_value = 1500000.
	range_value = max_value - min_value
	missing_value = 101850.
	rent = row['renta'].strip()
	if rent == 'NA' or rent == '':
		rent = missing_value
	else:
		rent = float(rent)
		if rent < min_value:
			rent = min_value
		elif rent > max_value:
			rent = max_value
	return round((rent-min_value) / range_value, 6)
	
def processData(in_file_name, cust_dict):
	x_vars_list = []
	y_vars_list = []
	count = 0
	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:
			continue

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:	
			target_list = getTarget(row)
			cust_dict[cust_id] =  target_list[:]
			continue

		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2016-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*n_products)
			x_vars_list.append(x_vars + prev_target_list) 
			# Prev product has been used as a feature with all profile features for prediction! 
		elif row['fecha_dato'] == '2015-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == n_products
						x_vars_list.append(x_vars+prev_target_list)
						y_vars_list.append(ind)

	return x_vars_list, y_vars_list, cust_dict

def processDataVal_by_month(in_file_name, cust_dict_tr, cust_dict_val):
	x_vars_list_tr = []
	y_vars_list_tr = []
	x_vars_list_val_ = [] 
	y_vars_list_val_ = [] # 1-by-n_products
	val_id = []

	count = 0
	for row in csv.DictReader(in_file_name):
		if not count % 1000000:
			print "processed line: ", count
		count += 1

		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-04-28', '2015-05-28', '2016-04-28', '2016-05-28']:
			continue

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2015-04-28']:	
			target_list = getTarget(row)
			cust_dict_tr[cust_id] =  target_list[:]
			continue

		if row['fecha_dato'] in ['2016-04-28']:	
			target_list = getTarget(row)
			cust_dict_val[cust_id] =  target_list[:]
			continue


		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2016-05-28':
			prev_target_list = cust_dict_val.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]

			x_vars_list_val_.append(x_vars + prev_target_list)
			y_vars_list_val_.append(new_products)
			val_id.append(cust_id)

			# Prev product has been used as a feature with all profile features for prediction! 
		elif row['fecha_dato'] == '2015-05-28':
			prev_target_list = cust_dict_tr.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == n_products
						x_vars_list_tr.append(x_vars+prev_target_list)
						y_vars_list_tr.append(ind)

	return x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id

def processDataVal(in_file_name, cust_dict_tr, cust_dict_val, val_per):
	# validation through splitting the prev-cur month train set 
	# split the 2015-05-28, 2015-06-28 data for validation 
	# val_per: the ratio of val set out of all set

	x_vars_list_tr = [] 
	y_vars_list_tr = []	# 1-by-1
	x_vars_list_val_ = [] 
	y_vars_list_val_ = [] # 1-by-n_products
	val_id = []
	np.random.seed(69069)
	count_line = 0
	count_prev_mon = 0
	count_cur_mon = 0
	for row in csv.DictReader(in_file_name):
		if not count_line % 1000000:
			print "processed line: ", count_line
		count_line += 1

		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-05-28', '2015-06-28']:
			continue

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2015-05-28']:	
			target_list = getTarget(row)
			r = np.random.rand()
			if r < val_per:
				cust_dict_val[cust_id] =  target_list[:]
			else:
				cust_dict_tr[cust_id] =  target_list[:]					
			count_prev_mon += 1
			continue

		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) ) # first categorical features added then numerical stacked after
		x_vars.append( getAge(row) ) # any numerical value was rescaled into 0 to 1 
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2015-06-28':
			r = np.random.rand()
			if r < val_per:
				prev_target_list = cust_dict_val.get(cust_id, [0]*n_products)
			else:
				prev_target_list = cust_dict_tr.get(cust_id, [0]*n_products)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if r < val_per:
				x_vars_list_val_.append(x_vars+prev_target_list)
				y_vars_list_val_.append(new_products)
				val_id.append(cust_id)
			else:
				if sum(new_products) > 0:
					for ind, prod in enumerate(new_products):
						if prod>0:
							assert len(prev_target_list) == n_products
							x_vars_list_tr.append(x_vars+prev_target_list)
							y_vars_list_tr.append(ind)

	return x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id
			
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

def post_process_pred(preds, test_id, train_file):
	# preds: already ordered prod id 
	# collect cust_id from 2016-05-28
	cust_dict_last_mon = {}
	for row in csv.DictReader(train_file):
		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2016-05-28']:	
			target_list = getTarget(row)
			cust_dict_last_mon[cust_id] =  target_list[:]
	
	# select top 7 preds for each test_id excluding already exist product
	pred_after_proc = []
	for i in range(test_id.shape[0]):
		temp_ = []
		cust_id_test = test_id[i]
		prev_target_list = cust_dict_last_mon.get(cust_id_test, [0]*n_products)
		for j in range(preds.shape[1]):
			if prev_target_list[preds[i,j]] == 1:
				continue
			temp_.append(preds[i,j])
			if len(temp_) == 7:
				break
		pred_after_proc.append(temp_)
	pred_after_proc = np.array(pred_after_proc)
	return pred_after_proc

def post_process_val(preds, cust_dict_val, val_id):
	# preds: already ordered prod id 
	
	# select top 7 preds for each val_id excluding already exist product
	pred_after_proc = []
	for i in range(len(val_id)):
		temp_ = []
		cust_id_test = val_id[i]
		prev_target_list = cust_dict_val.get(cust_id_test, [0]*n_products)
		for j in range(preds.shape[1]):
			if prev_target_list[preds[i,j]] == 1:
				continue
			temp_.append(preds[i,j])
			if len(temp_) == 7:
				break
		pred_after_proc.append(temp_)
	pred_after_proc = np.array(pred_after_proc)
	return pred_after_proc

def validation_by_month():
	start_time = datetime.datetime.now()
	print "test apk: ", apk([],["am","b","c"])
	data_path = "../input/"
	train_file =  open(data_path + "train_ver2.csv")

	print("Validation...")
	cust_dict_tr = {} # prev-mon val id
	cust_dict_val = {} # prev-mon val id
	x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id = \
		processDataVal_by_month(train_file, cust_dict_tr, cust_dict_val)
	train_X = np.array(x_vars_list_tr)
	train_y = np.array(y_vars_list_tr)
	val_X = np.array(x_vars_list_val_)
	val_Y = np.array(y_vars_list_val_)
	print("		Building model...")
	model = runXGB(train_X, train_y, seed_val=0)
	del train_X, train_y
	print("     Predicting on Val...")
	xgval = xgb.DMatrix(val_X)
	preds = model.predict(xgval)
	del val_X, xgval
	target_ = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)
	preds_after_proc = post_process_val(preds, cust_dict_val, val_id) # remove already exist products
	final_preds = [list(target_[pred]) for pred in preds_after_proc]
	score = 0.0 
	for i in range(val_Y.shape[0]):		
		actual = [target_[idx] for idx in range(n_products) if val_Y[i, idx] > 0]
		score += apk(actual, final_preds[i])
		if not i%1000:
			print "actual: ", actual
			print "target_", target_
			print "val_Y: ", val_Y[i, :]
			print "pred: ", final_preds[i]
			print "score: ", apk(actual, final_preds[i])
	score /= val_Y.shape[0]
	del val_Y
	print("      MAP@7 score on val set is " + str(score))
	print(datetime.datetime.now()-start_time)

def validation():
	start_time = datetime.datetime.now()
	print "test apk: ", apk([],["am","b","c"])
	data_path = "../input/"
	train_file =  open(data_path + "train_ver2.csv")

	print("Validation...")
	cust_dict_val = {} # prev-mon val id
	cust_dict_tr = {} # prev-mon tr id
	x_vars_list_tr, y_vars_list_tr, x_vars_list_val_, y_vars_list_val_, val_id = \
		processDataVal(train_file, cust_dict_tr, cust_dict_val, 0.25)
	train_X = np.array(x_vars_list_tr)
	train_y = np.array(y_vars_list_tr)
	val_X = np.array(x_vars_list_val_)
	val_Y = np.array(y_vars_list_val_)
	print("		Building model...")
	model = runXGB(train_X, train_y, seed_val=0)
	del train_X, train_y
	print("     Predicting on Val...")
	xgval = xgb.DMatrix(val_X)
	preds = model.predict(xgval)
	del val_X, xgval
	target_ = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)
	preds_after_proc = post_process_val(preds, cust_dict_val, val_id) # remove already exist products
	final_preds = [list(target_[pred]) for pred in preds_after_proc]
	score = 0.0 
	for i in range(val_Y.shape[0]):		
		actual = [target_[idx] for idx in range(n_products) if val_Y[i, idx] > 0]
		score += apk(actual, final_preds[i])
		if not i%1000:
			print "actual: ", actual
			print "pred: ", final_preds[i]
			print "score: ", apk(actual, final_preds[i])
	score /= val_Y.shape[0]
	del val_Y
	print("      MAP@7 score on val set is " + str(score))
	print(datetime.datetime.now()-start_time)

def train_and_test():

	start_time = datetime.datetime.now()
	data_path = "../input/"
	train_file =  open(data_path + "train_ver2.csv")

	print("training...")
	x_vars_list, y_vars_list, cust_dict = processData(train_file, {})
	train_X = np.array(x_vars_list)
	train_y = np.array(y_vars_list)
	#print(np.unique(train_y))
	del x_vars_list, y_vars_list
	
	#print(train_X.shape, train_y.shape)
	print(datetime.datetime.now()-start_time)
	test_file = open(data_path + "test_ver2.csv")
	x_vars_list, y_vars_list, cust_dict = processData(test_file, cust_dict)
	test_X = np.array(x_vars_list)
	del x_vars_list
	test_file.close()
	#print(test_X.shape)
	print(datetime.datetime.now()-start_time)

	print("Building model..")
	model = runXGB(train_X, train_y, seed_val=0)
	del train_X, train_y

	print("Predicting..")
	xgtest = xgb.DMatrix(test_X)
	preds = model.predict(xgtest)
	del test_X, xgtest
	print(datetime.datetime.now()-start_time)

	print("Getting the top products..")
	target_ = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	#preds = np.fliplr(preds)[:,:7]
	preds = np.fliplr(preds)
	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	preds_after_proc = post_process_pred(preds, test_id, train_file) # remove already exist products
	final_preds = [" ".join(list(target_[pred])) for pred in preds_after_proc]
	out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
	out_df = out_df[["ncodpers","added_products"]]
	out_df.to_csv('../pred/sub_xgb_new_v3.csv', index=False)
	train_file.close()
	print(datetime.datetime.now()-start_time)

if __name__ == "__main__":
	validation_by_month()
	





