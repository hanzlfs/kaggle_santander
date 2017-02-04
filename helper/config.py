

import lightgbm as lgb
from featureNames import FEATNAME


def get_config(filename = 'lgbm_submission', file_fi = 'lgbm_feature_importance', \
	model_path = None)):

	CONFIG_DATA = {
	"path_current_train": "train_current_lag_17_int.csv.gz",
	"path_previous_train": "train_previous_month_dataset.csv.gz",
	"path_current_test": "test_current_lag_17_int.csv.gz",
	"path_previous_test": "eval_previous_month_dataset.csv.gz",
	"model_path": model_path,
	"filename": filename,
	"file_fi": file_fi
	}

	CONFIG_FEATURES = {
		'train_month': [1,2,5,6,10,11,16], 
		'test_month': [17],
		'eval_month': None, 
		'features_profile_num': None, 
		'features_profile_cate': ['ind_actividad_cliente', 'renta','age','indrel',\
							'indrel_1mes','indext','segmento','month', 'ind_nuevo', 'tiprel_1mes'], 
		'features_product_yn': True,
		'features_product_change_yn': True,
		'features_product_lags': [2,3,4,5,6,7,8,9,10,11],# Allow lag combination 
		'features_profile_lags': [1,2,3,4,5,6,7,8,9,10],
		'features_profile_name_for_lags': ['ind_actividad_cliente', 'indrel', 'indext', 'indrel_1mes', 'segmento',\
								 'ind_nuevo', 'tiprel_1mes'], 
		'features_product_change_lags': {'lags':[2,3,4,5,6,7,8,9,10,11], 'use':'month-based'}, 
		'features_profile_change_lags': [0,1,2,3,4], 
		'features_profile_change': ['indrel', 'indext', 'indrel_1mes', 'segmento'],
		'features_combination_tensor': [['indrel','indrel_1mes','indext','segmento']],
		'features_interaction_mult': None,
		'features_product_lags_cummax': None,
		'features_time_sequence': [[col + '_L' + str(lag) for lag in [1,2,3]] for col in FEATNAME.COLNAMES['product']],
		'features_nn_used' = None,
		'features_nn_product_lags' = None,
		'features_nn_distance_lags' = None,
		'features_nn_summary_var_lags' = None,
		'features_nn_search_type_train' = None,
		# "global": search through all past/future months for nn 
		# "prev-mon": search through prev month for nn (when mon == 1, search through current month) 
		# "cur-mon": search through current month for nn

		'features_nn_search_type_test' = None,
		# "global" global search on train and test months
		# "global-test" global search just in test
		# "global-train" global search just in train
		# "prev-mon": search in train of month 16	
		'use_gbdt_feature': False		
	}

	CONFIG_PARAMS = {
		'n_estimators': 100,
		'nthread' = 8
	}

	return CONFIG_DATA, CONFIG_FEATURES, CONFIG_PARAMS

def get_model(model = None, param = None):
	if model == "LGBM":
		clf = lgb.LGBMClassifier(**param) # specify nthread = 8 to speed up 
	else:
		raise ValueError("other models not included as they are not effective")
	return clf












