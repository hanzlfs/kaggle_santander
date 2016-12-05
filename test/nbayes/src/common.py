"""
Definition of constants and common utilities
"""
import numpy as np
import itertools
"""
max number of lags
"""
max_lag = 5

"""
Define different feature groups
"""
feat_profile_cat = []
feat_profile_cat_lag = []
feat_prod = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
                    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',\
                     'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',\
                      'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',\
                       'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
feat_prod_change = []
feat_prod_lag = []

def create_interaction_list(profile_feature, is_prod_feature = False, profile_lag = [0], prod_lag = [1], \
                                   interact_order = 2, interact_option = 'individual'):
       """
       create interaction groups given experiment conditions
       
       profile_feature: list of profile features in interaction
       is_prod_feature: whether or not to include 24 product features 
       profile_lag: list of profile lags in interaction, default [0]
       prod_lag: default [1], note that for lag 1 product feature is from train_prev and eval_prev
       interact_order: default 2, the order to pair features 
       interact_option: default 'individual', i.e. profile and prod features interact with themselves only
                            if 'mutual', get profile-prod pairwise interactions, in this case interact_order = 2
                            if 'All', merge profile and prod features and form interactions with order
       return [[group1],[group2] ... ]
       """
       if (not profile_feature) and (not is_prod_feature):
              return []
       result = [] 

       profile_lag = [x for x in profile_lag if x > 0] # the real lags
       prod_lag = [x for x in prod_lag if x > 1] # the real lags

       prod_feature = feat_prod
       if interact_option == 'all':
              feature_list = []
              if profile_feature:
                     feature_list.extend(profile_feature)
                     if profile_lag:
                            feature_list.extend([str(x) + '_L' + str(lag) for x in profile_feature for lag in profile_lag])
              if is_prod_feature:
                     feature_list.extend(prod_feature)
                     if prod_lag:
                            feature_list.extend([str(x) + '_L' + str(lag) for x in prod_feature for lag in prod_lag])
              result.extend(list(itertools.combinations(feature_list, interact_order)))

       if interact_option == 'individual':              
              if profile_feature:
                     feature_list = []
                     feature_list.extend(profile_feature)
                     if profile_lag:
                            feature_list.extend([str(x) + '_L' + str(lag) for x in profile_feature for lag in profile_lag])
                     result.extend(list(itertools.combinations(feature_list, interact_order)))
              if is_prod_feature:
                     feature_list = []
                     feature_list.extend(prod_feature)
                     if prod_lag:
                            feature_list.extend([str(x) + '_L' + str(lag) for x in prod_feature for lag in prod_lag])
                     result.extend(list(itertools.combinations(profile_feature, interact_order)))
       
       return map(list, result)

def get_feat_prod_lag(lags):
	# get lag product features with selected lags 
	feat = []
	for lag in lags:
		feat.extend([str(x) + '_L' + str(lag) for x in feat_prod])
	return feat

def get_feat_lag(features, lags):
	# get general lag feature in lags for feats
	feat = []
	for lag in lags:
		feat.extend([str(x) + '_L' + str(lag) for x in features])
	return feat

def isInt(value):
	try:
		int(value)
		return True
	except ValueError:
		return False

"""
Basic dic type
"""
dictionary_types = {
                            "sexo":'category',
                            "ult_fec_cli_1t":str,
                            "indresi":'category',
                            "indext":'category',
                            "indrel":'category',
                            "indfall":'category',
                            "nomprov":'category',
                            "segmento":'category',
                            "ind_empleado":'category',
                            "pais_residencia":'category',
                            "antiguedad":float,
                            "ind_nuevo":'category',
                            'indrel_1mes':'category',
                            'tiprel_1mes':'category',
                            'canal_entrada':'category',
                            "age":float,
                            "ind_actividad_cliente":'category',
                            "ind_ahor_fin_ult1":np.int8,
                            "ind_aval_fin_ult1":np.int8,
                            "ind_cco_fin_ult1":np.int8,
                            "ind_cder_fin_ult1":np.int8,
                            "ind_cno_fin_ult1":np.int8,
                            "ind_ctju_fin_ult1":np.int8,
                            "ind_ctma_fin_ult1":np.int8,
                            "ind_ctop_fin_ult1":np.int8,
                            "ind_ctpp_fin_ult1":np.int8,
                            "ind_deco_fin_ult1":np.int8,
                            "ind_deme_fin_ult1":np.int8,
                            "ind_dela_fin_ult1":np.int8,
                            "ind_ecue_fin_ult1":np.int8,
                            "ind_fond_fin_ult1":np.int8,
                            "ind_hip_fin_ult1":np.int8,
                            "ind_plan_fin_ult1":np.int8,
                            "ind_pres_fin_ult1":np.int8,
                            "ind_reca_fin_ult1":np.int8,
                            "ind_tjcr_fin_ult1":np.int8,
                            "ind_valo_fin_ult1":np.int8,
                            "ind_viv_fin_ult1":np.int8,
                            "ind_nomina_ult1":np.int8,
                            "ind_nom_pens_ult1":np.int8,
                            "ind_recibo_ult1":np.int8,

                            "ind_ahor_fin_ult1_change":'category',
                            "ind_aval_fin_ult1_change":'category',
                            "ind_cco_fin_ult1_change":'category',
                            "ind_cder_fin_ult1_change":'category',
                            "ind_cno_fin_ult1_change":'category',
                            "ind_ctju_fin_ult1_change":'category',
                            "ind_ctma_fin_ult1_change":'category',
                            "ind_ctop_fin_ult1_change":'category',
                            "ind_ctpp_fin_ult1_change":'category',
                            "ind_deco_fin_ult1_change":'category',
                            "ind_deme_fin_ult1_change":'category',
                            "ind_dela_fin_ult1_change":'category',
                            "ind_ecue_fin_ult1_change":'category',
                            "ind_fond_fin_ult1_change":'category',
                            "ind_hip_fin_ult1_change":'category',
                            "ind_plan_fin_ult1_change":'category',
                            "ind_pres_fin_ult1_change":'category',
                            "ind_reca_fin_ult1_change":'category',
                            "ind_tjcr_fin_ult1_change":'category',
                            "ind_valo_fin_ult1_change":'category',
                            "ind_viv_fin_ult1_change":'category',
                            "ind_nomina_ult1_change":'category',
                            "ind_nom_pens_ult1_change":'category',
                            "ind_recibo_ult1_change":'category',
                            'product_buy':np.int8,
}

def get_dict_type_w_lag():
	dictionary_types_w_lag = {}
	for k in dictionary_types:
		v = dictionary_types[k]
		if k not in dictionary_types_w_lag:
			dictionary_types_w_lag[k] = v
			for lag in range(1, max_lag+1):
				dictionary_types_w_lag[str(k) + '_L' + str(lag)] = v
	return dictionary_types_w_lag


if __name__ == "__main__":
	print get_dict_type_w_lag()



