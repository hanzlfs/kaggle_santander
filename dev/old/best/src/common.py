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
mapping_dict = {
'ind_empleado'  : {'N':0, 'B':1, 'F':2, 'A':3, 'S':4},
'sexo'          : {'V':0, 'H':1},
'ind_nuevo'     : {'0':0, '1':1},
'indrel'        : {'1':0, '99':1},
'indrel_1mes'   : {'1.0':0, '1':0, '2.0':1, '2':1, '3.0':2, '3':2, '4.0':3, '4':3, 'P':4},
'tiprel_1mes'   : {'I':0, 'A':1, 'P':2, 'R':3, 'N':4},
'indresi'       : {'S':0, 'N':1},
'indext'        : {'S':0, 'N':1},
'conyuemp'      : {'S':0, 'N':1},
'indfall'       : {'S':0, 'N':1},
'tipodom'       : {'1':0},
'ind_actividad_cliente' : {'0':0, '1':1},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11},
'nomprov' : {'ZARAGOZA': 52, 'BURGOS': 11, 'GRANADA': 23, 'UNKNOWN': 0, 'CIUDAD REAL': 17, 'BADAJOZ': 7, 'JAEN': 27, 'LEON': 28, 'SORIA': 45, 'SANTA CRUZ DE TENERIFE': 42, 'CEUTA': 16, 'HUESCA': 26, 'VALLADOLID': 50, 'LERIDA': 29, 'ZAMORA': 51, 'CUENCA': 20, 'RIOJA, LA': 40, 'PONTEVEDRA': 39, 'MELILLA': 33, 'TARRAGONA': 46, 'CORDOBA': 18, 'SEVILLA': 44, 'ALICANTE': 3, 'CASTELLON': 15, 'MADRID': 31, 'OURENSE': 36, 'VALENCIA': 49, 'TOLEDO': 48, 'HUELVA': 25, 'ALBACETE': 2, 'CORUNA, A': 19, 'CADIZ': 13, 'GIRONA': 22, 'TERUEL': 47, 'AVILA': 6, 'BARCELONA': 9, 'SEGOVIA': 43, 'NAVARRA': 35, 'MALAGA': 32, 'SALAMANCA': 41, 'PALENCIA': 37, 'ALMERIA': 4, 'MURCIA': 34, 'GUADALAJARA': 24, 'ASTURIAS': 5, 'BALEARS, ILLES': 8, 'ALAVA': 1, 'LUGO': 30, 'CANTABRIA': 14, 'CACERES': 12, 'PALMAS, LAS': 38, 'GIPUZKOA': 21, 'BIZKAIA': 10}
}

# customer profile features 
feat_profile = ['ind_empleado', 'pais_residencia', 'sexo', \
'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', \
'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', \
'ind_actividad_cliente', 'renta', 'segmento'] 

# target quantities of product (cannot be used for train)
target_prod = ['new_products', 'buy_class', 'n_new_products']

feat_prod = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
				'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',\
					'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',\
						'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',\
							'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

def create_interaction_list(profile_feature, is_prod_feature = False, profile_lag = [0], prod_lag = [1], \
																	 interact_order = 2, interact_option = 'individual'):
	"""
	Modify this function if you think it could be more general or simple

	"""
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

def get_feat_change(features, lags):
	# get status change feature names
	feat_names = [col + '_change_L' + str(lag) for col in features for lag in lags]
	return feat_names

def get_feat_prod_change(lags):
	# get status change product names
	feat_names = [col + x + str(lag) for col in feat_prod \
												for x in ['_change_L','_add_L','_drop_L'] for lag in lags]
	return feat_names

def isInt(value):
	try:
		int(value)
		return True
	except ValueError:
		return False

"""
Basic dic type
"""

feat_profile = ['ind_empleado', 'pais_residencia', 'sexo', \
'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', \
'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', \
'ind_actividad_cliente', 'renta', 'segmento'] 

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
		"ind_nuevo":'category',
		'indrel_1mes':'category',
		'tiprel_1mes':'category',
		'canal_entrada':'category',
		"age":float,
		"renta":float,
		"antiguedad":float,
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
			for lag in range(0, max_lag+1):
				dictionary_types_w_lag[str(k) + '_L' + str(lag)] = v
	return dictionary_types_w_lag

if __name__ == "__main__":
	print get_dict_type_w_lag()



