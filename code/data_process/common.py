import numpy as np
import itertools

max_lag = 5
feat_profile_cat = []
feat_profile_cat_lag = []
feat_prod = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',\
             'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',\
             'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',\
             'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
feat_prod_change = []
feat_prod_lag = []

def create_interaction_list(profile_feature, is_prod_feature = False, \
                            profile_lag = [0], prod_lag = [1], \
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

#Basic dic type
dictionary_types = {"sexo":'category', "ult_fec_cli_1t":str,
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
                     "indrel_1mes":'category',
                     "tiprel_1mes":'category',
                     "canal_entrada":'category',
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
                     "product_buy":np.int8 }

def get_dict_type_w_lag():
	dictionary_types_w_lag = {}
	for k in dictionary_types:
		v = dictionary_types[k]
		if k not in dictionary_types_w_lag:
			dictionary_types_w_lag[k] = v
			for lag in range(1, max_lag+1):
				dictionary_types_w_lag[str(k) + '_L' + str(lag)] = v
	return dictionary_types_w_lag

#Define evaluation function
def eval_function_1(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_6',individual,verbose=False)
    return ret[0:1]

#Define evaluation function
def eval_function_2(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_8',individual,verbose=False)
    return [np.sum(ret)/2]

#Define evaluation function
def eval_function_3(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_9',individual,verbose=False)
    return ret[0:1]


def get_genomic_score(test_month, filename, genome, verbose=False):
    """
    Receives only test month and the genome
    Returns the score and saves the configuration and results in a file
    It's the same function as above but without training with month 16
    
    If the genome size is 35 then use_product is set to True
    If the genome size is 36 then all the parameters are in the search
    len(categorical_columns) = 18
    So we need a genome of 18+2+1 = 21
    """
    if verbose:
        print(genome)
    #Decide which train months to use, from 1 to 16
    if np.sum(genome[0:16]) > 0:
        used_months = np.array(range(1,17))[np.array(genome[0:16]) == 1]
        train_month = used_months
    else:
        #Select a random month
        used_months = np.random.randint(1,17,1)[0]
        train_month = [used_months]
    if verbose:
        print('train_month', train_month)
    #Decide wich category input columns to use
    categorical_columns = dataset.categorical_columns
    used_index = np.arange(len(categorical_columns))[
        np.array(genome[16:34]) == 1]
    input_columns = [categorical_columns[i] for i in used_index]
    if verbose:
        print('input_columns', input_columns)
    #Decide on using change columns and product as input
    use_change = genome[34] == 1
    #This allows to use a shorter genome to fix some properties
    if len(genome) >= 36: 
        use_product = genome[35] == 1
    else:
        use_product = True
    #Build message for training 
    msg ={'train_month':list(train_month),
          'eval_month':test_month,
          'input_columns':input_columns,
          'use_product':use_product,
          'use_change':use_change}
    if verbose:
        print(msg)
    ret = naive_bayes_workflow(msg)
    #Print and save to file 
    text = '\t'.join([str(a) for a in ret[0]]) + '\t'
    text += '%s\t%s\t' % ( use_change, use_product)
    if verbose:
        print(text)
    text += "','".join(input_columns)
    text += "\t" + ",".join([str(a) for a in train_month])
    text += '\n'
    with open(dataset_root+'logs/%s.log' % filename, 'a') as f:
        f.write(text)
        
    return ret[0]

#Define evaluation function
def eval_function_4(individual):
    """
    Tries to optimize just the training score
    """
    ret = get_genomic_score([5,16],'genetic_search_10',individual,verbose=False)
    return [np.sum(ret)/2]

if __name__ == "__main__":
	print get_dict_type_w_lag()
