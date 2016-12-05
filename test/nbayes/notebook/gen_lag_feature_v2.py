
# coding: utf-8

# ## Generate Lag Features from scratch

# In[1]:

########################
##########Import Modules
########################
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#get_ipython().magic(u'pylab inline')
#pylab.rcParams['figure.figsize'] = (10, 6)
import random
import time
from sklearn import preprocessing
from ast import literal_eval
import datetime
#from pandasql import sqldf
#from pandasql import load_meat, load_births

# In[2]:

####################################
print "load original train set#######"
####################################
#limit_rows   = 700000
#limit_people = 1.2e4
filename = "../input/train_ver2_cleaned.csv"
df           = pd.read_csv(filename,dtype={"sexo":str,\
                                            "ind_nuevo":str,\
                                               "ult_fec_cli_1t":str,\
                                                    "indext":str}, header = 0)

print "missing anything? in original ", df.isnull().any()
##########################################
print "load original nbayes datasets#######"
##########################################
dataset_root = '../'
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
                            "antiguedad":np.int16,
                            "ind_nuevo":'category',
                            'indrel_1mes':'category',
                            'tiprel_1mes':'category',
                            'canal_entrada':'category',
                            "age":np.int8,
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
limit_rows   = 20000000
start_time = time.time()
df_eval_current = pd.read_csv(dataset_root + "input/eval_current_month_dataset.csv.gz",
                           dtype=dictionary_types,
                           nrows=limit_rows)
print('It took %i seconds to load the dataset' % (time.time()-start_time))
start_time = time.time()
df_eval_previous = pd.read_csv(dataset_root + "input/eval_previous_month_dataset.csv.gz",
                           dtype=dictionary_types,
                           nrows=limit_rows)
print('It took %i seconds to load the dataset' % (time.time()-start_time))
df_train_current = pd.read_csv(dataset_root + "input/train_current_month_dataset.csv.gz",
                           dtype=dictionary_types,
                           nrows=limit_rows)
print('It took %i seconds to load the dataset' % (time.time()-start_time))
start_time = time.time()
df_train_previous = pd.read_csv(dataset_root + "input/train_previous_month_dataset.csv.gz",
                           dtype=dictionary_types,
                           nrows=limit_rows)
print('It took %i seconds to load the dataset' % (time.time()-start_time))
print(len(df_eval_current), len(df_eval_previous))
print(len(df_train_current), len(df_train_previous))

# In[167]:
print "######### joinning lag 1 - 5 data into current train/val"
max_lag = 5
print "####### create lag-i key for merge in df_eval_current #####"
for lag in range(1,max_lag+1):
    df_train_current['fecha_dato_lag_' + str(lag)] = df_train_current['fecha_dato'] - lag
#print df_train_current.head()
#print df_train_current.tail()

print "####### create lag-i key for merge in df_eval_current #####"
for lag in range(1,max_lag+1):
    df_eval_current['fecha_dato_lag_' + str(lag)] = df_eval_current['fecha_dato'] - lag
#print df_eval_current.head()
#print df_eval_current.tail()

# In[148]:

# In[168]:

print "### join with df_train_current on 'ncodpers' and 'fecha_dato'(w.lag)"
for lag in range(1, max_lag+1):
    print "appending lag: ", lag
    df_lag = df.add_suffix('_L' + str(lag))
    df_train_current = df_train_current.merge(df_lag, how = 'left',\
                                              left_on = ['ncodpers', 'fecha_dato_lag_' + str(lag)], \
                                                right_on = ['ncodpers_L' + str(lag), 'fecha_dato_L' + str(lag)])
    
print "### join with df_eval_current on 'ncodpers' and 'fecha_dato'(w.lag)"
for lag in range(1, max_lag+1):
    print "appending lag: ", lag
    df_lag = df.add_suffix('_L' + str(lag))
    df_eval_current = df_eval_current.merge(df_lag, how = 'left',\
                                              left_on = ['ncodpers', 'fecha_dato_lag_' + str(lag)], \
                                                right_on = ['ncodpers_L' + str(lag), 'fecha_dato_L' + str(lag)])

print "save data to file "
print(len(df_eval_current), len(df_eval_previous))
print(len(df_train_current), len(df_train_previous))


df_train_current.to_csv("../input/train_current_month_dataset_w_lag5.csv", index = False)
df_eval_current.to_csv("../input/eval_current_month_dataset_w_lag5.csv", index = False)




