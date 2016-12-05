
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




def clean_lag_feature(df, istrain = True):
    max_lag = 5
    ###############################
    ### only keep necessary columns
    ###############################
    cols_profile = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age',\
     'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi',\
      'indext', 'canal_entrada', 'indfall', 'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']

    cols_prod = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
                    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',\
                     'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',\
                      'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',\
                       'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    cols_buy = ['new_products', 'buy_class', 'n_new_products']

    cols_keep = []
    cols_keep.extend(cols_profile)
    if istrain:            
        cols_keep.extend(cols_buy)
    else:
        cols_keep.extend(['new_products'])
    #cols_keep.extend(cols_prod)
    for lag in range(1, max_lag+1):
        cols_keep.extend([str(x) + '_L' + str(lag) for x in cols_profile])
    for lag in range(1, max_lag+1):
        cols_keep.extend([str(x) + '_L' + str(lag) for x in cols_prod])
    df = df[cols_keep]


    ##############################################
    ### Fill missing profile with right-next-month 
    ##############################################
    for col in cols_profile:
        df[col + '_L' + str(1)].fillna(df[col], inplace = True)         
        for lag in range(2, max_lag + 1):
            df[col + '_L' + str(lag)].fillna(df[col + '_L' + str(lag - 1)], inplace = True)           
            

    ##############################################
    ### Fill missing product with 0 #############
    ##############################################
    for col in cols_prod:
        for lag in range(1, max_lag + 1):
            df[col + '_L' + str(lag)].fillna(0, inplace = True)   

    ###########################################################
    ### Check if there is still something missing #############
    ###########################################################

    print "missing anything? ", sum(df.isnull().any())
    return df

if __name__ == "__main__":
    # In[2]:
    print "read dataset "
    df_train_current = pd.read_csv("../input/train_current_month_dataset_w_lag5.csv", header = 0)
    df_eval_current = pd.read_csv("../input/eval_current_month_dataset_w_lag5.csv", header = 0)

    print "clean missing vals "
    df_train_current = clean_lag_feature(df_train_current,istrain = True)
    df_eval_current = clean_lag_feature(df_eval_current, istrain = False)

    print "write dataset "
    df_train_current.to_csv("../input/train_current_month_dataset_w_lag5_clean.csv", index = False)
    df_eval_current.to_csv("../input/eval_current_month_dataset_w_lag5_clean.csv", index = False)






