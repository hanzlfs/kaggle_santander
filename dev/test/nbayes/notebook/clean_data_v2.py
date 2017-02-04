
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


# In[2]:

####################################
######load original train set#######
####################################
#limit_rows   = 700000
#limit_people = 1.2e4
filename = "../input/train_ver2.csv"
df           = pd.read_csv(filename,dtype={"sexo":str,\
                                            "ind_nuevo":str,\
                                               "ult_fec_cli_1t":str,\
                                                  "indext":str}, header = 0)



#####################################################
print "convert fencha_dato to months after 2015-01-28"
####################################################
def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month

def conv_to_mon(date_str):
    d0 = datetime.datetime.strptime('2015-01-28', "%Y-%m-%d")
    d1 = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return diff_month(d1, d0)
    
#print df["fecha_dato"].apply(conv_to_mon)
df["fecha_dato"] = df["fecha_dato"].apply(conv_to_mon)
#print df.head()
#print df.tail()
    

# I printed the values just to double check the dates were in standard Year-Month-Day format. I expect that customers will be more likely to buy products at certain months of the year (Christmas bonuses?), so let's add a month column. I don't think the month that they joined matters, so just do it for one.

# In[17]:

########################################################
print "add month column and covert 3 features to numeric#"
########################################################
df['month'] = (df.fecha_dato)%12 + 1
#print df[['month']].tail()
#df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
df["renta"] = pd.to_numeric(df["renta"], errors="coerce")
#print df[["age", "antiguedad", "renta"]].head()


# Are there any columns missing values?

# In[21]:

###############################
print "Impute Missing Values###"
###############################

######### 1. Check which columns contains missing vals############
#df.isnull().any()


# Definitely. Onto data cleaning.
# 
# ## Data Cleaning
# 
# Going down the list, start with `age`

# In[24]:

#################################
######### 2. Impute Age #########
#################################

######### fill na with median ##
df["age"].fillna(df["age"].median(),inplace=True)
df["age"] = df["age"].astype(int)
######### Cut off age boundary ##
df.loc[df.age < 0,"age"]  = 0
df.loc[df.age > 200,"age"]  = 200
#print df[['age']].describe()
#df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
#df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
#print df[["age"]].isnull().any()

# Looks better.  
# 
# Next `ind_nuevo`, which indicates whether a customer is new or not. How many missing values are there?

# In[30]:

#################################
print "2. Impute ind_nuevo #########"
#################################

#### How many missing vals? ####
#print df["ind_nuevo"].isnull().sum()
#months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
#print months_active
#months_active.max()
df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1


# Now, `antiguedad`

# In[35]:

#################################
print "2. Impute antiguedad ##"
#################################
#df.loc[df["antiguedad"]<0,"antiguedad"]
df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad <0, "antiguedad"]      = 0 # Thanks @StephenSmith for bug-find
#print np.sum(df["antiguedad"].isnull())
#df.loc[df["antiguedad"].isnull(),"ind_nuevo"].describe()


# Some entries don't have the date they joined the company. Just give them something in the middle of the pack

# In[38]:

#################################
print "3. fecha_alta #########"
#################################
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
#df["fecha_alta"].describe()


# Next is `indrel`, which indicates:
# 
# > 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
# 
# This sounds like a promising feature. I'm not sure if primary status is something the customer chooses or the company assigns, but either way it seems intuitive that customers who are dropping down are likely to have different purchasing behaviors than others.

# In[84]:

#################################
print "4. indrel #############"
#################################
#pd.Series([i for i in df.indrel]).value_counts()
df.loc[df.indrel.isnull(),"indrel"] = 1


# In[44]:

####################################################
print "5. drop tipodom and cod_prov #############"
###################################################
#df["tipodom"].dropna().value_counts()
df.drop(["tipodom","cod_prov"],axis=1,inplace=True)


# In[47]:

####################################################
print " 5. ind_actividad_cliente #############"
###################################################
df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].median()
#print np.sum(df["ind_actividad_cliente"].isnull())


# In[51]:

#################################
print "6. nomprov ############"
#################################
df.nomprov.unique()
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"


# Now for gross income, aka `renta`

# In[67]:

#################################
print " 7. renta ############"
#################################
#print df.renta.isnull().sum()
#df.loc[df.renta.notnull(),:].groupby("nomprov").agg([{"Sum":sum},{"Mean":mean}])
#incomes = df.loc[df.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":median}})
#incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
#incomes.reset_index(inplace=True)
#incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()],ordered=False)
#print incomes.head()

###### Replace nan with nomprov median#####
df.loc[df['renta'].isnull(), "renta"] = \
    df[['renta','nomprov']].groupby("nomprov").transform(lambda x: x.fillna(x.median())).loc[df['renta'].isnull(), 'renta']


# The next columns with missing data I'll look at are features, which are just a boolean indicator as to whether or not that product was owned that month. Starting with `ind_nomina_ult1`..

# In[74]:

##########################
print "products missing####"
##########################
#df.ind_nomina_ult1.isnull().sum()
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

# In[77]:

#############################
print "ALl Other missings####"
#############################
#print df.isnull().any()
string_data = df.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
for col in missing_columns:
    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
del string_data

# Okay, based on that and the definitions of each variable, I will fill the empty strings either with the most common value or create an unknown category based on what I think makes more sense.
# In[78]:
df.loc[df.indfall.isnull(),"indfall"] = "N"
df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
df.tiprel_1mes = df.tiprel_1mes.astype("category")

# As suggested by @StephenSmith
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

df.indrel_1mes.fillna("P",inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
df.indrel_1mes = df.indrel_1mes.astype("category")

unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"

# Let's check back to see if we missed anything

# In[86]:
print "missing anything? ", df.isnull().any()
#print df.indrel
df.to_csv("../input/train_ver2_cleaned.csv", index = False)






