# generate sample csv
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
#from sklearn import preprocessing, ensemble

from common import *

def select_user_id(in_file_name, cust_ID, lst_mon_use):
	count_line = 0
	for row in csv.DictReader(in_file_name):
		if not count_line % 1000000:
			print "processed line: ", count_line
		count_line += 1
		sample = []
		if row['fecha_dato'] not in lst_mon_use:
			continue

		if row['ncodpers'] not in cust_ID:
			cust_ID[row['ncodpers']] = 1
		else:
			cust_ID[row['ncodpers']] += 1

def sample_with_user_id(in_file_name, out_file_name, cust_ID):

	header = ['ncodpers'] + ['fecha_dato'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + TARGET
	with open(out_file_name, 'w') as fp:
		wr = csv.writer(fp, delimiter=',')
		# write header
		wr.writerow(header)		
		count_line = 0
		for row in csv.DictReader(in_file_name):
			if not count_line % 1000000:
				print "processed line: ", count_line
			count_line += 1
			sample = []
			if row['ncodpers'] not in cust_ID:				
				continue

			r = np.random.rand()
			if r > 0.01:
				continue

			cust_id = row['ncodpers']
			sample.append(cust_id)
			sample.append(row['fecha_dato'])

			for col in FEAT_CAT:
				sample.append( row[col] ) # first categorical features added then numerical stacked after
			sample.append( row['age'] ) # any numerical value was rescaled into 0 to 1 
			sample.append( row['antiguedad'] )
			sample.append( row['renta'] )
			target_list = getTarget(row)
			sample.extend( target_list)
			wr.writerow(sample)	

def sampleData_for_val(in_file_name, out_file_name, cur_mon = 5, sample_per=0.01):
	# sample data into file for hash feature test
	np.random.seed(69069)
	
	header = ['ncodpers'] + ['fecha_dato'] + FEAT_CAT + \
				['age'] + ['antiguedad'] + ['renta'] + TARGET

	prev_mon = cur_mon - 1
	
	prev_mon_tr = '2015-' + str(prev_mon).zfill(2) + '-28'
	cur_mon_tr = '2015-' + str(cur_mon).zfill(2) + '-28'
	prev_mon_val = '2016-' + str(prev_mon).zfill(2) + '-28'	
	cur_mon_val	= '2016-' + str(cur_mon).zfill(2) + '-28'

	lst_mon_use = [prev_mon_tr, cur_mon_tr, prev_mon_val, cur_mon_val]

	with open(out_file_name, 'w') as fp:
		wr = csv.writer(fp, delimiter=',')
		# write header
		wr.writerow(header)		
		count_line = 0
		for row in csv.DictReader(in_file_name):
			if not count_line % 1000000:
				print "processed line: ", count_line
			count_line += 1
			sample = []
			if row['fecha_dato'] not in lst_mon_use:
				continue
			r = np.random.rand()
			if r > sample_per:
				continue
			cust_id = row['ncodpers']
			sample.append(cust_id)
			sample.append(row['fecha_dato'])

			for col in FEAT_CAT:
				sample.append( row[col] ) # first categorical features added then numerical stacked after
			sample.append( row['age'] ) # any numerical value was rescaled into 0 to 1 
			sample.append( row['antiguedad'] )
			sample.append( row['renta'] )
			target_list = getTarget(row)
			sample.extend( target_list)
			wr.writerow(sample)	

if __name__ == "__main__":	

	data_path = "../input/"
	in_file_name =  open(data_path + "train_ver2.csv")
	out_file_name = "../input/tr_sample_by_cid_small.csv" # sample train that have the same customer ids in lst_mon_use 
	lst_mon_use = ['2016-05-28']
	cust_ID = {}
	select_user_id(in_file_name, cust_ID, lst_mon_use)
	in_file_name.close()
	print "num of user ids in ", lst_mon_use, len(cust_ID)
	in_file_name =  open(data_path + "train_ver2.csv")
	sample_with_user_id(in_file_name, out_file_name, cust_ID)
	in_file_name.close()

