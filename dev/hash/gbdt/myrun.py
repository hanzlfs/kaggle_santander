#!/usr/bin/env python3

import subprocess, sys, os, time

def create_link():
	# create symbol links 
	if not os.path.isfile('gbdt'): 
		cmd = 'ln -s ../../solvers/gbdt/gbdt gbdt'
		subprocess.call(cmd, shell=True)

	if not os.path.isfile('ffm-train'): 
		cmd = 'ln -s ../../solvers/libffm-1.13/ffm-train ffm-train'
		subprocess.call(cmd, shell=True)

	if not os.path.isfile('ffm-predict'): 
		cmd = 'ln -s ../../solvers/libffm-1.13/ffm-predict ffm-predict'
		subprocess.call(cmd, shell=True)

def count_feature():
	cmd = 'python count.py ../input/tr_sample_feature_v3.csv > ../input/fc.trva.t10.txt'
	subprocess.call(cmd, shell=True) 

	cmd = 'python count_target.py ../input/tr_sample_feature_v3.csv > ../input/fc.trva.t10.target.txt'
	subprocess.call(cmd, shell=True) 

def pre_A(file_type = 'train', i_prod = 0):
	if file_type == 'train':
		csv_input = "../input/tr_sample_feature_v3.csv"
		out_dense = '../input/tr' + str(i_prod) + '.gbdt.dense'
		out_sparse = '../input/tr' + str(i_prod) + '.gbdt.sparse'
	else:
		csv_input = "../input/val_sample_feature_v3.csv"
		out_dense = '../input/val' + str(i_prod) + '.gbdt.dense'
		out_sparse = '../input/val' + str(i_prod) + '.gbdt.sparse'

	cmd = 'python mypre-a.py -f ' + file_type + ' -i ' + str(i_prod) + ' ' \
			+ csv_input + ' ' + out_dense + ' ' + out_sparse
	subprocess.call(cmd, shell=True) 

def gbdt_feature(i_prod = 0):
	val_dense = '../input/val' + str(i_prod) + '.gbdt.dense'
	val_sparse = '../input/val' + str(i_prod) + '.gbdt.sparse'
	tr_dense = '../input/tr' + str(i_prod) + '.gbdt.dense'
	tr_sparse = '../input/tr' + str(i_prod) + '.gbdt.sparse'
	val_out = '../input/val' + str(i_prod) + '.gbdt.out'
	tr_out = '../input/tr' + str(i_prod) + '.gbdt.out'

	str_appd = ' '.join([val_dense, val_sparse, tr_dense, tr_sparse, val_out, tr_out])
	cmd = './gbdt -t 30 -d 7 -s 1 ' + str_appd 
	subprocess.call(cmd, shell=True)

	str_rm = ' '.join([val_dense, val_sparse, tr_dense, tr_sparse])
	cmd = 'rm -f ' + str_rm
	subprocess.call(cmd, shell=True)


def pre_B(file_type = 'train', i_prod = 0):
	if file_type == 'train':
		csv_input = "../input/tr_sample_feature_v3.csv"
		gbdt_input = '../input/tr' + str(i_prod) + '.gbdt.out'
		ffm_output = '../input/tr' + str(i_prod) + '.ffm'
	else:
		csv_input = "../input/val_sample_feature_v3.csv"
		gbdt_input = '../input/val' + str(i_prod) + '.gbdt.out'
		ffm_output = '../input/val' + str(i_prod) + '.ffm'

	cmd = 'python mypre-b.py -f ' + file_type + ' -i ' + str(i_prod) + ' ' + ' '.join([csv_input, gbdt_input, ffm_output])
	subprocess.call(cmd, shell=True) 

def ffm_(i_prod = 0):

	start = time.time()

	tr_in = '../input/tr' + str(i_prod) + '.ffm'
	val_in = '../input/val' + str(i_prod) + '.ffm'
	tr_out = '../output/tr' + str(i_prod) + '.out'		
	val_out = '../output/val' + str(i_prod) + '.out'	

	cmd = './ffm-train -l 0.0002 --auto-stop -k 4 -t 18 -s 1 -p ' + val_in + ' ' + tr_in +  ' model'
	subprocess.call(cmd, shell=True)

	cmd = './ffm-predict ' + tr_in + ' model ' + tr_out
	subprocess.call(cmd, shell=True)

	cmd = './ffm-predict ' + val_in + ' model ' + val_out
	subprocess.call(cmd, shell=True)

	cmd = 'rm -f model'
	subprocess.call(cmd, shell=True)

	print('time used in ffm = {0:.0f}'.format(time.time()-start))

if __name__ == "__main__":

	# preprocess
	print "preprocess === "
	create_link()
	count_feature()

	# preA
	print "preA === "	
	for i_prod in range(24):

	#i_prod = 23
		print "prod number selected = ", i_prod
		for file_type in ['train', 'val']:
			pre_A(file_type = file_type, i_prod = i_prod)

		# gbdt feature
		print "gbdt feature === "
		gbdt_feature(i_prod = i_prod)

		# preB
		print "preB === "
		for file_type in ['train', 'val']:
			pre_B(file_type = file_type, i_prod = i_prod)

		#ffm
		print "running ffm === "
		ffm_(i_prod = i_prod)

