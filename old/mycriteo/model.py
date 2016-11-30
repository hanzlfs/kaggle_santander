from sklearn.linear_model import LogisticRegression
import numpy as np
from common import *

def eval(y_true, preds):
# actual: nsample by nclass binary
# pred: nsample by nclass 0-1 probability
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)[:,:7]
	#final_preds = [list(TARGET[pred]) for pred in preds]
	score = 0.0 
	for i in range(y_true.shape[0]):		
		actual = [TARGET[idx] for idx in range(n_products) if y_true[i, idx] > 0]
		final_pred = [TARGET[idx] for idx in list(preds[i, :])]
		score += apk(actual, final_pred)
		"""
		if not i%10:
			print "actual: ", actual
			#print "target_", target_
			print "val_Y: ", y_true[i, :]
			print "pred: ", final_pred
			print "score: ", apk(actual, final_pred)
		"""
		
	score /= y_true.shape[0]
	#print("      MAP@7 score on val set is " + str(score))
	return score

def collect_ffm_pred():
	# collect ffm prediction on tr and val
	return 0

def logreg(X_train, y_train):
	clf = LogisticRegression(penalty='l1', C=0.1)
	clf.fit(X_train, y_train)
	print "class order ", clf.classes_
	return clf

#clf.predict_proba(X)


def olddata(file_in_path, tr_or_val, nlimit = 50):
	X = []
	y = []
	count = 0
	f = open_with_first_line_skipped(file_in_path, skip=True)
	for line in f:
		if count % 100 == 0:
			print "count line ", count
		if count > nlimit:
			break
		count += 1

		X_cur = []
		line = line.strip().rstrip('\n')			
		lst = line.split(',')
		#print lst
		if tr_or_val == "train":
			label = int(lst[0])			
			y.append(label)
			lst_feat = lst[1:]
		else:
			actual = map(int, lst[0:24])
			y.append(actual)
			lst_feat = lst[25:]		
		X_cur.extend(map(float,lst_feat))

		X.append(X_cur)
	X = np.array(X)
	y = np.array(y)
	#print X.shape, y.shape
	#print X[0:2, 0:100]
	#print np.sum(X[0:2,:])
	return X, y

def onehot(file_in_path, ndigit = 1e6, tr_or_val = 'train', nlimit = 50):
	X = []
	y = []
	count = 0
	with open(file_in_path,'r') as f:
		for line in f:
			if count % 100 == 0:
				print "count line ", count
			if count > nlimit:
				break
			count += 1

			X_cur = []
			line = line.strip().rstrip('\n')			
			lst = line.split()
			#print lst
			if tr_or_val == "train":
				label = int(lst[0])			
				y.append(label)
				lst_feat = lst[1:]
			else:
				actual = map(int, lst[0:24])
				y.append(actual)
				lst_feat = lst[25:]

			for astr in lst_feat:
				feat = [0] * int(ndigit)
				hash_val = int(astr.split(':')[1]) % (int(ndigit) - 1)        		
				feat[hash_val] = 1
				X_cur.extend(feat)
			X.append(X_cur)
	X = np.array(X)
	y = np.array(y)
	#print X.shape, y.shape
	#print X[0:2, 0:100]
	#print np.sum(X[0:2,:])
	return X, y

def gen_binary_class_files(file_in_path, tr_or_val = 'train'):
	# from any train/val set with label(s) + feature, n-classification, generaten copies of the files, only with 
	# label = 0/1 for each product, for those not appeared, fill label = 0 for all samples

	# popuate data
	X = []
	y = []
	with open(file_in_path,'r') as f:
		for line in f:
			X_cur = []
			line = line.strip().rstrip('\n')			
			lst = line.split()
			#print lst
			if tr_or_val == "train":
				label = int(lst[0])			
				y.append(label)
				lst_feat = lst[1:]
			else:
				actual = map(int, lst[0:24])
				y.append(actual)
				lst_feat = lst[25:]

			X_cur.extend(lst_feat)
			X.append(X_cur)	
    
	# write into 24 files
	for i in range(24): #products 1 to 24
		count_pos = 0
		count_neg = 0
		count_total = 0
		print "writing to sub file ", i
		if tr_or_val == 'train':
			out_file = '../input/tr_'+ str(i) + '.hashtk'
		else:
			out_file = '../input/val_'+ str(i) + '.hashtk'

		with open(out_file, 'w') as fw:
			for l in range(len(X)): 
				line_w_lst = []
				if tr_or_val == 'train':
					label_val = int(y[l] == i)					
				else:
					label_val = int(y[l][i])
				
				#if label_val == 0: NO need to change 0 to -1 for ffm
				#	label_val = -1
				if label_val == 1:
					count_pos += 1
				else:
					count_neg += 1
				count_total += 1

				line_w_lst.append(label_val)
				line_w_lst.extend(X[l])
				line_w = ' '.join(map(str, line_w_lst))
				fw.write(line_w + '\n')
		print "pos instances ", count_pos, " neg instances ", count_neg, " total instances ", count_total

def test_baseline(nlimit = 50):

	file_in_path = "../input/tr_sample_feature_v2.csv"
	X, y = olddata(file_in_path, tr_or_val = 'train', nlimit = nlimit)
	clf = logreg(X, y)
	print "baseline train score  ", clf.score(X, y)

	file_in_path = "../input/val_sample_feature_v2.csv"
	X_val, y_val = olddata(file_in_path, tr_or_val = 'val', nlimit = nlimit)
	pred = clf.predict_proba(X_val)
	score = eval(y_val, pred)

	print "baseline map@7 = ", score

def test_hash(nlimit = 50):
	file_in_path = "../input/tr.hashtk"
	X, y = onehot(file_in_path, ndigit = 1e3, tr_or_val = 'train', nlimit = nlimit)
	clf = logreg(X, y)
	print "hashtrick train score  ", clf.score(X, y)

	file_in_path = "../input/val.hashtk"
	X_val, y_val = onehot(file_in_path, ndigit = 1e3, tr_or_val = 'val', nlimit = nlimit)
	pred = clf.predict_proba(X_val)
	score = eval(y_val, pred)

	print "hashtrick map@7 = ", score


if __name__ == "__main__":
	"""
	nlimit = 2000
	test_baseline(nlimit = nlimit)
	test_hash(nlimit = nlimit)
	"""
	file_in_path = "../input/tr.hashtk"
	gen_binary_class_files(file_in_path, tr_or_val = 'train')

	file_in_path = "../input/val.hashtk"
	gen_binary_class_files(file_in_path, tr_or_val = 'val')
	






