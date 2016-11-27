from sklearn.linear_model import LogisticRegression
import numpy as np

def logreg(X_train, y_train):
	clf = LogisticRegression(penalty='l1', C=0.1)
	clf.fit(X_train, y_train)
	print clf.classes_
	return clf

#clf.predict_proba(X)

def onehot(file_in_path, ndigit = 1e6, tr_or_val = 'train'):
	X = []
	y = []
	count = 0
	with open(file_in_path,'r') as f:
		for line in f:
			if count % 100 == 0:
				print "count line ", count
			if count > 2000:
				break
			count += 1

			X_cur = []
			line = line.strip().rstrip('\n')			
			lst = line.split()
			#print lst
			label = int(lst[0])			
			y.append(label)

			for astr in lst[1:]:
				feat = [0] * int(ndigit)
				hash_val = int(astr.split(':')[1]) % (int(ndigit) - 1)        		
				feat[hash_val] = 1
				X_cur.extend(feat)
			X.append(X_cur)
	X = np.array(X)
	y = np.array(y)
	print X.shape, y.shape
	print X[0:2, 0:100]
	print np.sum(X[0:2,:])
	return X, y

if __name__ == "__main__":
	file_in_path = "../input/tr.hashtk"
	X, y = onehot(file_in_path, ndigit = 1e3, tr_or_val = 'train')
	clf = logreg(X, y)
	print clf.score(X, y)




