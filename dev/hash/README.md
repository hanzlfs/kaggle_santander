## The advanced hashtrick 
#### generates hash tricked feature data used to test in model like xgb, lr, or FFM
#### follows the pipeline of [criteo](https://github.com/guestwalk/kaggle-2014-criteo)
- xgboost 
	- PreA.py Generate from raw feature GBDT features using **xgboost**, not criteo's gbdt library which cannot deal with multi-classification problem
	- PreB.py Transform the features, including GBDT features into high dimensional hash
	- count.py, count_target.py summarize feature frequent for filtering features in preA and preB
	- common.py, run.py utilities
- gbdt
	- Does the same pipeline with Criteo, except with our own data structure, feature names, etc for kaggle santander
	- Transform our problem into 24 binary classifications, feed into **Criteo's gbdt-hash-ffm pipeline** one by one
	- Summarize 24 prediction results

