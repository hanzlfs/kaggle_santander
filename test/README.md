## general models training, validation and submission
#### test multiple feature tricks
- src/model_fn.py generate validation and submission using xgb, allows to add and test different feature tricks
	- [X] Add month as a feature
	- [X] Use all year data for training
	- [X] Allows to one-hot any combinations of existing features
	- [X] Binning numerical features 
	**.......Still in progress......**
	- [ ] Summarize feature frequency and trim non-frequent features, especially for those have too many field
	- [ ] Allows to change the binning strategy for numerical features
	- [ ] Keep binary feature as it is, do not one-hot it
	- [ ] XGB/RF feature importance, give insight to make additional features
	- [ ] *user purchase history features, i.e. number of purchases/adds, length of holds, combanition of all owned products into a string, i.e. 'prod1-prod2-prod3', then hash it; sequence of product history for past 3 months, '010','110','011', then hash it, etc
	- [ ] *GBDT tree leaf indices as features
	- ...
- src/uTILS.py; src/common_fn.py: utilities
- src/gen_csv.py: used to sample a small dataset from train for feature analysis
- Discussions on Kaggle @BreakfastPirate
	- I'd rather not say a lot more at this point. I can confirm that I did get 0.030 with a training set that had about 46,000 rows from about 37,000 accounts (the additional rows being because some accounts had multiple products). Feature engineering is important. Going back and looking at the raw data for a few accounts is helpful.
	- What are new features did you add? (only if you don't mind sharing of course) I did add features. But nothing too sophisticated. I'd rather leave that for people to discover on their own. Obviously you need to know what each account did in the previous month to know if each product is new.
