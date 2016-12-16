## Generate knn feature
- **1.Data Preparation**
- [X] Generate bin-ids according to age and renta for each user
- [X] Generate nearest neighbor ncodpers for each ncodper, save to file
	- [X] When the query is from test set (month 17), we find its neighbors in month 16 train set, because almost all users in test set are inactive.
	- [ ] When the query is from train set (month 1 - 16), we SHOULD have found its neighbors in previous months (0 - 15), however, **there are accasions that bin-ids in current month cannot find matches in previous month**. So currently I find neighbors from current month instead of from previous month. If speed permit, we can probably remove the bin and find NN directly. 

- **2. Feature Pipeline**
- [ ] Append the neighbor ncodpers to the end of train and test set
- [ ] Join selected neighbor features into train and test set
- [ ] Test performance using LGBM and submit, collect LB score and Validation score
