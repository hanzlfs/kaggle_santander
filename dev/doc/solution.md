## Santander Product Recommendation Solution
This summary provides an internal review of [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) solution. The purpose is to recap our pathway to find the optimum solution, find out key pros and cons in our approaches dealing with data comparing to top winner solutions. The public version of our solution will be reorganized from this one. We participated the contest ~Nov 22 and continued working at mostly full time on this for a month. 

#### Problem understanding

It took us a while to fully understand the problem in the corrent path. Originally we thought this would be a typical recommender system problem as indicated in the contest title, which led us to explore some successful past solutions for CTR. With the help with contest forum we categorized this problem more into traditional predictive modeling. 
Basically, santander is a bank with 24 major products and wish to rank recommended products to users. A major difference to typical recommender system is the time factor. We are given 17 months' user behavioral and profile data from 2015-01-28 to 2016-05-28, the target is to predict a group of possible **additional product** in 2016-06-28 in addition to what they already had at 2016-05-28 and submit the top 7 of them (according to the MAP@7 evaluation metric). This definition of target variable confuses us for a few days. What does it mean for "additional products"? If a user, for example, has a product at 2016-04-28 and does not appear at 2016-05-28, did this user still hold the product at 2016-05-28? It turned out this confuses a lot of participants. We stick with the comparison of previous month ONLY to define the target, and quickly became more confident on this with the help of forum and kernels. 
The kernel [when less is more](https://www.kaggle.com/sudalairajkumar/santander-product-recommendation/when-less-is-more) helped us to clear a lot of doubts and most of our early stage submissions were based on this script. 

#### Early stage exploration (Nov 22 - Nov 30)

- EDA

	We didn't have our own early stage EDA. The datasize is large with ~10,000,000 records in train and ~900,000 records in test. There were already excellend EDA kernels such as [Detailed Cleaning/Visualization (Python)](https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python). 

- Feature and Model

	There are two major efforts from Nov 22 to Nov 30: make submissions using [kernel script](https://www.kaggle.com/sudalairajkumar/santander-product-recommendation/when-less-is-more) and include the 3-idiots FFM framework. 
	We make all submissions based on [when less is more](https://www.kaggle.com/sudalairajkumar/santander-product-recommendation/when-less-is-more). The first few submissions gave us ~0.027 LB. There were no major improvements in this period. 
	In this script, numerical features like "age", "renta" and "antiguedad" were rescaled into 0 - 1, after an elementary median-missing-imputation. Categorical features were directly encoded with integer indices and treated as numerical in training. The selected model was xgboost for multi-classification (24 classes) with 90 num_rounds. 

- FFM, Tree features and feature hashing

	We wanted to test the effectiveness of tree leaf features from the idea of [Xinran He, et al](https://pdfs.semanticscholar.org/daf9/ed5dc6c6bad5367d7fd8561527da30e9b8dd.pdf) and the approach by [3 idiots FFM](https://github.com/guestwalk/kaggle-2014-criteo). The FFM framework has self-contained pre-processing and feature engineer pipelines. We wanted to borrow their idea of boosted tree-leaf features and feature hashing, and be able to use other models in them. Therefore we implemented the boosted tree-leaf features using xgboost and tested with Logistic Regression and KNN. We used the month-based val score as an evaluation. 
	Our major interest was to see how well could FFM perform on our data. We feed the xgboost-generated tree leaf feature into FFM package for 24 binary classification problems, yet FFM failed to train and gave NAN loss values. We then tried to use their framework from the beginning. However for several products where samples are extremely imbalanced (mostly 0 and only a few 1), FFM failed to work properly. The others, the convergence and loss-values of FFM was also not stable. We spent again ~2-3 days including FFM, along with binary-classification version of LR, XGB in our model and compare their performances using val score. Turned out FFM did nothing significantly better. So we paused using 3-idiots' FFM and gbdt features. We did make a submission with binary xgb, but it did not improve at all. 
	None of tree features, and FFM framework has made any improvements on LB.

- Numerical feature binning
	
	We tested binning numerical variables into categorical with 0,20,40,60,80 percentile cutoffs and did not make improvements. 

- Use all months' data to train
	
	We also attempted to use all past 17 months' data for train, instead of just using the 2015-06 data. The result stays the same, no improvements. To this point we seemed to get stuck. We spent quite a while on this making several mistakes with the prev-current month comparision when collecting training data. Our code started to grow very large and out of control. 

- Understanding the reason of feature hashing 
	
	We read blogs about feature hashing, starting to understand that the purpose of feature hashing is to reduce feature space dimension, where one-hot is infeasible, as apposed to augment feature dimension, for the latter is meaningless as feature hashing can by no means enrich feature information, as originally just another approach for encoding. This led us to tune our direction to augment original feature space instead of pursuing second layer feature tricks. 

- Validation
	
	We started to encounter the major difficulty thoughout the contest in stable validations. Due to the time partition nature of the problem, we initially followed the month-based-validation. We use 2015-05 data as the train, and make validation on 2016-05. To make submissions, we train on 2015-06 data and predict on 2016-06 data. It seems a reasonable validation as the author of the kernel as products buying has a similar distribution in the same month. However the validation score cannot match the LB. We do observe increasing val score with decreasing LB score. Most experienced the same problem according to the forum. We realized the difficulty is inherent as a good model for May may not work well for June. We did not spend time on consistent validations at this period. 

#### Middle Stage Work (Dec 1 to Dec 11)

- EDA

	We did not build our own EDA, and rely on the posted EDA on forum for data exploration. 

- Feature and Model

	The [ironbar/naive bayes](https://www.kaggle.com/c/santander-product-recommendation/forums/t/25871/naive-bayes-genetic-algorithms-lb-0-0272-cleaned-and-reduced-dataset-included) script caught our attention as it claimed to reduce the training data size with a fairly good LB score. So we tried it out. Replacing their NaiveBayes model with logistic regression, we were able to jump out of the 0.027 zone and reach 0.028, and then improved to 0.029 using light gbm. This immediately reminded us that this guy's feature engineer was very useful. 

	There were several differences ironbar has in his script. First he only kept the customers who actually did add some products in each month vs. previous month and removed any customer who either stays the same, or did not appear in prev/current month. This way the size of train set was reduced a lot. As Tom posted later, he did the samething. We started to realize this makes sense as the MAP@7 evaluation only focus on true possitives. 

	Another component is the feature of "product change". In addition, he did some tricks to change age, renta and antiguedad into categorical bins with artificial bin size. So far we stuck at 0.029 area and was not able to move to 0.030, until we spotted the 'lag' features in forum. 

	We immediately tried out the lag features. To do so we use ironbar's processed data as a baseline, and join from original data with the users with 0,1,2,3,4,5 time lags. We cleaned the original data using [Detailed Cleaning/Visualization (Python)](https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python). This gave us a 0.03 LB score. We pushed harder on lag features as it turned out to be very useful. We added selected profile lag 5 features in addition to product, the LB increased to 0.0301247. Then we chose a group of profile feature interactions (combinations actually) and continued to make improvements. Also inspired by the 'product_change' features by ironbar, we also added lag 5 for this and increased to 0.0301747. We reached to 0.0302029 with another 3 days work modifying product change generation. We splited the change into three scenarios, add, drop or maintain, without noticing it. 

	We did a lot of work to speed up our feature generation. The old way to generate product change feature and interaction feature is extremely slow. The 0.0302 run initially took ~5 hours to finish. We found out array matrix operation is, among all other attempts, the most efficient. Then we reduced to 888 secs to reach the same score. 

	The validation on selected months 5, 16 (2015-06, 2016-05) cannot match the trend of LB. But we did not invest time on building more reliable validation, instead we tried out ideas on more feature combination and feature sequence (combination with itself in selected history lags). But none of them worked out well. Without EDA, we tried to include all profile features but it brought our score to 0.029. 

	We were not condident for our data cleaning and join, so we spent the weekend re-do the job. We replaced our missing data cleaning with values in ironbar, but it seriously harms the performances. We got back to old data, and try some more feature combinations, but got no improvements. Not until the end of the weekend did we realize something worth doing but has not been done: the longer lag features. 

#### Final Stage Work (Dec 12 to Dec 21)
	
We reached 0.030389 with lag 10 product feature and continued to make improvements adding profile lags, product/profile change lags. We found out that any lag features beyond lag 11 would harm the performance. We guess this is due to seasonality effects, and longer lags gives less effective training samples. Adding the product change lag 10 features gave us 0.0304666 and adding lag 11 increased the score to 0.0305449. It seems the 12 month seasonality plays an important role here. 

We found any second-layer features upon product features would harm the results. For example, we added the product sequence features, the product combinations, and the cumulative min/max product features along with the lags, none of them brought any improvements. We reached a conclusion that product history features are so important that any modification would bring only disturb. 

We reached our best score 0.0306346 by adding some additional profile features, into lags as well. These features were added by some prior domain knowledge, and added one by one. We did not do any EDA on these features before reaching this score. 

We adopted another approach for validation, the plain 0.1/0.9 split. This gives faster validation but still not consistent with the LB trend. We simply record the val score. We stick with lgbm single model and parameter tuning does not bring any improvements. 

We did a lot of work in the final 5 days. We added the knn features inspired by facebook challange and tested out blending, but none of these worked out well. As we observed age, renta ranked high in lgbm feature importance, we tested smaller bin for them but no improvements. (Turned out the smaller bin age feature gave us the highest **private** score, 0.0309061). 

We started to do our own EDA on feature-target correlations following [Another EDA script](https://www.kaggle.com/alabsinatheer/santander-product-recommendation/comprehensive-exploration-and-visualization-1) and focused on plot features to show the distribution of **added** product vs. the values of categorical features or binned numerical features. This gave us many insights at the last 2 days of compitition and we mostly focused on adding remaining profile features based on these plots. However we were not able to move up one digit of score. 

#### Discussion 
	
In this contest we benifit a lot from the correct direction throughout the month: focus on feature engineer! We also found the reason why feature engineer is useful: by expanding feature information the potential of model performance will be better, which is inherently similar to why deep learning performs so well. 

The contest data is very sparse, and very hard to build validation. Therefore most of our improvements came from insight and trial-and-error. 

We need to work harder on:

- Code management and organization.
	There were multiple times the size of the code got out of control, and we had to memorize the configs using papers. We need a clear design from the beginning. 
- EDA
	We got hasty in the beginning as we want to see LB to increase quickly. We did not do our own EDA and only rely on others. EDA can bring some unique insight, though not always useful in a direct way.
- Validation
	Not until the last day did we found some partial consistency of LB and Validation. We could have done this earlier. 
- Coding speed and correctness
	Signicant amount of time was invested into debugging large and heavy code. Should have done unit tests on correctness and speed before pushing to production. 
	

#### Compare to top winners
	
We will summarize top winner solutions here in comparison with our solution. 





