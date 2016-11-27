## feature engineer
#### GBDT hash feature processing
- [kaggle-2014-criteo](https://github.com/guestwalk/kaggle-2014-criteo/tree/7f19bce9ad46670a9b80f37c951aab10621a2b85)
- **1. Modify code with our features**
	- Our Full feature set
	```
	["fecha_dato","ncodpers","ind_empleado","pais_residencia","sexo","age","fecha_alta","ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","renta","segmento","ind_ahor_fin_ult1","ind_aval_fin_ult1","ind_cco_fin_ult1","ind_cder_fin_ult1","ind_cno_fin_ult1","ind_ctju_fin_ult1","ind_ctma_fin_ult1","ind_ctop_fin_ult1","ind_ctpp_fin_ult1","ind_deco_fin_ult1","ind_deme_fin_ult1","ind_dela_fin_ult1","ind_ecue_fin_ult1","ind_fond_fin_ult1","ind_hip_fin_ult1","ind_plan_fin_ult1","ind_pres_fin_ult1","ind_reca_fin_ult1","ind_tjcr_fin_ult1","ind_valo_fin_ult1","ind_viv_fin_ult1","ind_nomina_ult1","ind_nom_pens_ult1","ind_recibo_ult1"]

	```
	- Need to change
		- converters/common.py: line 3 
		```
		HEADER="Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"
		```
		- converters/pre-a.py: line 17
		```
		target_cat_feats = ['C9-a73ee510', 'C22-', 'C17-e5ba7672', 'C26-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53', 'C14-1adce6ef', 'C25-001f3601', 'C22-ad3062eb', 'C17-07c540c4', 'C6-', 'C23-423fab69', 'C17-d4bb7bd8', 'C2-38a947a1', 'C25-e8b83407', 'C9-7cc72ec2']
		```
		- *Optional converters/txt2csv.py lines 14&17
		
- **2. Test use the sample feture data**
	- tr_sample_feature.csv, "label" -> label, all other fields are features
	- val_sample_feature.csv, "lb_*" -> labels, all other fields are features
		
#### FFM
- 1. Recommender Systems: Advances in Collaborative Filtering  http://www.slideshare.net/ChangsungMoon/recommender-systems-advances-in-collaborative-filtering
- 2. Factorization Machines in Python https://github.com/coreylynch/pyFM
- 3. Factorization Machines: A New Way of Looking at Machine Learning(must read) https://securityintelligence.com/factorization-machines-a-new-way-of-looking-at-machine-learning/

#### sparse learning

- 1.  Probabilistic Multi-Label Classification with Sparse Feature Learning https://pdfs.semanticscholar.org/2784/78d79d97a0c52b1ad05d1d850733d6fff7c4.pdf
- 2.  Advances in Collaborative Filtering  https://datajobs.com/data-science-repo/Collaborative-Filtering-[Koren-and-Bell].pdf

#### Vowpal Wabbit (multiclassification possible)
- Vowpal Wabbit is a fast machine learning library for online learning
- Resource: (1) http://hunch.net/~vw/ (2) https://github.com/JohnLangford/vowpal_wabbit (3) https://medium.com/@chris_bour/what-i-learned-from-the-kaggle-criteo-data-science-odyssey-b7d1ba980e6#.mpl9u6fdh

