
class featureNames(object):
	"""
	pre-define feature column names
	"""

	def __init__(self, lags = None, changes = True):

		self.COLNAMES = {}
		self.__getnames_product(lags = lags, changes = changes)
		self.__getnames_profile(lags = lags)
	
	def __getnames_product(self, lags, changes):
		"""
		get product feature names, including lags and changes
		"""

		self.COLNAMES.update({"product": ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
				'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',\
					'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',\
						'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',\
							'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']})

		if lags:
			self.COLNAMES.update({"product_lag": [col + '_L' + str(lag) \
				for col in self.COLNAMES["product"] for lag in lags]})

		if changes:
			self.COLNAMES.update({"product_change": [col + '_change' for col in self.COLNAMES["product"]]})
			self.COLNAMES.update({"product_change_lag": [col + '_L' + str(lag) \
				for col in self.COLNAMES["product_change"] for lag in lags]})

	def __getnames_profile(self, lags):
		"""
		get customer profile feature names, including lags and changes for categorical
		"""
		self.COLNAMES.update({"profile_num": ['age','antiguedad','renta'],\
								"profile_cate": ['ind_empleado', 'pais_residencia', 'sexo',\
									'ind_nuevo', 'indrel', 'indrel_1mes', \
									'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', \
									'ind_actividad_cliente', 'segmento']})
		
		if lags:
			self.COLNAMES.update({"profile_cate_lag": [col + '_L' + str(lag) \
									for col in self.COLNAMES["profile_cate"] for lag in lags]})

# create global object
FEATNAME = featureNames(lags = range(0, 18), changes = True)




