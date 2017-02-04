class KnnRentaPercentile(object):
    # df must be the Percentile data with Age
    profile_cols = ['renta','age','indrel','indrel_1mes','indext','segmento']
    
    def _transform(self, X):
        from sklearn.preprocessing import OneHotEncoder
        one_hot_cols = ['indrel', 'indrel_1mes',  'indext', 'segmento']
        numeric_cols = ['renta', 'age']
        features = FeatureUnion(transformer_list =  oneHotEncoding(one_hot_cols, cat_dict) + \
                         [('other', ColumnSelectTransformer(numeric_cols))])
        
        return features.fit(X).transform(X).toarray()
    
    def _bin_process(self, df, bins):
        length = df.shape[0]
        renta_percentile = map(lambda x : int(x/ length*bins) + 1, np.linspace(0, length -1 , num=length))
        df['renta_percentile'] = renta_percentile
        df = df[['ncodpers', 'renta_percentile'] + profile_cols].drop_duplicates()
        return df
        
    def __init__(self, df, renta_bins):
        self.df = self._bin_process(df, bins = renta_bins)    
    
    def find_neighbors(self, n_neighbors, algo = 'kd_tree', metrics = 'cosine'):
        from sklearn.neighbors import NearestNeighbors
        X = self._transform(self.df[profile_cols])
        codpers = self.df['ncodpers'].tolist()
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X)
        neigh_ids = map(lambda x : knn.kneighbors(x, return_distance=False)[0].tolist(), X)
        
        def get_codpers(points):
            return map(lambda x : codpers[x], points)
            
        neighbor_codpers = map(lambda p : (p[0], get_codpers(p[1][1:])), zip(codpers, neigh_ids))
        return neighbor_codpers
        
    def test(self):
        X = self.df[profile_cols]
        ids = self.df['ncodpers']
        return ids, self._transform(X) 
