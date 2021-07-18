import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# reduce dimensions with PCA for easier compute
def reduce_dims(df, n):
	resort_df_T = df.T
	svd = TruncatedSVD(n_components=n)
	svdmatrix = svd.fit_transform(resort_df_T)
	return svdmatrix

# compute similarities


resort_df = pd.read_csv("ski_resort_data.csv").select_dtypes(['number']).fillna(0)
reduced_matrix = reduce_dims(resort_df, 10)
corr_matrix = np.corrcoef(reduced_matrix)

#reset column headers
snowbird = 13

similar_snowbird = corr_matrix[snowbird]
print(similar_snowbird)

#TODO: pull out resort name list before svd
#TODO: does this need to be a utility matrix, anything diff to accomodate