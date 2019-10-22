import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import sys
# import urllib.request

#specify base path and job_ID to retrieve model output
pwd1 = '/Users/aprilxu/Documents/GitHub/RecEng'

print('-------------------------')
print("Path: " + pwd1)
print('-------------------------')

file_path = pwd1 + '/'
base_path = pwd1 + '/'
sys.path.append(base_path +'util/')
sys.path.append(base_path +'model/')

#add to path so that you can load defined packages
import algo as model
import utility as utl

col_factor = np.load(base_path + 'job/' + 'model/col.npy') # 9724 x 5
row_factor = np.load(base_path + 'job/' + 'model/row.npy') # 610 x 5
item_ID_mapping_dd_raw = pd.read_csv(base_path + 'job/' + 'model/item_ID_mapping_dd.csv')
movie_spec = pd.read_csv(base_path + 'data/Dataset_LatestSmall/links.csv')

user_idx_array = range(row_factor.shape[0])

headers = ['user_id', 'item_id', 'rating', 'timestamp']
header_row = 0
input_file = base_path + 'data/Dataset_LatestSmall/ratings.csv'
delimiter = ','
ratings_df = pd.read_csv(input_file,
                       sep=delimiter,
                       names=headers,
                       header=header_row,
                       dtype={
                           'user_id': np.int32,
                           'item_id': np.int32,
                           'rating': np.float32,
                           'timestamp': np.float32,
                       })


item_ID_mapping_dd_orig = item_ID_mapping_dd_raw.set_index('item_original_ID')
item_ID_mapping_dd_rebase = item_ID_mapping_dd_raw.set_index('item_rebased_ID')


existing_topN_rating = pd.DataFrame()
recommended_topN = pd.DataFrame()
for i in user_idx_array:
    client_id = i + 1
    already_rated = ratings_df[ratings_df['user_id'] == client_id][['item_id','rating']]
    topN = already_rated.sort_values('rating',ascending=False)[:10]
    topN = pd.merge(topN,movie_spec,left_on='item_id',right_on='movieId', how='left')
    topN['user_id'] = client_id
    
    existing_topN_rating = pd.concat([existing_topN_rating, topN])
    
    # get recommendations for this user
    user_rated = list(item_ID_mapping_dd_orig.loc[list(topN['item_id'])]['item_rebased_ID'])
    
    recommendations_rebased, recommendattions_prob = utl.generate_recommendations(i, 
                                                                                 user_rated, 
                                                                                 row_factor, 
                                                                                 col_factor, 
                                                                                 50)
    
    recommendations = item_ID_mapping_dd_rebase.loc[recommendations_rebased].reset_index()
    
    rec = pd.merge(pd.concat([recommendations,pd.DataFrame(recommendattions_prob,columns=['predicted'])],axis=1)
                   ,movie_spec,
                   left_on = 'item_original_ID', 
                   right_on = 'movieId', 
                   how = 'left')
    rec['user_id'] = client_id
    
    recommended_topN = pd.concat([recommended_topN, rec])

recommended_topN.to_csv(base_path + 'output/recommended_topN.csv')
