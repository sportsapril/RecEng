import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import sys

#specify base path and job_ID to retrieve model output
base_path = '/Users/april.xu/recommendationEngine/wals_ml_engine/'
job_ID = 'wals_ml_local_20190428_172743'

#add to path so that you can load defined packages
sys.path.append(base_path+ '/trainer/')
import model as md


#load model output and movie specs
col_factor = np.load(base_path + 'jobs/' + job_ID + '/model/col.npy') # 9724 x 5
row_factor = np.load(base_path + 'jobs/' + job_ID + '/model/row.npy') # 610 x 5
user_map =   np.load(base_path + 'jobs/' + job_ID + '/model/user.npy')  #100836
item_map =   np.load(base_path + 'jobs/' + job_ID + '/model/item.npy')
item_ID_mapping_dd = pd.read_csv(base_path + 'jobs/' + job_ID + '/model/item_ID_mapping_dd.csv')
movie_spec = pd.read_csv('/Users/april.xu/recommendationEngine/data/links.csv')

#TEST run on client, to incorporate more
client_id = 505
user_idx = client_id - 1


#load ratings.csv file again to get already rated movies
headers = ['user_id', 'item_id', 'rating', 'timestamp']
header_row = 0
input_file = '/Users/april.xu/recommendationEngine/data/ratings.csv'
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


already_rated = list(ratings_df[ratings_df['user_id'] == client_id]['item_id'])
item_ID_mapping_dd = item_ID_mapping_dd.set_index('item_original_ID')
user_rated = list(item_ID_mapping_dd.loc[already_rated]['item_rebased_ID'])


#get recommendations
recommendations_rebased, recommendattions_prob = md.generate_recommendations(user_idx, user_rated, row_factor, col_factor, 5)


#map rebased movie ID to original movie ID
recommendations = item_ID_mapping_dd.reset_index().set_index('item_rebased_ID').loc[recommendations_rebased].reset_index()

#append more dim data
pd.merge(pd.concat([recommendations,pd.DataFrame(recommendattions_prob,columns=['predicted'])],axis=1),movie_spec,
         left_on = 'item_original_ID', right_on = 'movieId', how = 'left')
