import datetime
import numpy as np
import os
import pandas as pd
from scipy.sparse import coo_matrix
import sh
import tensorflow as tf


def split_train_and_test(args, input_file):
  headers = ['user_id', 'item_id', 'rating', 'timestamp']
  header_row = 0 if args.headers else None
  print('input file:')
  print(input_file)
  print('input delimiter:')
  print(args.delimiter)
  print(headers)
  print(header_row)
  ratings_df = pd.read_csv(input_file, 
    sep=args.delimiter,
    names=headers,
    header=header_row,
    dtype = {'user_id': np.int32, 
    'item_id': np.int32, 
    'rating': np.float32, 
    'timestamp':np.float32
    }
    )

                         

  np_users = ratings_df.user_id.as_matrix()
  np_items = ratings_df.item_id.as_matrix()
  unique_users = np.unique(np_users)
  unique_items = np.unique(np_items)

  n_users = unique_users.shape[0]
  n_items = unique_items.shape[0]

  # make indexes for users and items if necessary
  max_user = unique_users[-1]
  max_item = unique_items[-1]
  if n_users != max_user or n_items != max_item:
    # make an array of 0-indexed unique user ids corresponding to the dataset
    # stack of user ids
    z = np.zeros(max_user+1, dtype=int)
    z[unique_users] = np.arange(n_users)
    u_r = z[np_users]

    # make an array of 0-indexed unique item ids corresponding to the dataset
    # stack of item ids
    z = np.zeros(max_item+1, dtype=int)
    z[unique_items] = np.arange(n_items)
    i_r = z[np_items]

    # construct the ratings set from the three stacks
    np_ratings = ratings_df.rating.as_matrix()
    ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
    ratings[:, 0] = u_r
    ratings[:, 1] = i_r
    ratings[:, 2] = np_ratings
  else:
    ratings = ratings_df.as_matrix(['user_id', 'item_id', 'rating'])
    # deal with 1-based user indices
    ratings[:, 0] -= 1
    ratings[:, 1] -= 1

  item_ID_mapping = pd.concat([pd.DataFrame(np_items),pd.DataFrame(ratings[:,1])],axis=1)
  item_ID_mapping.columns = ['item_original_ID','item_rebased_ID']

  item_ID_mapping_dd = item_ID_mapping.drop_duplicates()
  item_ID_mapping_dd = item_ID_mapping_dd.set_index('item_original_ID')

  tr_sparse, test_sparse = create_train_and_test_sparse(ratings,n_users, n_items)

  return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse, item_ID_mapping_dd


def save_model(args, user_map, item_map, row_factor, col_factor,item_ID_mapping_dd):
  """

  These matrices together constitute the "recommendation model."

  Inputs:
    args:         input args to training job
    user_map:     user map numpy array
    item_map:     item map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
    item_ID_mapping_dd: original item ID to rebased item ID mapping
  """

  model_dir = os.path.join(args.output_dir, 'model')

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args.job_name)

  os.makedirs(model_dir)
  np.save(os.path.join(model_dir, 'user'), user_map)
  np.save(os.path.join(model_dir, 'item'), item_map)
  np.save(os.path.join(model_dir, 'row'), row_factor)
  np.save(os.path.join(model_dir, 'col'), col_factor)
  item_ID_mapping_dd.to_csv(os.path.join(model_dir, 'item_ID_mapping_dd.csv'))


  if gs_model_dir:
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
  """Generate recommendations for a user.

  Inputs:
    user_idx: the row index of the user in the ratings matrix,

    user_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations)

    row_factor: the row factors of the recommendation model
    col_factor: the column factors of the recommendation model

    k: number of recommendations requested

  Output:
    list of k item indexes with the predicted highest rating, excluding
    those that the user has already rated
  """

  # bounds checking for args

  assert (col_factor.shape[0] - len(user_rated)) >= k

  # retrieve user factor
  user_f = row_factor[user_idx]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(user_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(user_rated)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in user_rated]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items, pred_ratings[recommended_items]

def validate_file(input_file):
  """
  Ensure the training ratings file is stored locally.
  """
  if input_file.startswith('gs:/'):
    input_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(input_path)
    tmp_input_file = os.path.join(input_path, os.path.basename(input_file))
    sh.gsutil("cp", "-r", input_file, tmp_input_file)
    return tmp_input_file
  else:
    return input_file

def create_train_and_test_sparse(ratings, n_users, n_items):
  """Given ratings, create sparse matrices for train and test sets.

  Inputs:
    ratings:  list of ratings tuples  (u, i, r)
    n_users:  number of users
    n_items:  number of items

  Output:
     train, test sparse matrices in scipy coo_matrix format.
  """
  # pick a random test set of entries, sorted ascending
  test_set_size = len(ratings) / 10
  test_set_idx = np.random.choice(xrange(len(ratings)),
                                  size=test_set_size, replace=False)
  test_set_idx = sorted(test_set_idx)

  # sift ratings into train and test sets
  ts_ratings = ratings[test_set_idx]
  tr_ratings = np.delete(ratings, test_set_idx, axis=0)

  # create training and test matrices as coo_matrix's
  u_tr, i_tr, r_tr = zip(*tr_ratings)
  tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

  u_ts, i_ts, r_ts = zip(*ts_ratings)
  test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))

  return tr_sparse, test_sparse