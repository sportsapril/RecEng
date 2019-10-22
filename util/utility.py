import datetime
import numpy as np
import os
import pandas as pd
from scipy.sparse import coo_matrix
import sh
import tensorflow as tf



# def create_test_and_train_sets(args, input_file, data_type='ratings'):
#   """Create test and train sets, for different input data types.

#   Args:
#     args: input args for job
#     input_file: path to csv data file
#     data_type:  'ratings': MovieLens style ratings matrix
#                 'web_views': Google Analytics time-on-page data

#   Returns:
#     array of user IDs for each row of the ratings matrix
#     array of item IDs for each column of the rating matrix
#     sparse coo_matrix for training
#     sparse coo_matrix for test

#   Raises:
#     ValueError: if invalid data_type is supplied
#   """
#   if data_type == 'ratings':
#     return _ratings_train_and_test(args['headers'], args['delimiter'],
#                                    input_file)
#   elif data_type == 'web_views':
#     return _page_views_train_and_test(input_file)
#   else:
#     raise ValueError('data_type arg value %s not supported.' % data_type)

# rename _ratings_train_and_test to split_train_and_test
#def _ratings_train_and_test(hasHeader, delimiter, input_file):

def split_train_and_test(args, input_file):
  headers = ['user_id', 'item_id', 'rating', 'timestamp']
  header_row = 0 if args['headers'] else None
  ratings_df = pd.read_csv(input_file,
                         sep=args['delimiter'],
                         names=headers,
                         header=header_row,
                         dtype={
                             'user_id': np.int32,
                             'item_id': np.int32,
                             'rating': np.float32,
                             'timestamp': np.float32,
                         })

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

  #item_ID_mapping_dd.to_csv(+'item_ID_mapping_dd.csv')



  tr_sparse, test_sparse = _create_sparse_train_and_test(ratings,
                                                       n_users, n_items)

  return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse, item_ID_mapping_dd

def train_model(args, tr_sparse):
  """Instantiate WALS model and use "simple_train" to factorize the matrix.

  Args:
    args: training args containing hyperparams
    tr_sparse: sparse training matrix

  Returns:
     the row and column factors in numpy format.
  """
  dim = args['latent_factors']
  num_iters = args['num_iters']
  reg = args['regularization']
  unobs = args['unobs_weight']
  wt_type = args['wt_type']
  feature_wt_exp = args['feature_wt_exp']
  obs_wt = args['feature_wt_factor']

  tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # generate model
  input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse,
                                                                dim,
                                                                reg,
                                                                unobs,
                                                                args['weights'],
                                                                wt_type,
                                                                feature_wt_exp,
                                                                obs_wt)

  # factorize matrix
  session = wals.simple_train(model, input_tensor, num_iters)

  tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # evaluate output factor matrices
  output_row = row_factor.eval(session=session)
  output_col = col_factor.eval(session=session)

  # close the training session now that we've evaluated the output
  session.close()

  return output_row, output_col

def save_model(args, user_map, item_map, row_factor, col_factor,item_ID_mapping_dd):
  """Save the user map, item map, row factor and column factor matrices in numpy format.

  These matrices together constitute the "recommendation model."

  Args:
    args:         input args to training job
    user_map:     user map numpy array
    item_map:     item map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
    item_ID_mapping_dd: original item ID to rebased item ID mapping
  """
  model_dir = os.path.join(args['output_dir'], 'model')

  # if our output directory is a GCS bucket, write model files to /tmp,
  # then copy to GCS
  gs_model_dir = None
  if model_dir.startswith('gs://'):
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args['job_name'])

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

  Args:
    user_idx: the row index of the user in the ratings matrix,

    user_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations)

    row_factor: the row factors of the recommendation model
    col_factor: the column factors of the recommendation model

    k: number of recommendations requested

  Returns:
    list of k item indexes with the predicted highest rating, excluding
    those that the user has already rated
  """

  # bounds checking for args
  #assert (row_factor.shape[0] - len(user_rated)) >= k AX Commented
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