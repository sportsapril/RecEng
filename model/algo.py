import math
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import WALSModel

def get_rmse(output_row, output_col, actual):
  """Compute rmse between predicted and actual ratings.
  Inputs:
    output_row: evaluated numpy array of row_factor
    output_col: evaluated numpy array of col_factor
    actual: coo_matrix of actual (test) values

  Outupts:
    rmse
  """
  mse = 0
  for i in xrange(actual.data.shape[0]):
    row_pred = output_row[actual.row[i]]
    col_pred = output_col[actual.col[i]]
    err = actual.data[i] - np.dot(row_pred, col_pred)
    mse += err * err
  mse /= actual.data.shape[0]
  rmse = math.sqrt(mse)
  return rmse


def tf_train_wrapper(model, input_tensor, num_iterations):
  """wrapper function to train model on input for num_iterations.

  Inputs:
    model:            WALSModel instance
    input_tensor:     SparseTensor for input ratings matrix
    num_iterations:   number of row/column updates to run

  Outputs:
    tensorflow session, for evaluating results
  """
  sess = tf.Session(graph=input_tensor.graph)

  with input_tensor.graph.as_default():
    row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
    col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

    sess.run(model.initialize_op)
    sess.run(model.worker_init)
    for _ in xrange(num_iterations):
      sess.run(model.row_update_prep_gramian_op)
      sess.run(model.initialize_row_update_op)
      sess.run(row_update_op)
      sess.run(model.col_update_prep_gramian_op)
      sess.run(model.initialize_col_update_op)
      sess.run(col_update_op)

  return sess

LOG_RATINGS = 0
LINEAR_RATINGS = 1
LINEAR_OBS_W = 100.0


def make_wts(data, wt_type, obs_wt, feature_wt_exp, axis):
  """Generate observed item weights.

  Inputs:
    data:             coo_matrix of ratings data
    wt_type:          weight type, LOG_RATINGS or LINEAR_RATINGS
    obs_wt:           linear weight factor
    feature_wt_exp:   logarithmic weight factor
    axis:             axis to make weights for, 1=rows/users, 0=cols/items

  Outputs:
    vector of weights for cols (items) or rows (users)
  """
  # recipricol of sum of number of items across rows (if axis is 0)
  frac = np.array(1.0/(data > 0.0).sum(axis))

  # filter any invalid entries
  frac[np.ma.masked_invalid(frac).mask] = 0.0

  # normalize weights according to assumed distribution of ratings
  if wt_type == LOG_RATINGS:
    wts = np.array(np.power(frac, feature_wt_exp)).flatten()
  else:
    wts = np.array(obs_wt * frac).flatten()

  # check again for any numerically unstable entries
  assert np.isfinite(wts).sum() == wts.shape[0]
  return wts


def run_wals(data, dim, reg, unobs, weights=False,
               wt_type=LINEAR_RATINGS, feature_wt_exp=None,
               obs_wt=LINEAR_OBS_W):
  """Create the WALSModel and input, row and col factor tensors.

  Inputs:
    data:           scipy coo_matrix of item ratings
    dim:            number of latent factors
    reg:            regularization constant
    unobs:          unobserved item weight
    weights:        True: set obs weights, False: obs weights = unobs weights
    wt_type:        feature weight type: linear (0) or log (1)
    feature_wt_exp: feature weight exponent constant
    obs_wt:         feature weight linear factor constant

  Outputs:
    input_tensor:   tensor holding the input ratings matrix
    row_factor:     tensor for row_factor
    col_factor:     tensor for col_factor
    model:          WALSModel instance
  """
  row_wts = None
  col_wts = None

  num_rows = data.shape[0]
  num_cols = data.shape[1]

  if weights:
    assert feature_wt_exp is not None
    row_wts = np.ones(num_rows)
    col_wts = make_wts(data, wt_type, obs_wt, feature_wt_exp, 0)

  row_factor = None
  col_factor = None

  with tf.Graph().as_default():

    input_tensor = tf.SparseTensor(indices=zip(data.row, data.col),
                                   values=(data.data).astype(np.float32),
                                   dense_shape=data.shape)

    model = WALSModel(num_rows, num_cols, dim,
                                        unobserved_weight=unobs,
                                        regularization=reg,
                                        row_weights=row_wts,
                                        col_weights=col_wts)

    # retrieve the row and column factors
    row_factor = model.row_factors[0]
    col_factor = model.col_factors[0]

  return input_tensor, row_factor, col_factor, model

def train_model(args, tr_sparse, params):
  print(params)
  """

  Inputs:
    args: user input
    tr_sparse: sparse training matrix
    params: hyper params for the factorization


  Ouptut:
     the row and column factors (np)
  """
  # params = {
  #   'weights': True,
  #   'latent_factors': 5,
  #   'num_iters': 20,
  #   'regularization': 9.997,
  #   'unobs_weight': 0.001,
  #   'wt_type': 0,
  #   'feature_wt_factor': 200,
  #   'feature_wt_exp': 0.08,
  #   'delimiter': '\t'
  #   }
  # params.update({k: arg for k, arg in arguments.iteritems() if arg is not None})
  # if args.use_optimized:
  #   params.update({'latent_factors': 34,
  #     'regularization': 9.83,
  #     'unobs_weight': 0.001,
  #     'feature_wt_factor': 189.8,
  #     })
  # params.update(task_data)
  # params.update({'output_dir': output_dir})
  # params.update({'job_name': job_name})

  # print(params)

  # dim = args.latent_factors
  # num_iters = args.num_iters
  # reg = args.regularization
  # unobs = args.unobs_weight
  # wt_type = args.wt_type
  # feature_wt_exp = args.feature_wt_exp
  # obs_wt = args.feature_wt_factor

  tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # generate model

  input_tensor, row_factor, col_factor, model = run_wals(tr_sparse, params['latent_factors'],
                                                                params['regularization'],
                                                                params['unobs_weight'],
                                                                params['weights'],
                                                                params['wt_type'],
                                                                params['feature_wt_exp'],
                                                                params['feature_wt_factor'])

  # factorize matrix
  session = tf_train_wrapper(model, input_tensor, params['num_iters'])

  tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

  # evaluate output factor matrices
  output_row = row_factor.eval(session=session)
  output_col = col_factor.eval(session=session)

  # close the training session now that we've evaluated the output
  session.close()

  return output_row, output_col
