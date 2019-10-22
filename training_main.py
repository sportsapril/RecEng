import argparse
import json
import os
import sys
import tensorflow as tf

# custom packages
pwd1 = '/Users/aprilxu/Documents/GitHub/RecEng'

print('-------------------------')
print("Path: " + pwd1)
print('-------------------------')

file_path = pwd1 + '/'
base_path = pwd1 + '/'
sys.path.append(base_path +'util/')
sys.path.append(base_path +'model/')


# base_path = os.getcwd()
# os.chdir('..')
# up_dir = os.path.abspath(os.curdir)
# sys.path.append(base_path + '/model/')
# sys.path.append(base_path + '/utility/')

import algo as model
import utility as utl


#parse argument 
parser = argparse.ArgumentParser()

parser.add_argument(
  '--extract_file_name',
  help='path to training dataset',
  required=True
)
# parser.add_argument(
#   '--job_dir',
#   help='GCS location',
#   required=True
# )

# hyper params for model
parser.add_argument(
  '--latent_factors',
  type=int,
  help='Number of latent factors',
)
parser.add_argument(
  '--num_iters',
  type=int,
  help='Number of iterations',
)
parser.add_argument(
  '--regularization',
  type=float,
  help='L2 regularization factor',
)
parser.add_argument(
  '--unobs_weight',
  type=float,
  help='Weight for unobserved values',
)
parser.add_argument(
  '--wt_type',
  type=int,
  help='Rating weight type (0=linear, 1=log)',
  default=1
)
parser.add_argument(
  '--feature_wt_factor',
  type=float,
  help='Feature weight factor',
)
parser.add_argument(
  '--feature_wt_exp',
  type=float,
  help='Feature weight exponent',
)

# other args
parser.add_argument(
  '--gcs_bucket',
  help='gcs bucket path',
  required=False
)
parser.add_argument(
  '--output_dir',
  help='GCS location to write output, used if training on local',
)
parser.add_argument(
  '--verbose-logging',
  default=False,
  action='store_true',
  help='TF setting for logging'
)
parser.add_argument(
  '--hyperparam_tune',
  default=False,
  action='store_true',
  help='Tune hyper parameters'
)
parser.add_argument(
  '--delimiter',
  type=str,
  default=',',
  help='Delimiter for csv data files'
)
parser.add_argument(
  '--headers',
  default=True,
  action='store_true',
  help='Input file has a header row'
)
parser.add_argument(
  '--use_optimized',
  default=False,
  action='store_true',
  help='Use GCP trained optimal hyper param'
)

args = parser.parse_args()
arguments = args.__dict__

# add optional gcs path to training data
# if args.gcs_bucket:
# 	args.train_file = os.path.join(args.gcs_bucket, args.extract_file_name)
# else:
# 	args.train_file = base_path + '/data/Dataset_LatestSmall/' + args.extract_file_name
args.train_file = base_path + 'data/Dataset_LatestSmall/' + args.extract_file_name

# set job name as job directory name
job_dir =  base_path[:-1] if base_path.endswith('/') else base_path + '/jobs/'
# job_dir = job_dir[:-1] if job_dir.endswith('/') else job_dir
job_name = os.path.basename(job_dir)

# set output directory for model
if args.hyperparam_tune:
	# if tuning, join the trial number to the output path
	config = json.loads(os.environ.get('TF_CONFIG', '{}'))
	trial = config.get('task', {}).get('trial', '')
	output_dir = os.path.join(job_dir, trial)
elif args.output_dir:
	output_dir = args.output_dir	
else:
	output_dir = job_dir

if args.verbose_logging:
	tf.logging.set_verbosity(tf.logging.INFO)

# Find out if there's a task value on the environment variable.
# If there is none or it is empty define a default one.
env = json.loads(os.environ.get('TF_CONFIG', '{}'))
task_data = env.get('task') or {'type': 'master', 'index': 0}

# update default params with any args provided to task
params = {
    'weights': True,
    'latent_factors': 5,
    'num_iters': 20,
    'regularization': 9.997,
    'unobs_weight': 0.001,
    'wt_type': 0,
    'feature_wt_factor': 200,
    'feature_wt_exp': 0.08,
    'delimiter': '\t'
}
params.update({k: arg for k, arg in arguments.iteritems() if arg is not None})
if args.use_optimized:
	params.update({'latent_factors': 34,
		'regularization': 9.83,
		'unobs_weight': 0.001,
		'feature_wt_factor': 189.8,
		})
params.update(task_data)
params.update({'output_dir': output_dir})
params.update({'job_name': job_name})



def training_main(args):
  # process input file
  # print(args)
  # print(utl)
  print(args.extract_file_name)
  print(base_path)
  print('step 1: validating file name')
  print(args.train_file)
  input_file = utl.validate_file(args.train_file)
  print(input_file)
  print('step 2: splitting training and testing data sets')
  user_map, item_map, tr_sparse, test_sparse, item_ID_mapping_dd = utl.split_train_and_test(args, input_file)

  # train model
  print('step 3: training the model')
  output_row, output_col = model.train_model(args, tr_sparse)

  # save trained model to job directory
  print('step 4: saving the model')
  utl.save_model(args, user_map, item_map, output_row, output_col,item_ID_mapping_dd)

  # log results
  print('step 5: get results')
  train_rmse = model.get_rmse(output_row, output_col, tr_sparse)
  test_rmse = model.get_rmse(output_row, output_col, test_sparse)

  if args.hyperparam_tune:
    # write test_rmse metric for hyperparam tuning
    util.write_hptuning_metric(args, test_rmse)

  tf.logging.info('train RMSE = %.2f' % train_rmse)
  tf.logging.info('test RMSE = %.2f' % test_rmse)



if __name__ == '__main__':
  # job_args = arg_parse()
  print(args)
  training_main(args)