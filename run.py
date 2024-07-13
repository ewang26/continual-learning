###################################
###################################
### MODIFY ONLY THE FOLLOWING:  ###
###   1. QUEUE                  ###
###   2. run()                  ###
###################################
###################################


import json
import os
import sys
import hashlib
import pickle
import numpy as np
from mnist import run_mnist
from cifar10 import run_cifar10

# If this flag is set to True, the jobs won't be submitted to odyssey;
# they will instead be ran one after another in your current terminal
# session. You can use this to either run a sequence of jobs locally
# on your machine, or to run a sequence of jobs one after another
# in an interactive shell on odyssey.
DRYRUN = False

# This is the base directory where the results will be stored.
# On Odyssey, you may not want this to be your home directory
# If you're storing lots of files (or storing a lot of data).
OUTPUT_DIR = 'output_500'

# This list contains the jobs and hyper-parameters to search over.
# The list consists of tuples, in which the first element is
# the name of the job (here it describes the method we are using)
# and the second is a dictionary of parameters that will be
# be grid-searched over. 
# Note that the second parameter must be a dictionary in which each
# value is a list of options.
# QUEUE = [
#     ('mnist', dict(
#         p=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
#         T=[2, 3, 4, 5],
#         learning_rate=[0.001], # consider [0.01, 0.005, 0.001]
#         batch_size=[10, 30], # consider [10, 30, 50, 65]
#         num_centroids=[2, 4, 6], 
#         model_training_epoch=[10, 20], # consider [10, 20, 50]
#         early_stopping_threshold=[0.1, 1., 5.], # consider [0.1, 0.5, 1., 5., 10.]
#         random_seed=range(20),
#         class_balanced=[True, False],
#         ),
#     ),
#     ('cifar10', dict(
#         p=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
#         T=[2, 3, 4, 5],
#         learning_rate=[0.001], # consider [0.01, 0.005, 0.001]
#         batch_size=[10, 30], # consider [10, 30, 50, 65]
#         num_centroids=[2, 4, 6], 
#         model_training_epoch=[10, 20], # consider [10, 20, 50]
#         early_stopping_threshold=[0.1, 1, 5.], # consider [0.1, 0.5, 1., 5., 10.]
#         random_seed=range(20),
#         class_balanced=[True, False],
#         ),
#     ),
# ]

#for testing purposes
QUEUE = [
    ('mnist', dict(
        p=[0.01, 0.02, 0.05, 0.1, 0.2], 
        T=[5],
        learning_rate=[0.01], # consider [0.01, 0.005, 0.001]
        batch_size=[60], # consider [10, 30, 50, 65]
        num_centroids=[4], 
        model_training_epoch=[10], # consider [10, 20, 50]
        early_stopping_threshold=[0.1, 5.], # consider [0.1, 0.5, 1., 5., 10.]
        random_seed=range(20),
        class_balanced=[True, False],
        ),
    ),
    ('cifar10', dict(
        p=[0.01, 0.02, 0.05, 0.1, 0.2], 
        T=[5],
        learning_rate=[0.001], # consider [0.01, 0.005, 0.001]
        batch_size=[60], # consider [10, 30, 50, 65]
        num_centroids=[4], 
        model_training_epoch=[10], # consider [10, 20, 50]
        early_stopping_threshold=[0.1, 5.], # consider [0.1, 0.5, 1., 5., 10.]
        random_seed=range(20),
        class_balanced=[True, False],
        ),
    ),
]


def run(exp_dir, exp_name, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    2. Get your experiment's parameters from 'exp_kwargs'
    3. Run your experiment
    4. Store the results however you see fit in 'exp_dir'
    '''

    # Add the experiment name to the experiment parameter dictionary
    exp_kwargs['exp_name'] = exp_name

    print('Running experiment {}:'.format(exp_name))
    print('Results are stored in:', exp_dir)
    print('with hyperparameters', exp_kwargs)
    print('\n')

    # Hash the experimental parameters to file name and create file for results
    fname = hyperparameters_to_results_filename(exp_dir, exp_kwargs)

    # If file name exists (i.e. results from a prior run exists), skip experiment
    if os.path.exists(fname):
        print('Experiment previously completed. Skipping!\n')
        return

    # Parse experiment name and run corresponding script
    if exp_name == 'mnist':
        results = run_mnist(exp_kwargs, train_full_only=False)
    elif exp_name == 'cifar10':
        results = run_cifar10(exp_kwargs, train_full_only=False)
    else:
        raise Exception('Unspecified {}'.foramt(exp_name))

    # Add experiment parameter dictionary to results dictionary
    results.update(exp_kwargs)
    
    # Pickle results
    with open(fname, 'wb') as f:
        pickle.dump(results, f)

    print('Done.')


'''
hyperparameters_to_results_filename: hashing function to give file a unique file name based on exp_kwargs(parameters for each experiment) 
input: exp_dir, exp_kwargs
output: a file name 
'''
def hyperparameters_to_results_filename(exp_dir, exp_kwargs):   
    hex_code = hashlib.md5(json.dumps(exp_kwargs).encode('utf-8')).hexdigest()
    return os.path.join(exp_dir, '{}.pickle'.format(hex_code))


def main():
    assert(len(sys.argv) > 2)

    exp_dir = sys.argv[1]
    exp_name = sys.argv[2]
    exp_kwargs = json.loads(sys.argv[3])
    
    run(exp_dir, exp_name, exp_kwargs)


if __name__ == '__main__':
    main()
    
