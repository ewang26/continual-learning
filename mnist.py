from data import *
from models import *
from train_task import *

import numpy as np
from numpy.random import RandomState
import pdb

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
torch.set_default_dtype(torch.float64) #change this to float32 if on GPU

def gray_to_rgb(img):
	return img.repeat(3, 1, 1)

def run_mnist(exp_kwargs, train_full_only=True):
	# Define parameters for MNIST dataset
	channels = 3
	feature_dim = 2028
	input_shape = (28, 28)
	input_dim = 16 * 4 * 4

	# Define parameters for task creation
	max_data_size = 6000
	classes_per_task = 2 
	num_tasks = 5

	# Define MNIST experimental parameters
	# model_PATH = './mnist'
	model_PATH = None
	
	# Define parameters for instantiating memory selection methods
	p = 0.01
	num_centroids = 2
	device = 'cpu'
	class_balanced = True
	
	
	# Deinfe CL pipline parameters
	batch_size = 10
	model_training_epoch = 10
	check_point = 1
	lr = 0.001
	early_stopping_threshold = 10000.
	execute_early_stopping = False

	# Seed torch generator
	random_seed = 1

	# Parse experimental parameters
	if 'p' in exp_kwargs.keys():
		p = exp_kwargs['p']
	if 'T' in exp_kwargs.keys():
		num_tasks = exp_kwargs['T']
	if 'random_seed' in exp_kwargs.keys():
		random_seed = exp_kwargs['random_seed']
	if 'learning_rate' in exp_kwargs.keys():
		lr = exp_kwargs['learning_rate']
	if 'batch_size' in exp_kwargs.keys():
		batch_size = exp_kwargs['batch_size']
	if 'num_centroids' in exp_kwargs.keys():
		num_centroids = exp_kwargs['num_centroids']
	if 'model_training_epoch' in exp_kwargs.keys():
		model_training_epoch = exp_kwargs['model_training_epoch']
	if 'early_stopping_threshold' in exp_kwargs.keys():
		early_stopping_threshold = exp_kwargs['early_stopping_threshold']
	if 'execute_early_stopping' in exp_kwargs.keys():
		execute_early_stopping = exp_kwargs['execute_early_stopping']
	if 'model_PATH' in exp_kwargs.keys():
		model_PATH = exp_kwargs['model_PATH']
	if 'class_balanced' in exp_kwargs.keys():
		class_balanced = exp_kwargs['class_balanced']
	if 'max_data_size' in exp_kwargs.keys():
		max_data_size = exp_kwargs['max_data_size']

	# Define parameters for instantiating memory selection methods
	num_exemplars = int(p * max_data_size / classes_per_task)

	# Seed pytorch generator
	generator = torch.Generator().manual_seed(random_seed)

	# Define data transform for MNIST data
	transform = transforms.Compose([transforms.ToTensor(), 
									transforms.Lambda(gray_to_rgb), 
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# Generate MNIST training data
	trainset = torchvision.datasets.MNIST(
		root='./data', 
		train=True,
	    download=True, 
	    transform=transform,
	)

	# Generate MNIST test data
	testset = torchvision.datasets.MNIST(
		root='./data', 
		train=False,
	    download=True, 
	    transform=transform,
	)

	# Create data for tasks
	tasks_data, test_data = make_tasks_data(
		trainset, 
		testset, 
		max_data_size=max_data_size, 
		num_tasks=num_tasks,
		classes_per_task=classes_per_task,
	)

	# Verify correctness of data created for tasks
	for t in tasks_data.keys():
		y = tasks_data[t][1]
		assert len(torch.unique(y)) == classes_per_task
		y = test_data[t][1]
		assert len(torch.unique(y)) == classes_per_task

	# Define training loss
	criterion = nn.CrossEntropyLoss(reduction='none') #no reduction, for per-sample weights

	# Instantiate models
	torch.manual_seed(random_seed)
	model_1 = MNISTNet()
	torch.manual_seed(random_seed)
	model_2 = MNISTNet()
	torch.manual_seed(random_seed)
	model_3 = MNISTNet()
	models = {'M1': model_1, 'M2': model_2, 'M3': model_3}

	# Define parameters for CL training
	kwargs = {
		'model_training_epoch': model_training_epoch, 
		'check_point': check_point, 
		'early_stopping_threshold': early_stopping_threshold, 
		'execute_early_stopping': execute_early_stopping,
		'lr': lr,
		'model_PATH': model_PATH,
		'class_balanced': class_balanced,
	}

	# Initialize results
	results = {}

	# Train model M1 on tasks 0 to T-1, train model M2 on tasks 0 to T; save model weights
	if train_full_only:
		_, _, _, _ = CL_tasks(
			tasks_data, 
			test_data, 
			models, 
			criterion, 
			use_memory_sets=False, 
			random_seed=1, 
			**kwargs,
		)

	# Construct memory sets, train model M3 on memory sets on tasks 0 to T-1 union full training set on task T;
	# Evaluate model performance and gradient similarities
	else:

		# If saved model weights are not available, train models M1 and M2
		if model_PATH is None:
			_, _, models, _ = CL_tasks(
			tasks_data, 
			test_data, 
			models, 
			criterion, 
			use_memory_sets=False, 
			random_seed=1, 
			**kwargs,
			)

		# Initialize memory set managers
		managers = [
		 	RandomMemorySetManager(p), #random memory set
		 	KMeansMemorySetManager(p, num_centroids, device, max_iter=50), #kmeans memory set
		 	LambdaMemorySetManager(p), #lambda memory set
		# 	GSSMemorySetManager(p), #GSS memory set
		 	iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='icarl', architecture='cnn'), #icarl memory set
		 	iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='replay', architecture='cnn'), #icarl memory set,
		 ]

		#managers = [
		#	RandomMemorySetManager(p)]

		# Iterate through all memory managers
		for memory_set_manager in managers:

			# Get the name of the memory set manager
			memory_set_type = memory_set_manager.__class__.__name__
			method_name = f'{memory_set_type} memory selection'

			# Append iCaRL loss function type to memory set manager name
			if memory_set_type == 'iCaRL':
				kwargs['icarl_loss_type'] = memory_set_manager.loss_type
				method_name = f'{memory_set_type} memory selection ({memory_set_manager.loss_type})'

			# Create memory sets and train M3
			performances, grad_similarities, models, _ = CL_tasks(
				tasks_data, 
				test_data, 
				models, 
				criterion, 
				memory_set_manager=memory_set_manager, 
				use_memory_sets=True, 
				random_seed=1, 
				**kwargs,
			)

			# Evaluate M3 on test data
			task_performances = evaluate(
				models['M3'], 
				criterion, 
				test_data, 
				batch_size=batch_size, 
				generator=generator,
			)

			# Append performances and gradient similarities to restuls
			results[method_name] = {
				'model performances': performances, 
				'M3 per task performance': task_performances, 
				'gradient similarities': grad_similarities
			}
		
	return results

def main():

	exp_kwargs = {
		'p': 0.01,
		'T': 5,
		'model_training_epoch': 5,
		'batch_size': 30, 
		'max_data_size': 500,
		'model_PATH': './MNIST', 
	}

	# Set train_full_only to TRUE first, and run to train and save models M1 and M2
	# Then set train_full_only to FALSE, and run to compute memory sets and M3
	train_full_only = False

	results = run_mnist(exp_kwargs, train_full_only=train_full_only)

	if not train_full_only:
		print('MNIST, p={p}, T={T}')
		print(results)

if __name__ == "__main__":
	main()
