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

def make_tasks_data(trainset, testset, num_tasks=5, max_data_size=1000, classes_per_task=2):
	tasks_data = {}
	test_data = {}

	# Initialize task
	t = 0
	# Iterate through tasks
	for c in range(0, num_tasks * classes_per_task, classes_per_task):
		print(f'task {t}, classes {c}, {c + 1}')

		# Select classes for task t in training set
		first_two_classes_idx = np.where((np.array(trainset.targets) >= c) & (np.array(trainset.targets) < c + classes_per_task))[0][:max_data_size]
		imgs, labels = zip(*trainset)

		X = torch.stack(imgs, 0) #turn images into 3-D tensors
		y = torch.Tensor(labels).long() #turn labels into tensors
		X = X[first_two_classes_idx] #take subset of images for the current task
		y = y[first_two_classes_idx] #take subset of labels for the current task

		tasks_data[t] = (X, y) #append train data for task t to dictionary of task training data

		# Select classes for task t in testing set
		first_two_classes_idx = np.where((np.array(testset.targets) >=  c) & (np.array(testset.targets) < c + classes_per_task))[0][:max_data_size]
		imgs, labels = zip(*testset)

		X = torch.stack(imgs, 0) #turn images into 3-D tensors
		y = torch.Tensor(labels).long() #turn labels into tensors
		X = X[first_two_classes_idx] #take subset of images for the current task
		y = y[first_two_classes_idx] #take subset of labels for the current task

		test_data[t] = (X, y) #append test data for task t to dictionary of task test data

		# Increment task
		t += 1

	return tasks_data, test_data

def run_mnist():
	# Define parameters for MNIST dataset
	channels = 3
	feature_dim = 2028
	input_shape = (28, 28)
	input_dim = 16 * 4 * 4

	# Define parameters for task creation
	max_data_size = 1000
	classes_per_task = 2 
	num_tasks = 5

	# Define MNIST experimental parameters
	model_PATH = './mnist'
	train_full = False
	
	# Define parameters for instantiating memory selection methods
	p = 0.01
	num_centroids = 2
	device = 'cpu'
	num_exemplars = int(p * max_data_size / 2.)
	
	# Deinfe CL pipline parameters
	batch_size = 10
	model_training_epoch = 10
	check_point = 1
	lr = 0.001
	early_stopping_threshold = 5.

	# Seed torch generator
	random_seed = 1
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
	model_1 = MNISTNet(in_channels=3, out_channels=10)
	torch.manual_seed(random_seed)
	model_2 = MNISTNet(in_channels=3, out_channels=10)
	torch.manual_seed(random_seed)
	model_3 = MNISTNet(in_channels=3, out_channels=10)
	models = {'M1': model_1, 'M2': model_2, 'M3': model_3}

	# Define parameters for CL training
	kwargs = {
		'model_training_epoch': model_training_epoch, 
		'check_point': check_point, 
		'early_stopping_threshold': early_stopping_threshold, 
		'lr': lr,
		'model_PATH': model_PATH,
		'class_balanced': True,
	}

	# Train model M1 on tasks 0 to T-1, train model M2 on tasks 0 to T; save model weights
	if train_full:
		_, _ = CL_tasks(
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
		# Initialize memory set managers
		managers = [
			RandomMemorySetManager(p), #random memory set
			KMeansMemorySetManager(p, num_centroids, device, max_iter=50), #kmeans memory set
			LambdaMemorySetManager(p), #lambda memory set
			GSSMemorySetManager(p), #GSS memory set
			iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='icarl', architecture='cnn'), #icarl memory set
			iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='replay', architecture='cnn'), #icarl memory set,
		]

		# Open output file for results
		f = open('output_MNIST.txt', 'a')

		# Iterate through all memory managers
		for memory_set_manager in managers:
			memory_set_type = memory_set_manager.__class__.__name__
			method_name = f'{memory_set_type} memory selection'

			if memory_set_type == 'iCaRL':
				kwargs['icarl_loss_type'] = memory_set_manager.loss_type
				method_name = f'{memory_set_type} memory selection ({memory_set_manager.loss_type})'

			performances, models = CL_tasks(
				tasks_data, 
				test_data, 
				models, 
				criterion, 
				memory_set_manager, 
				use_memory_sets=True, 
				random_seed=1, 
				**kwargs,
			)

			task_performances = evaluate(
				models['M3'], 
				criterion, 
				test_data, 
				batch_size=batch_size, 
				generator=generator,
			)

			print(f'{method_name}, p: {p}, tasks: {num_tasks}, classes per task: {classes_per_task} \n **********************', file=f)
			print(performances, file=f)
			print(task_performances, file=f)
			print('\n\n', file=f)
		
		f.close()
	return None

def main():
	run_mnist()
	
if __name__ == "__main__":
	main()