from data import *
from models import *
from tasks_training import *

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
torch.set_default_dtype(torch.float64)

def make_tasks_data(max_data_size=1000):
	# Generate CIFAR training data
	transform = transforms.Compose([transforms.ToTensor(),
								    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                        download=True, transform=transform)
	tasks_data = {}

	# Initialize task
	t = 0
	# Iterate through tasks (2 classes per task)
	for c in range(0, num_tasks * 2, 2):
		print(f'task {t}, classes {c}, {c + 1}')

		# Select two classes
		first_two_classes_idx = np.where((np.array(trainset.targets) == c) | (np.array(trainset.targets) == c + 1))[0][:max_data_size]
		imgs, labels = zip(*trainset)

		X = torch.stack(imgs, 0) #turn images into 3-D tensors
		y = torch.Tensor(labels).long() #turn labels into tensors
		X = X[first_two_classes_idx] #take subset of images for the first two classes
		y = y[first_two_classes_idx] #take subset of labels for the first two classes

		tasks_data[t] = (X, y)

		# Select two classes
		first_two_classes_idx = np.where((np.array(testset.targets) == c) | (np.array(testset.targets) == c + 1))[0][:max_data_size]
		imgs, labels = zip(*testset)

		X = torch.stack(imgs, 0) #turn images into 3-D tensors
		y = torch.Tensor(labels).long() #turn labels into tensors
		X = X[first_two_classes_idx] #take subset of images for the first two classes
		y = y[first_two_classes_idx] #take subset of labels for the first two classes

		test_data[t] = (X, y)
		# Increment task
		t += 1

	return tasks_data, test_data

def main():
	# Define experimental parameters
	batch_size = 10
	model_training_epoch = 30
	classes = 2
	channels = 3
	feature_dim = 2028
	input_shape = (3, 32, 32)
	check_point = 1
	input_dim = 16 * 5 * 5
	p = 0.01
	num_centroids = 2
	device = 'cpu'
	early_stopping_threhold = 0.1
	max_data_size = 1000
	classes_per_task = 2
	num_exemplars = int(p * max_data_size / 2.)
	icarl_loss_type = 'replay'
	num_tasks = 3
	model_PATH = './cifar10'

	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=50)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)
	#icarl memory set
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn')

	method = f'Random \n\n **********************'
	# method = f'iCaRL ({icarl_loss_type} loss) \n\n **********************'
	mset_manager = random_mset

	# Create data for tasks
	tasks_data, test_data = make_tasks_data(max_data_size=max_data_size, num_tasks=num_tasks)

	# Define training loss
	criterion = nn.CrossEntropyLoss()

	# Define model architecture
	model_1 = CifarNet(CIFAR10_ARCH["in_channels"],
        			   CIFAR10_ARCH["out_channels"],
        			   CIFAR10_ARCH["l1_out_channels"],
        			   CIFAR10_ARCH["l2_out_channels"],
        			   CIFAR10_ARCH["l3_out_channels"],
        			   CIFAR10_ARCH["l4_out_channels"])
	model_2 = CifarNet(CIFAR10_ARCH["in_channels"],
        			   CIFAR10_ARCH["out_channels"],
        			   CIFAR10_ARCH["l1_out_channels"],
        			   CIFAR10_ARCH["l2_out_channels"],
        			   CIFAR10_ARCH["l3_out_channels"],
        			   CIFAR10_ARCH["l4_out_channels"])
	model_3 = CifarNet(CIFAR10_ARCH["in_channels"],
        			   CIFAR10_ARCH["out_channels"],
        			   CIFAR10_ARCH["l1_out_channels"],
        			   CIFAR10_ARCH["l2_out_channels"],
        			   CIFAR10_ARCH["l3_out_channels"],
        			   CIFAR10_ARCH["l4_out_channels"])
	models = {'M1': model_1, 'M2': model_2, 'M3': model_3}

	# Define parameters for CL training
	kwargs = {'model_training_epoch': model_training_epoch, 
			  'check_point': check_point, 
			  'early_stopping_threshold': early_stopping_threshold, 
			  'lr': lr,
			  'model_PATH': model_PATH,
			  'class_balanced': True}

	# CL training on tasks
	# CL_tasks(tasks_data, models, criterion, mset_manager, use_memory_sets=False, random_seed=1, **kwargs)
	CL_tasks(tasks_data, test_data, models, criterion, mset_manager, use_memory_sets=True, random_seed=1, **kwargs)


if __name__ == "__main__":
	main()