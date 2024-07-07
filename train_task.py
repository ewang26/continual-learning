from data import *

import numpy as np
from numpy.random import RandomState
from pathlib import Path
import os
from tqdm import tqdm
import pdb

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
torch.set_default_dtype(torch.float64) #change this to float32 if on GPU

# Split data into tasks, each with an equal number of classes
def make_tasks_data(
	trainset, 
	testset, 
	num_tasks=5, 
	max_data_size=1000, 
	classes_per_task=2
):
	'''
	Creates training and test data for T number of tasks with K number of classes per task.

	Args:
		trainset (pytorch Dataset): training data
		testset (pytorch Dataset): test data
		num_tasks (int): total number of tasks to create
		max_data_size (int): max dataset size for each task
		classes_per_task (int): number of classes per task
	Returns:
		tasks_data (dictionary): training data sets (Tensors), keyed by task
		test_data (dictionar): test data sets (Tensors), keyed by task

	'''

	# Intialize task data dictionaries
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

		# Create input and output data as pytoch tensors
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

# Combine memory sets from previous tasks
def combine_memory_sets(memory_sets, omit_task=0):
	'''
	Combines a dictionary of memory sets, keyed by tasks, into a single pytorch Tensor.
	'''

	memory_x = []
	memory_y = []

	for t in range(len(memory_sets.keys()) - omit_task):
		X = memory_sets[t][0]
		y = memory_sets[t][1]
		task_size = X.shape[0]

		# Concatenate indices to input data as a column for matching data to corresponding weight
		indices = torch.arange(task_size) #indices for data
		y = torch.stack((y, indices), dim=1) #add indices to input as the last column

		memory_x.append(X)
		memory_y.append(y)

	memory_x = torch.cat(memory_x, dim=0)
	memory_y = torch.cat(memory_y, dim=0)

	return memory_x, memory_y

# Compute gradients of model at a batch of data
def get_gradients(X, y, model, criterion, weights=None):
	'''
	Computes the gradients of a model evaluated at a batch of data
	'''

	outputs = model(X)
	# Compute per datapoint weights
	indices = y[:, -1].flatten()
	y = y[:, :-1].flatten()
	if weights is not None:
		assert len(weights) == len(y)
	else: 
		weights = torch.ones(y.shape[0])

	loss = criterion(outputs, y)
	assert loss.shape == y.shape #unreduced loss should have same shape as y's in each batch
	loss = torch.sum(loss * weights) #sum of element-wise multiplication of loss and weights
	loss.backward()

	grad_list = []
	for name, p in model.named_parameters():
	    grad_list.append(p.grad.clone().detach().cpu().numpy().flatten())

	return np.concatenate(grad_list)

# Compute gradient similarity
def comupte_gradient_similarity(model_1, model_2, memory_sets, memory_weights, tasks_data, criterion):
	'''
	Computes the gradient similarity between each model evaluated on the memory sets and on the full training set for tasks 1 through T-1.
	'''

	# Combine memory sets for tasks 1 through T-1
	combined_memory_x, combined_memory_y = combine_memory_sets(memory_sets)
	combined_memory_weights = torch.cat([*memory_weights.values()], dim=0)
	# Evaluate model gradients on combined memory set
	model_1_memory_grad = get_gradients(combined_memory_x, combined_memory_y, model_1, criterion, combined_memory_weights)
	model_2_memory_grad = get_gradients(combined_memory_x, combined_memory_y, model_2, criterion, combined_memory_weights)

	# Combine training data for tasks 1 through T-1
	combined_train_x, combined_train_y = combine_memory_sets(tasks_data, omit_task=1) #note that combine_memory_set doesn't care if the input is memory sets or full training data
	# Evaluate model gradients on combined memory set
	model_1_full_grad = get_gradients(combined_train_x, combined_train_y, model_1, criterion)
	model_2_full_grad = get_gradients(combined_train_x, combined_train_y, model_2, criterion)

	# Compute gradient similarity for model 1 on memory sets vs model 1 on full training sets
	sim_memory_1 = np.dot(model_1_memory_grad, model_1_full_grad) / (np.linalg.norm(model_1_memory_grad) * np.linalg.norm(model_1_full_grad))
	# Compute gradient similarity for model 2 on memory sets vs model 2 on full training sets
	sim_memory_2 = np.dot(model_2_memory_grad, model_2_full_grad) / (np.linalg.norm(model_2_memory_grad) * np.linalg.norm(model_2_full_grad))

	return sim_memory_1, sim_memory_2

# Evaluate model on batched dataset
def batch_eval(evalloader, model, criterion, weights=None):
	'''
	Evaluate model (for classification) trained on task T on a data set (batched) according to a given loss function.
	Uses per data point weights.

	Args:
		evalloader (DataLoader): pytorch DataLoader for evaluation dataset (assuming classification)
		model (Pytorch model): pytorch model extending nn.Module trained on task T
		criterion (Pytorch loss function): pytorch loss function
		weights (Tensor): pytorch Tensor with per-data-point weights for training data

	Returns:
		total_loss (float): total loss of model for the evaluation dataset  
		accuracy (float): accuracy of model over evaluation dataset
	'''

	model.eval()

	# Initialize running loss & accuracy
	total_loss = 0
	total_samples = 0
	total_correct = 0

	with torch.no_grad():
		# Iterate over batches of evaluation dataset
		for batch_x, batch_y in evalloader:
			# Compute per datapoint weights
			indices = batch_y[:, -1].flatten()
			batch_y = batch_y[:, :-1].flatten()
			if weights is not None:
				batch_weights = weights[indices]
			else: 
				batch_weights = torch.ones(batch_y.shape[0])
			# Compute weighted loss and accuracy
			outputs = model(batch_x) #get model output for current batch
			loss = criterion(outputs, batch_y)  #evaluate loss at batch without reduction
			assert loss.shape == batch_y.shape #unreduced loss should have same shape as y's in each batch
			loss = torch.sum(loss * batch_weights) #sum of element-wise multiplication of loss and weights
			total_loss += loss #increment running total loss
			_, predicted = torch.max(outputs, 1) #predict classification label based on model output
			total_correct += (predicted == batch_y).sum().item() #increment running total for correctly predicted labels
			total_samples += batch_y.size(0) #increment running total for number of data points
		accuracy = total_correct / total_samples #compute classification accuracy

	return total_loss, accuracy

# Evaluate the model trained on current task on test sets, keyed by task
def evaluate(model, criterion, test_data, batch_size=10, generator=None):
	'''
	Evaluate model (for classification) trained on task T on test data for tasks t <= T, according to a given loss function. 

	Inputs:
		model (Pytorch model): pytorch model extending nn.Module trained on task T
		criterion (Pytorch loss function): pytorch loss function
		test_data (dictionary): dictionary of test DataSets and classes per task, keyed by task t <= T
		batch_size (int): batch size to instantiate test DataLoader
		generator (pytorch Generator): pytorch generator to instantiate test DataLoader

	Returns:
		performances (dictionary): model performance on tasks t <= T, keyed by T
	'''

	# Initialize performance for 
	performances = {}

	# Iterate through tasks t <= current_task
	for t in test_data.keys():
		# Get data for task t
		X = test_data[t][0]
		y = test_data[t][1]
		task_size = X.shape[0]

		# Concatenate indices to input data as a column for matching data to corresponding weight
		indices = torch.arange(task_size) #indices for data
		y = torch.stack((y, indices), dim=1) #add indices to input as the last column

		# Create test TensorDatasets and DataLoader for current task
		testset = TensorDataset(X, y)
		testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		# Evaluate the model trained on current task on test data from prior tasks
		_, accuracy = batch_eval(testloader, model, criterion)
		performances[t] = accuracy

	return performances

# Compute per-sample gradient 
def compute_grad(model, sample, target):
	'''
	Computes gradient of a given model (wrt to model parameters) at a single data sample.

	Args:
		model (pytorch Model): trained model (classifier)
		sample (pytorch Tensor): single input
		target (pytorch Tensor): single label

	Returns:
		flatten_grad (pytorch Tensor): gradient of the model, flattened 
	'''
	sample = sample.unsqueeze(0)
	target = target.unsqueeze(0)

	criterion = nn.CrossEntropyLoss()
	outputs = model(sample)
	loss = criterion(outputs, target)
	structured_grad = torch.autograd.grad(loss, list(model.parameters()))
	flatten_grad = [layer.flatten() for layer in structured_grad]
	flatten_grad = torch.cat(flatten_grad, 0).reshape((1, -1))
	return flatten_grad

# Get per-sample gradients for a batch of data
def compute_sample_grads(model, batch_data, batch_targets, reshape=None):
	'''
	Computes the per-sample gradient (wrt to model parameters) of a given model for a batch of samples.

	Args:
		model (pytorch Model): trained model (classifier)
		batch_data (pytorch Tensor): a set of inputs
		batch_targets (pytorch Tensor): a set of labels
		reshape (tuple): shape of single input, if reshaping is required

	Returns:
		sample_grads (pytorch Tensor): tensor of per-sample gradients, shape (N, D),
									   where N is the batch size and D is the number of model parameters
	'''
	batch_size = batch_data.shape[0]
	if reshape is not None:
		sample_grads = [compute_grad(model, batch_data[i].reshape(reshape), batch_targets[i]) for i in range(batch_size)]
	else:
		sample_grads = [compute_grad(model, batch_data[i], batch_targets[i]) for i in range(batch_size)]
	sample_grads = torch.cat(sample_grads, 0)
	return sample_grads

# Split data into train, validation and test
def train_validation_split(X, y, val_p=0.1, generator=None):
	# Get tvalidation and train dataset sizes
	data_size = X.shape[0]
	val_size = int(val_p * data_size)
	train_size = data_size - val_size

	# Convert tensors to TensorDataset
	dataset = TensorDataset(X, y)

	# Split dataset into train and validation
	trainset, valset = random_split(dataset, [data_size - val_size, val_size], generator=generator)

	return trainset, valset

# Train the model
def train(
	trainloader, 
	weights, 
	model, 
	criterion, 
	early_stopper, 
	epochs=10, 
	optimizer='Adam', 
	lr=0.001, 
	momentum=0.9, 
	check_point=5,
):
	'''
	Trains model (for classification) trained on task T on a data set (batched) according to a given loss function.
	Uses per data point weights.

	Args:
		trainloader (DataLoader): pytorch DataLoader for train dataset (assuming classification)
		weights (Tensor): pytorch Tensor with per-data-point weights for training data
		model (Pytorch model): pytorch model extending nn.Module
		criterion (Pytorch loss function): pytorch loss function
		early_stopper {dictionary}: contains validation dataset and early stopping threshold 
			'valloader': DataLoader
			'threshold': float
		epochs (float): training epochs
		optimizer (string): optimizer type
		lr (float): learning rate
		momentum (float): momentum for SGD
		check_point (int): print train and validation loss per check_point number of epochs

	Returns:
		model (Pytorch model): trained model  
		callbacks (dictionary): contains evaluation metrics computed during training
			'train_loss': list
			'train_acc': list
			'val_loss': list
			'val_acc': list
	'''

	model.train()
	# Set optimizer
	if optimizer == 'SGD':
		optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
	optimizer = Adam(model.parameters(), lr=lr)

	# Intialize callbacks
	callbacks = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
	prev_val_loss = float("Inf")

	# Iterate through epochs
	for epoch in tqdm(range(epochs)):

		# Initialize losses
		total_correct = 0
		total_samples = 0
		train_loss = 0

		# Iterate through batches
		for batch_x, batch_y in trainloader:
			# Extract indices
			indices = batch_y[:, -1].flatten()
			batch_y = batch_y[:, :-1].flatten()
			# Compute per datapoint weights
			batch_weights = weights[indices]
			# Zero the parameter gradients
			optimizer.zero_grad()
			# Forward pass
			outputs = model(batch_x)
			# Compute batch loss
			loss = criterion(outputs, batch_y) #unreduced loss
			assert loss.shape == batch_y.shape #unreduced loss should have same shape as y's in each batch
			loss = torch.sum(loss * batch_weights) #sum of element-wise multiplication of loss and weights
			# Backwards 
			loss.backward()
			# Take gradient step
			optimizer.step()
			# Total loss
			train_loss += loss.item()
			# Batched accuracy calculations
			with torch.no_grad():
				_, predicted = torch.max(outputs, 1)
				total_correct += (predicted == batch_y).sum().item()
				total_samples += batch_y.size(0)

		# Calculate model accuracy for the last pass through the training data
		train_acc = total_correct / total_samples
		# Calculate model loss and accuracy on validation data
		val_loss, val_acc = batch_eval(early_stopper['valloader'], model, criterion, weights=weights)

		# Save train, validation callbacks
		callbacks['train_loss'].append(train_loss)
		callbacks['val_loss'].append(val_loss)
		callbacks['train_acc'].append(train_acc)
		callbacks['val_acc'].append(val_acc)

		# Print training stats at checkpoint
		if epoch % check_point == 0:    
			print(f'epoch {epoch + 1} train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, train acc: {train_acc:.3f}, val acc: {val_acc:.3f}')
			print(f'diff {abs(val_loss - prev_val_loss)}')
		# Early stopping
		if abs(val_loss - prev_val_loss) <= early_stopper['threshold']:
			break
		else:
			prev_val_loss = val_loss

	return model, callbacks

# Create data loaders for current task in training
def create_dataloaders(
	tasks_data, 
	test_data=None,
	val_p=0.1, 
	batch_size=10, 
	memory_sets=None, 
	memory_weights=None,
	generator=None,
):
	'''
	Create train, validation and test DataLoaders for models M1, M2 and M3

	Args:
		tasks_data (dictionary): training data, tuples (X, y) keyed by task
		test_data (dictionary): test data, tuples (X, y) keyed by task
		val_p (float): percentage of validation data to retain (number between 0 and 1)
		batch_size (int): batch size for DataLoaders
		memory_sets (dictionary): memory sets, tuples (X, y) keyed by task 
		memory_weights (dictionary): memory weights, keyed by task
		generator (pytorch Generator): random number generator 

	Return:
		trainloaders (dictionary): dictionary of training DataLoaders, keyed by model (M1, M2, M3)
		valloaders (dictionary): dictionary of validation DataLoaders, keyed by model (M1, M2, M3)
		testloader (DataLoader): DataLoader for test data combined over tasks, keyed by task
		weights (dictionary): dictionary of per data point weights, keyed by task
	'''

	# Initialization
	val_sets = []
	train_sets = []
	test_sets = []
	weights = []

	data_size = 0
	# Iterate over tasks to create train, validation, test splits for each task
	for t in tasks_data.keys():
		# Get data for task t
		X = tasks_data[t][0]
		y = tasks_data[t][1]
		task_size = X.shape[0]
		data_size += task_size

		# Concatenate indices to input data as a column for matching data to corresponding weight
		indices = torch.arange(task_size) #indices for data
		y = torch.stack((y, indices), dim=1) #add indices to input as the last column
		
		# Get train, validation split
		trainset, valset = train_validation_split(X, y, val_p=val_p, generator=generator)

		# Initialize train, validation weights
		weight_t = torch.ones(task_size)
		weights.append(weight_t)

		# Store train, validation TensorDatasets
		val_sets.append(valset)
		train_sets.append(trainset)

		if test_data is not None:
			# Get data for task t
			X = test_data[t][0]
			y = test_data[t][1]
			task_size = X.shape[0]

			# Concatenate indices to input data as a column for matching data to corresponding weight
			indices = torch.arange(task_size) #indices for data
			y = torch.stack((y, indices), dim=1) #add indices to input as the last column

			# Store test TensorDatasets
			testset = TensorDataset(X, y)
			test_sets.append(testset)

	trainloaders = {'M1': [], 'M2': [], 'M3': []}
	valloaders = {'M1': [], 'M2': [], 'M3': []}
	data_weights = {'M1': [], 'M2': [], 'M3': []}

	if memory_sets is not None:
		# Combine train data for task T and memory set data for tasks 1 through T-1 (for training model M3)
		memory_x, memory_y = combine_memory_sets(memory_sets) #combine memory sets for tasks 1 through T-1
		memory_dataset = TensorDataset(memory_x, memory_y)
		combined_train = ConcatDataset([train_sets[-1], memory_dataset]) #combine memory sets for tasks 1 through T-1 with train data for task T
		trainloaders['M3'] = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		combined_val = ConcatDataset(val_sets) #combine validation sets for tasks 1 through T
		valloaders['M3'] = DataLoader(combined_val, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		memory_weights = torch.cat([*memory_weights.values()], dim=0)
		combined_weights = torch.cat([weights[-1], memory_weights])
		data_weights['M3'] = combined_weights

	else:
		# Combine train and validation data for tasks 1 through T-1 (for training model M1)
		combined_train = ConcatDataset(train_sets[:-1]) #combine training sets for tasks 1 through T-1
		trainloaders['M1'] = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		combined_val = ConcatDataset(val_sets[:-1]) #combine validation sets for tasks 1 through T-1
		valloaders['M1'] = DataLoader(combined_val, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		combined_weights = torch.cat(weights[:-1], dim=0) #combine weights for tasks 1 through T-1
		data_weights['M1'] = combined_weights

		# Combine train and validation data for tasks 1 through T (for training model M2)
		combined_train = ConcatDataset(train_sets) #combine training sets for tasks 1 through T
		trainloaders['M2'] = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		combined_val = ConcatDataset(val_sets) #combine validation sets for tasks 1 through T
		valloaders['M2'] = DataLoader(combined_val, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

		combined_weights = torch.cat(weights, dim=0) #combine weights for tasks 1 through T
		data_weights['M2'] = combined_weights

	if test_data is not None:
		combined_test = ConcatDataset(test_sets) #combine test sets for tasks 1 through T
		testloader = DataLoader(combined_test, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
	else:
		testloader = None

	return trainloaders, valloaders, testloader, data_weights

# Train models M1 on full training data for tasks 1 through T-1, M2 on full training data for tasks 1 through T
def train_full_models(
	tasks_data, 
	models,
	criterion,
	PATH, 
	epochs=10,
	optimizer='Adam',
	lr=0.001,
	momentum=0.9,
	check_point=2,
	early_stopping_threshold=0.1,
	val_p=0.1, 
	batch_size=10, 
	generator=None,
):
	# Create train, test, validation DataLoaders
	trainloaders, valloaders, _, data_weights = create_dataloaders(
		tasks_data, 
		val_p=val_p, 
		batch_size=batch_size, 
		generator=generator,
	)

	# Train models M1 on full training data for tasks 1 through T, M2 on full training data for tasks 1 through T-1
	for m in ['M1', 'M2']:
		print(f'Training model {m}')

		# Define early stopping criterion
		early_stopper = {'valloader': valloaders[m], 'threshold': early_stopping_threshold}

		# Train model 
		model, callbacks = train(
			trainloaders[m], 
			data_weights[m], 
			models[m], 
			criterion, 
			early_stopper,
			epochs=epochs, 
			optimizer=optimizer, 
			lr=lr, 
			momentum=momentum, 
			check_point=check_point,
		)

		# Store model
		models[m] = model

	if PATH is not None:
		# Define model path names
		PATH_1 = f'{PATH}_M1.pth'
		PATH_2 = f'{PATH}_M2.pth'
		# Save model weights for M1 and M2
		torch.save(models['M1'].state_dict(), PATH_1)
		torch.save(models['M2'].state_dict(), PATH_2)

	return models

# Compute memory sets for tasks 1 through T-1 using model M1 (pretrained on tasks 1 through T-1)
def compute_memory_sets(
	tasks_data, 
	classes_per_task, 
	memory_set_manager, 
	model, 
	rand, 
	class_balanced=True, 
):
	'''
	Compute the memory set for each task 1 through T-1.

	Args:
		tasks_data (dictionary): task data, keyed by task
		classes_per_task (int): number of classes per task
		memory_set_manager (object): memory set manager
		model (pytorch model): model for selection methods that rely out outputs and gradients
		rand (numpy random generator): seeded numpy random generator 
		class_balanced (boolean): flag for class balanced memory set selection

	Returns:
		memory_sets (dictionary): memory sets (memory_X, memory_y), keyed by task
		memory_weights (dictionary): weights for memory sets, keyed by task
	'''

	# Initialize memory sets for tasks
	memory_sets = {} #dictionary of memory sets, keyed by task
	memory_weights = {} #dictionary of weights for memory sets, keyed by task

	# Iterate over tasks and create memory sets
	for t in range(len(tasks_data.keys()) - 1):
		X = tasks_data[t][0]
		y = tasks_data[t][1]

		# Create memory sets (for methods that do not rely on current model): random, KMeans, iCaRL

		# If KMeans, need to scale labels to start at 0
		if memory_set_manager.__class__.__name__ == 'KMeansMemorySetManager':
			scaled_y = y - classes_per_task * t #shift class labels for current task to start at 0
			memory_x, memory_y = memory_set_manager.create_memory_set(X, scaled_y)
			memory_y = memory_y + classes_per_task * t #shift memory y labels to those of current task
		else:
			memory_x, memory_y = memory_set_manager.create_memory_set(X, y)

		# Update memory sets (for methods that do rely on current model): GSS, Lambda, GCR

		# If Lambda, update memory set based on hessians (approximated using model outputs)
		if memory_set_manager.__class__.__name__ == 'LambdaMemorySetManager':
			current_classes = classes_per_task * (t + 1)
			model.eval()
			with torch.no_grad():
				output = model(X)[:, current_classes - classes_per_task : current_classes]
			scaled_y = y - classes_per_task * t #shift class labels for current task to start at 0
			memory_x, memory_y = memory_set_manager.update_memory(
				memory_x, 
				memory_y, 
				X, 
				scaled_y, 
				output, 
				classes_per_task,
				class_balanced=class_balanced,
			)
			memory_y = memory_y + classes_per_task * t #shift memory y labels to those of current task

		# If GSS, update memory set based on gradients
		if memory_set_manager.__class__.__name__ == 'GSSMemorySetManager':
			similarity_scores = np.empty(0)

			# replacement_counter = 0 #for debugging

			shuffled_idx = rand.permutation(X.shape[0]) #randomly shuffle training set
			# Iterate through each data point in training set
			for i in tqdm(range(X.shape[0])):

				idx = shuffled_idx[i] #get index of data point

				grad_sample = compute_grad(model, X[idx], y[idx]) #compute gradient of model (wrt weights) at current data point

				# Compute the gradient of a random batch selected from the current memory set
				if memory_x.shape[0] == 0:
					grad_batch = compute_sample_grads(model, X[:2], y[:2]) #if memory set is empty, compute the per-sample gradient for two data points in training data
				else:
					# WP: need to randomly sample batch from memory sets
					grad_batch = compute_sample_grads(model, memory_x, memory_y) #compute the per-sample gradient for a random batch from the current memory set

				# prv_scores = similarity_scores.copy() #for debugging

				# Update the memory set: greedily maximize the gradient diversity in memory set
				memory_x, memory_y, similarity_scores = memory_set_manager.update_GSS_greedy(
					memory_x, memory_y, 
			   		similarity_scores, 
			   		X[idx:idx + 1, :], y[idx].reshape((-1,)), 
			   		grad_sample, grad_batch, class_balanced=class_balanced,
			   	)

			# 	# for debugging
			# 	if len(prv_scores) == len(similarity_scores): 
			# 		diff = np.linalg.norm(similarity_scores - prv_scores)
			# 		if diff > 0:
			# 			replacement_counter += 1
			# print('Number of GSS replacements:', replacement_counter) #for debugging
		
		assert (memory_x.shape[0] == int(int(memory_set_manager.p * X.shape[0]) / classes_per_task) * classes_per_task or (memory_x.shape[0]) == int(memory_set_manager.p * X.shape[0]))

		# If not GCR, set memory set weights to 1/p
		if memory_set_manager.__class__.__name__ != 'GCRMemorySetManager':
			memory_weights[t] = torch.ones(memory_x.shape[0]) * 1. / memory_set_manager.p

		# Save memory sets
		memory_sets[t] = (memory_x, memory_y)

	return memory_sets, memory_weights

# Compare models trained on full datasets with models trained on memory sets 
def CL_tasks(
	tasks_data,
	test_data, 
	models, 
	criterion, 
	memory_set_manager=None, 
	random_seed=1, 
	use_memory_sets=False, 
	return_models=False, 
	**kwargs,
):
	'''
	Given tasks 1 through T, either: 
		(1) train two models M1 (on training data for tasks 1 through T-1) 
		    and M2 (on training data for tasks 1 through T); save weights of both
		or
		(2) load pretrained models M1 and M2. Then use M1 to create memory sets for 
		    tasks 1 through T-1. Then train model M3 on data consisting of memory sets
		    for tasks 1 through T-1 and full training data for task T.

		    We compare the accuracy of M2 and M3 on the unioin of test data from all tasks.
		    We compare the gradients of M1 and M2 on the union of the memory sets for  
			tasks 1 through T-1.
			We compare the gradients of M1 and M2 on the union of the full training sets
			for tasks 1 through T-1.

	Args:
		tasks_data (dictionary): task data, keyed by task
		models (dictionary): pytorch models M1, M2, M3, keyed by model name
		criterion (nn loss function):  pytorch loss function (unreduced)
		memory_set_manager (object): memory set manager
		random_seed (int): random seed
		use_memory_sets (boolean): flag for switching between training M1, M2 and M3 
		return_models (boolean): flag for returning trained models 
		**kwargs: additional experimental parameters

	Returns:
		models (dictionary): trained pytorch models, keyed by model name (if return_models=True)

	'''
	# Parse settings for various memory selection and training methods
	loss_type = None
	epochs = 10
	batch_size = 10
	val_p = 0.1
	model_PATH = None
	grad_PATH = None
	class_balanced = True
	lr = 0.001
	classes_per_task = 2
	check_point = 10
	early_stopping_threshold = 0.01
	optimizer = 'Adam'
	momentum = 0.9

	if 'icarl_loss_type' in kwargs.keys():
		loss_type = kwargs['icarl_loss_type']
	if 'model_training_epoch' in kwargs.keys():
		epochs = kwargs['model_training_epoch']
	if 'batch_size' in kwargs.keys():
		batch_size = kwargs['batch_size']
	if 'val_p' in kwargs.keys():
		val_p = kwargs['val_p']
	if 'model_PATH' in kwargs.keys():
		model_PATH = kwargs['model_PATH']
	if 'grad_PATH' in kwargs.keys():
		grad_PATH = kwargs['grad_PATH']
	if 'class_balanced' in kwargs.keys():
		class_balanced = kwargs['class_balanced']
	if 'lr' in kwargs.keys():
		lr = kwargs['lr']
	if 'classes_per_task' in kwargs.keys():
		classes_per_task = kwargs['classes_per_task']
	if 'check_point' in kwargs.keys():
		check_point = kwargs['check_point']
	if 'early_stopping_threshold' in kwargs.keys():
		early_stopping_threshold = kwargs['early_stopping_threshold']
	if 'optimizer' in kwargs.keys():
		optimizer = kwargs['optimizer']
	if 'momentum' in kwargs.keys():
		momentum = kwargs['momentum']

	# Initialize performances, gradient similariites and memory sets
	performances = {}
	grad_similarities = {}
	memory_sets = {}

	# Create seeded random generator 
	generator = torch.Generator().manual_seed(random_seed)
	rand = RandomState(random_seed)

	# Train models M1, M2 on full training data sets. Save M1, M2 weights
	if not use_memory_sets:
		print('Training models M1 and M2')
		models = train_full_models(
			tasks_data, 
			models,
			criterion,
			model_PATH, 
			epochs=epochs,
			optimizer=optimizer,
			lr=lr,
			momentum=momentum,
			check_point=check_point,
			early_stopping_threshold=early_stopping_threshold,
			val_p=val_p, 
			batch_size=batch_size, 
			generator=generator,
		)

	# Train model M3 on memory sets, compare M3 with M1 and M2
	else:

		# Load models M1 and M2 weights if model PATH provided
		if model_PATH is not None:
			PATH_1 = f'{model_PATH}_M1.pth'
			PATH_2 = f'{model_PATH}_M2.pth'

			models['M1'].load_state_dict(torch.load(PATH_1))
			models['M2'].load_state_dict(torch.load(PATH_2))

		# Compute memory sets, using M1 if gradients are required
		memory_sets, memory_weights = compute_memory_sets(
			tasks_data, 
			classes_per_task, 
			memory_set_manager, 
			models['M1'], 
			rand,
			class_balanced=class_balanced, 
		)

		# Compute gradient similarity for M1 and M2 on full training data dn on memory sets
		M1_sim, M2_sim = comupte_gradient_similarity(models['M1'], models['M2'], memory_sets, memory_weights, tasks_data, criterion)
		grad_similarities['M1'] = M1_sim
		grad_similarities['M2'] = M2_sim

		# Create train, validation DataLoaders for model M3
		trainloaders, valloaders, testloader, data_weights = create_dataloaders(
			tasks_data,
			test_data, 
			val_p=val_p, 
			batch_size=batch_size,
			memory_sets=memory_sets, 
			memory_weights=memory_weights,
			generator=generator,
		)

		# Evaluate model M2 on test data for tasks 1 through T
		_, accuracy = batch_eval(testloader, models['M2'], criterion)
		performances['M2'] = accuracy

		# Train model M3 on memory sets for tasks 1 through T-1 and full training data for task T
		print(f'Training model M3')
		m = 'M3'
		early_stopper = {'valloader': valloaders[m], 'threshold': early_stopping_threshold}
		model, callbacks = train(
			trainloaders[m],
			data_weights[m], 
			models[m], 
			criterion, 
			early_stopper,
			epochs=epochs, 
			optimizer=optimizer, 
			lr=lr, 
			momentum=momentum, 
			check_point=check_point,
			)

		# Store trained model M3 and callbacks
		models[m] = model

		# Evaluate model M3 on test data for tasks 1 through T
		_, accuracy = batch_eval(testloader, models[m], criterion)

		# Store model M3's performance on combined test data for tasks 1 through T
		performances[m] = accuracy

	return performances, grad_similarities, models, memory_sets



