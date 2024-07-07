from data import *
from models import *

from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import pdb

import torch
torch.set_default_dtype(torch.float64)
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


#######
####### Util for training, evaluating, computing gradients for NN classifiers
#######

# For visualizing image data
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# CNN for MNIST & CIFAR
class CNNnet(nn.Module):
    def __init__(self, input_dim, output_dim, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.double()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Simple feedforward for toy
class FCnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train NN classifier
def train_classifier(input_dim, output_dim, channels, trainloader, epochs=10, architecture='fc', check_point=10, PATH=None):
	if architecture == 'fc':
		classifier = FCnet(input_dim, output_dim)
	if architecture == 'cnn':
		classifier = CNNnet(input_dim, output_dim, channels)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(epochs):  

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			X, y = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = classifier(X)

			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()

			# print loss
			running_loss += loss.item()
		if epoch % check_point == 0:    
		    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')

	if PATH is not None:
		torch.save(classifier.state_dict(), PATH)

	return classifier

# Evaluate trained classifier
def evaluate_classifier(loader, classifier, print_output=True):
	correct = 0
	total = 0

	classifier.eval()
	with torch.no_grad():
	    for data in loader:
	        X, y = data
	        outputs = classifier(X)
	        _, predicted = torch.max(outputs.data, 1)
	        total += y.size(0)
	        correct += (predicted == y).sum().item()

	accuracy = 100 * correct // total
	if print_output:
		print(f'Accuracy of the network: {accuracy} %')
	return accuracy

# Compute gradient per sample
def compute_grad(classifier, sample, target):
	sample = sample.unsqueeze(0)
	target = target.unsqueeze(0)

	criterion = nn.CrossEntropyLoss()
	outputs = classifier(sample)
	loss = criterion(outputs, target)
	structured_grad = torch.autograd.grad(loss, list(classifier.parameters()))
	flatten_grad = [layer.flatten() for layer in structured_grad]
	flatten_grad = torch.cat(flatten_grad, 0).reshape((1, -1))
	return flatten_grad

# Get per sample gradients for a batch
def compute_sample_grads(classifier, data, targets, reshape=None):
    """ manually process each sample with per sample gradient """
    batch_size = data.shape[0]
    if reshape is not None:
    	sample_grads = [compute_grad(classifier, data[i].reshape(reshape), targets[i]) for i in range(batch_size)]
    else:
    	sample_grads = [compute_grad(classifier, data[i], targets[i]) for i in range(batch_size)]
    sample_grads = torch.cat(sample_grads, 0)
    return sample_grads

#######
####### Utils for generating toy data and visualizing memory sets on toy
#######

# Generate mixture of gaussians toy data
def mixture_of_gaussians(rand, n_samples=400, n_classes=2, n_clusters_per_class=2, cluster_sep=2., cluster_var=1.):
	
	X = []
	y = []
	samples_per_cluster = int(n_samples / (n_classes * n_clusters_per_class))
	cov = cluster_var * np.identity(2)

	for i in range(n_classes):
		for _ in range(n_clusters_per_class): 
			random_vector = rand.normal(0, cluster_sep, 2)
			X += rand.multivariate_normal(random_vector, cov, samples_per_cluster).tolist()
			y += [i] * samples_per_cluster

	return np.array(X), np.array(y)

# Plot decision boundary for classifier on toy
def plot_decision_boundary(X, model, ax, cm, steps=1000):
    # define region of interest by data limits
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # make predictions across region of interest
    model.eval()
    outputs = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])))
    _, labels_predicted = torch.max(outputs, 1)
    # plot decision boundary in region of interest
    z = labels_predicted.detach().numpy().reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=cm, alpha=0.2)
    return ax

# Plot memory set against data for toy
def plot_memory_set(mset_manager, X, y, cm, title, ax, classifier, rand, method=None):
	X = torch.from_numpy(X)
	y = torch.from_numpy(y)
	
	memory_x, memory_y = mset_manager.create_memory_set(X, y)

	if method == 'lambda':
		output = classifier(X)
		memory_x, memory_y = mset_manager.update_memory(memory_x, memory_y, X, y, output, class_balanced=True)

	if method == 'GSS':
		similarity_scores = np.empty(0)
		replacement_counter = 0 

		shuffled_idx = rand.permutation(X.shape[0])
		for i in range(X.shape[0]):
			idx = shuffled_idx[i]
			grad_sample = compute_grad(classifier, X[idx], y[idx])
			if memory_x.shape[0] == 0:
				grad_batch = compute_sample_grads(classifier, X[:2], y[:2])
			else:
				grad_batch = compute_sample_grads(classifier, memory_x, memory_y)

			prv_scores = similarity_scores.copy()

			memory_x, memory_y, similarity_scores = mset_manager.update_GSS_greedy(memory_x, memory_y, 
																				   similarity_scores, 
																				   X[idx].reshape((1, -1)), y[idx].reshape((-1,)), 
																				   grad_sample, grad_batch, class_balanced=True)
			
			if i > mset_manager.memory_set_size:
				diff = np.linalg.norm(similarity_scores - prv_scores)
				if diff > 0:
					replacement_counter += 1

		print('Number of GSS replacements:', replacement_counter)
	
	memory_x = memory_x.detach().numpy()
	memory_y = memory_y.detach().numpy()

	ax.scatter(X[:, 0], X[:, 1], s=40, marker='o', c=y, cmap=cm, edgecolors='none', alpha=0.1, label='Data')
	ax.scatter(memory_x[:, 0], memory_x[:, 1], s=10, marker='^', c=memory_y, cmap=cm, alpha=0.7, label='Memory Set')
	if method == 'kmeans':
		ax.scatter(mset_manager.centroids[0][:, 0], mset_manager.centroids[0][:, 1], marker='1', color='purple', label='class 0 centroids')
		ax.scatter(mset_manager.centroids[1][:, 0], mset_manager.centroids[1][:, 1], marker='1', color='blue', label='class 1 centroids')
	
	legend = ax.legend(loc='best')

	for lh in legend.legend_handles: 
		lh.set_alpha(1)

	ax.set_title(title)

	return ax

#######
####### Utils for comparing model performance on full vs memory set
#######

# (Non-Incremental) Compare accuracy of models trained on full dataset vs memory set on one task 
def model_performance_compare(mset_manager, full_classifier, X, y, testloader, rand, input_shape, classes, channels, fPATH, input_dim, epochs=20, batch_size=10, method=None):
	full_accuracy = evaluate_classifier(testloader, full_classifier, print_output=False)
	memory_x, memory_y = mset_manager.create_memory_set(X, y)

	if method == 'lambda':
		trainset = torch.stack([x.reshape((channels, input_shape[0], input_shape[1])) for x in X])
		output = full_classifier(trainset)
		memory_x, memory_y = mset_manager.update_memory(memory_x, memory_y, X, y, output, class_balanced=True)


	if method == 'GSS':
		similarity_scores = np.empty(0)
		replacement_counter = 0 

		shuffled_idx = rand.permutation(X.shape[0])
		for i in range(X.shape[0]):
			idx = shuffled_idx[i]
			grad_sample = compute_grad(full_classifier, X[idx].reshape((channels, input_shape[0], input_shape[1])), y[idx])
			if memory_x.shape[0] == 0:
				grad_batch = compute_sample_grads(full_classifier, X[:2], y[:2], reshape=(channels, input_shape[0], input_shape[1]))
			else:
				grad_batch = compute_sample_grads(full_classifier, memory_x, memory_y, reshape=(channels, input_shape[0], input_shape[1]))

			prv_scores = similarity_scores.copy()

			memory_x, memory_y, similarity_scores = mset_manager.update_GSS_greedy(memory_x, memory_y, 
																				   similarity_scores, 
																				   X[idx].reshape((1, -1)), y[idx].reshape((-1,)), 
																				   grad_sample, grad_batch, class_balanced=True)
			if i % 100 == 0:
				print('GSS pass:', i)

			if i > mset_manager.memory_set_size:
				if len(similarity_scores) != len(prv_scores):
					replacement_counter += 1
				else:
					diff = np.linalg.norm(similarity_scores - prv_scores)
					if diff > 0:
						replacement_counter += 1

		print('Number of GSS replacements:', replacement_counter)

	memory_trainset = torch.stack([x.reshape((channels, input_shape[0], input_shape[1])) for x in memory_x])
	trainset = torch.utils.data.TensorDataset(memory_trainset, memory_y)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	memory_classifier = train_classifier(input_dim, classes, channels, trainloader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)

	with open(fPATH, 'a') as f:
		print(f'\n\nBenchmarking {method}', file=f)
		print(f'************************', file=f)
		print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)
		print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)
	f.close()

	return None

#######
####### Memory set selection comparison on toy data 
#######

# (Random, KMeans, Lambda, GSS)(non-Incremental) Memory set on toy data: 2D classification, 2 classes, multiple clusters per class
def toy_data_benchmarking():
	classes = 2
	input_dim = 2
	batch_size = 30
	epochs = 20
	rand = RandomState(1)
	p = 0.2
	num_centroids = 2
	device = 'cpu'

	# X, y = make_classification(
	# 			n_samples=400, n_features=2, n_redundant=0, n_informative=2, 
	# 			n_classes=classes, n_clusters_per_class=2, class_sep=2.,
	# 			random_state=10
	# 		)

	X, y = mixture_of_gaussians(rand, n_samples=400, n_classes=classes, 
								n_clusters_per_class=2, cluster_sep=6., 
								cluster_var=1.5)

	print('memory set size:', int(p * X.shape[0]))

	# Create data loader from data
	traindata = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
	trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)

	# Train a nn classifier
	classifier = train_classifier(2, classes, 1, trainloader, epochs=epochs)
	evaluate_classifier(trainloader, classifier)

	# Compute memory sets
	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=10)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)
	#iCARL


	# Plot memory sets against data
	fig, ax = plt.subplots(1, 4, figsize=(20, 5))
	cm = plt.cm.Spectral
	#plot random mset
	ax[0] = plot_memory_set(random_mset, X, y, cm, 'Memory Set Selection: Random', ax[0], classifier, rand)
	#plot kmeans mset
	ax[1] = plot_memory_set(kmeans_mset, X, y, cm, 'Memory Set Selection: KMeans', ax[1], classifier, rand, method='kmeans')
	#plot lambda mset
	ax[2] = plot_decision_boundary(X, classifier, ax[2], cm, steps=1000)
	ax[2] = plot_memory_set(lambda_mset, X, y, cm, 'Memory Set Selection: Lambda', ax[2], classifier, rand, method='lambda')
	#plot GSS mset
	ax[3] = plot_decision_boundary(X, classifier, ax[3], cm, steps=1000)
	ax[3] = plot_memory_set(GSS_mset, X, y, cm, 'Memory Set Selection: GSS', ax[3], classifier, rand, method='GSS')
	plt.show()

	return None

# (iCaRL)(Incremental) Memory set on toy data: 2D classification, 2 tasks, 2 classes per task, multiple clusters per class
def iCaRL_toy_data_benchmarking():
	classes = 2
	n_samples = 400
	input_dim = 2
	feature_dim = 120
	batch_size = 30
	epochs = 20
	rand = RandomState(10)
	p = 0.2
	num_centroids = 2
	device = 'cpu'
	num_exemplars = int(p * n_samples / 2.)

	# Generate data for Task 1
	X_t1, y_t1 = mixture_of_gaussians(RandomState(10), n_samples=n_samples, n_classes=classes, 
								n_clusters_per_class=2, cluster_sep=2., 
								cluster_var=1.)
	# Generate data for Task 2
	X_t2, y_t2 = mixture_of_gaussians(RandomState(24), n_samples=n_samples, n_classes=classes, 
								n_clusters_per_class=2, cluster_sep=3., 
								cluster_var=1.)
	# Adjust the cluster centers of data in Task 2
	X_t2 += np.array([[-13, 6]])
	y_t2 += 2
	# Define the tasks
	tasks = {0: (X_t1, y_t1), 1: (X_t2, y_t2)}

	# Plot memory sets against data
	fig, ax = plt.subplots(1, 3, figsize=(15, 5))
	cm = {0: plt.cm.PRGn, 1: plt.cm.bwr}
	memory_cm = {0: ('purple', 'green'), 1: ('blue', 'red')}

	# Scatter plot the data for the tasks
	for t in tasks.keys():
		X = tasks[t][0]
		y = tasks[t][1]
		ax[0].scatter(X[:, 0], X[:, 1], s=40, marker='o', c=y, cmap=cm[t], edgecolors='none', alpha=0.1, label='Data')
		ax[1].scatter(X[:, 0], X[:, 1], s=40, marker='o', c=y, cmap=cm[t], edgecolors='none', alpha=0.1, label='Data')
		ax[2].scatter(X[:, 0], X[:, 1], s=40, marker='o', c=y, cmap=cm[t], edgecolors='none', alpha=0.1, label='Data')

	# Instantiate iCaRL and compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='fc')
	for t in tasks.keys():
		X = tasks[t][0]
		y = tasks[t][1]
		classes = np.unique(y)
		X_0 = X[y == classes[0]]
		y_0 = y[y == classes[0]]
		X_1 = X[y == classes[1]]
		y_1 = y[y == classes[1]]

		# Construct exemplar from random features
		mset_0_rand = icarl.construct_exemplar_set(torch.from_numpy(X_0), torch.from_numpy(y_0))
		mset_1_rand = icarl.construct_exemplar_set(torch.from_numpy(X_1), torch.from_numpy(y_1))
		# Scatter plot memory set of random features
		ax[0].scatter(X_0[mset_0_rand][:, 0], X_0[mset_0_rand][:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label='Memory Set (Task 1)')
		ax[0].scatter(X_1[mset_1_rand][:, 0], X_1[mset_1_rand][:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label='Memory Set (Task 1)')

	# Instantiate iCaRL and compute memory sets using features trained by icarl loss function
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='fc')
	for t in tasks.keys():
		X = tasks[t][0]
		y = tasks[t][1]
		classes = np.unique(y)
		X_0 = X[y == classes[0]]
		y_0 = y[y == classes[0]]
		X_1 = X[y == classes[1]]
		y_1 = y[y == classes[1]]

		memory_x, memory_y = icarl.create_memory_set(torch.from_numpy(X), torch.from_numpy(y), loss_type='icarl')
		mset_0_icarl = memory_x[memory_y == classes[0]]
		mset_1_icarl = memory_x[memory_y == classes[1]]
				
		ax[1].scatter(mset_0_icarl[:, 0], mset_0_icarl[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
		ax[1].scatter(mset_1_icarl[:, 0], mset_1_icarl[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')

	# Instantiate iCaRL and compute memory sets using features trained by replay loss function
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='fc')
	for t in tasks.keys():
		X = tasks[t][0]
		y = tasks[t][1]
		classes = np.unique(y)
		X_0 = X[y == 0]
		y_0 = y[y == 0]
		X_1 = X[y == 1]
		y_1 = y[y == 1]

		memory_x, memory_y = icarl.create_memory_set(torch.from_numpy(X), torch.from_numpy(y), loss_type='replay')
		mset_0_replay = memory_x[memory_y == classes[0]]
		mset_1_replay = memory_x[memory_y == classes[1]]

		ax[2].scatter(mset_0_replay[:, 0], mset_0_replay[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
		ax[2].scatter(mset_1_replay[:, 0], mset_1_replay[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')

		for i in range(3):
			if i == 2:
				legend = ax[i].legend(loc='best', bbox_to_anchor=(1.1, 1.05))
			# else:
			# 	legend = ax[i].legend(loc='best')
			
				for lh in legend.legend_handles: 
					lh.set_alpha(1)

	ax[0].set_title('Memory Set Selection: iCARL \n(random features)')
	ax[1].set_title('Memory Set Selection: iCARL \n(features trained with iCaRL loss)')
	ax[2].set_title('Memory Set Selection: iCARL \n(features trained with replay loss)')
	plt.tight_layout()
	plt.show()

	return None

#######
####### Memory set selection comparison on image data
#######

# (Random, KMeans, Lambda, GSS)(non-Incremental) Memory set on CIFAR10 data
def CIFAR_benchmarking():

	PATH = './cifar_net.pth'
	batch_size = 10
	epochs = 100
	classes = 2
	channels = 3
	input_shape = (32, 32)
	input_dim = 16 * 5 * 5
	rand = RandomState(1)
	p = 0.05
	max_data_size = 1000
	num_centroids = 2
	device = 'cpu'
	class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# Generate CIFAR training data
	transform = transforms.Compose([transforms.ToTensor(),
								    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                        download=True, transform=transform)
	#select two classes: planes and car
	first_two_classes_idx = np.where((np.array(trainset.targets) == 0) | (np.array(trainset.targets) == 1))[0][:max_data_size]
	imgs, labels = zip(*trainset)
	X = torch.cat([img.flatten().reshape((1, -1)) for img in imgs], 0)
	y = torch.Tensor(labels).long()
	X = X[first_two_classes_idx]
	y = y[first_two_classes_idx]


	trainsubset = torch.utils.data.Subset(trainset, first_two_classes_idx)
	trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	first_two_classes_idx = np.where((np.array(testset.targets) == 0) | (np.array(testset.targets) == 1))[0]
	testsubset = torch.utils.data.Subset(testset, first_two_classes_idx)
	testloader = torch.utils.data.DataLoader(testsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)
	
	# # #data validation: visualize some training data
	# # # dataiter = iter(trainloader)
	# # # images, labels = next(dataiter)
	# # # imshow(torchvision.utils.make_grid(images))
	# # # print('images: ', ' '.join(f'{class_names[labels[j]]:5s}' for j in range(4)))

	# Train and save a cnn classifier
	classifier = train_classifier(input_dim, classes, channels, trainloader, epochs=epochs, architecture='cnn', check_point=10, PATH=PATH)
	evaluate_classifier(trainloader, classifier)

	# # Load cnn classifier from path
	# classifier = CNNnet(input_dim, classes, channels)
	# classifier.load_state_dict(torch.load(PATH))

	# Compute memory sets
	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=10)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)

	# Benchmark model performance on memory sets
	with open('output_CIFAR.txt', 'a') as f:
		print('CIFAR Benchmarking', file=f)
		print('Memory set size:', int(p * X.shape[0]), file=f)
	f.close()
	#random 
	model_performance_compare(random_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, 'output_CIFAR.txt', input_dim, epochs=epochs, batch_size=10, method='random')
	
	#kmeans
	model_performance_compare(kmeans_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, 'output_CIFAR.txt', input_dim, epochs=epochs, batch_size=10, method='kmeans')

	#lambda
	model_performance_compare(lambda_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, 'output_CIFAR.txt', input_dim, epochs=epochs, batch_size=10, method='lambda')

	#GSS
	model_performance_compare(GSS_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, 'output_CIFAR.txt', input_dim, epochs=epochs, batch_size=10, method='GSS')

	return None

def gray_to_rgb(img):
	return img.repeat(3, 1, 1)

def MNIST_benchmarking():
	PATH = './mnist_net.pth'
	batch_size = 10
	epochs = 20
	classes = 2
	channels = 3
	input_shape = (28, 28)
	input_dim = 16 * 4 * 4
	rand = RandomState(1)
	p = 0.01
	max_data_size = 1000
	num_centroids = 2
	device = 'cpu'
	class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# Generate MNIST training data
	transform = transforms.Compose([transforms.ToTensor(), 
									transforms.Lambda(gray_to_rgb), 
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.MNIST(root='./data', train=True,
	                                        download=True, transform=transform)
	testset = torchvision.datasets.MNIST(root='./data', train=False,
	                                        download=True, transform=transform)

	#select two classes
	first_two_classes_idx = np.where((np.array(trainset.targets) == 0) | (np.array(trainset.targets) == 1))[0][:max_data_size]
	imgs, labels = zip(*trainset)

	X = torch.cat([img.flatten().reshape((1, -1)) for img in imgs], 0)
	y = torch.Tensor(labels).long()
	X = X[first_two_classes_idx]
	y = y[first_two_classes_idx]

	trainsubset = torch.utils.data.Subset(trainset, first_two_classes_idx)
	trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	first_two_classes_idx = np.where((np.array(testset.targets) == 0) | (np.array(testset.targets) == 1))[0]
	testsubset = torch.utils.data.Subset(testset, first_two_classes_idx)
	testloader = torch.utils.data.DataLoader(testsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)
	
	# # #data validation: visualize some training data
	# # # dataiter = iter(trainloader)
	# # # images, labels = next(dataiter)
	# # # imshow(torchvision.utils.make_grid(images))
	# # # print('images: ', ' '.join(f'{class_names[labels[j]]:5s}' for j in range(4)))

	# # Train and save a cnn classifier
	# classifier = train_classifier(input_dim, classes, channels, trainloader, epochs=epochs, architecture='cnn', check_point=10, PATH=PATH)
	# evaluate_classifier(trainloader, classifier)

	# Load cnn classifier from path
	classifier = CNNnet(input_dim, classes, channels)
	classifier.load_state_dict(torch.load(PATH))

	# Compute memory sets
	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=10)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)

	# Benchmark model performance on memory sets
	f_PATH = 'output_MNIST.txt'
	with open(f_PATH, 'a') as f:
		print('MNIST Benchmarking', file=f)
		print('Memory set size:', int(p * X.shape[0]), file=f)
	f.close()
	# #random 
	model_performance_compare(random_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, f_PATH, input_dim, epochs=epochs, batch_size=10, method='random')
	
	# #kmeans
	model_performance_compare(kmeans_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, f_PATH, input_dim, epochs=epochs, batch_size=10, method='kmeans')

	#lambda
	model_performance_compare(lambda_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, f_PATH, input_dim, epochs=epochs, batch_size=10, method='lambda')

	#GSS
	model_performance_compare(GSS_mset, classifier, X, y, testloader, rand, input_shape, classes, channels, f_PATH, input_dim, epochs=epochs, batch_size=10, method='GSS')

	return None

def subset_data(X, y, classes):
	X_0 = X[y == classes[0]]
	y_0 = y[y == classes[0]]
	X_1 = X[y == classes[1]]
	y_1 = y[y == classes[1]]
	return X_0, y_0, X_1, y_1

# (iCaRL)(non-Incremental) Memory set on CIFAR10 data
def iCaRL_CIFAR_benchmarking():
	PATH = './cifar_net.pth'
	batch_size = 10
	epochs = 100
	classes = 2
	channels = 3
	feature_dim = 2028
	input_shape = (3, 32, 32)
	input_dim = 16 * 5 * 5
	p = 0.05
	max_data_size = 1000

	# Generate CIFAR training data
	transform = transforms.Compose([transforms.ToTensor(),
								    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                        download=True, transform=transform)
	# Select two classes: planes and car
	first_two_classes_idx = np.where((np.array(trainset.targets) == 0) | (np.array(trainset.targets) == 1))[0][:max_data_size]
	imgs, labels = zip(*trainset)

	X = torch.stack(imgs, 0) #turn images into 3-D tensors
	y = torch.Tensor(labels).long() #turn labels into tensors
	X = X[first_two_classes_idx] #take subset of images for the first two classes
	y = y[first_two_classes_idx] #take subset of labels for the first two classes

	X_0, y_0, X_1, y_1 = subset_data(X, y, (0, 1))

	num_exemplars = int(p * max_data_size / 2.) #number of images exemplar set

	# Make training and test data loaders
	trainsubset = torch.utils.data.Subset(trainset, first_two_classes_idx)
	trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	first_two_classes_idx = np.where((np.array(testset.targets) == 0) | (np.array(testset.targets) == 1))[0]
	testsubset = torch.utils.data.Subset(testset, first_two_classes_idx)
	testloader = torch.utils.data.DataLoader(testsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	# Load cnn classifier (trained on full dataset) from path
	classifier = CNNnet(input_dim, classes, channels)
	classifier.load_state_dict(torch.load(PATH))
	# Get test accuracy of classifier (trained on full dataset)
	full_accuracy = evaluate_classifier(testloader, classifier, print_output=False)

	# Benchmark model performance on memory sets
	f = open('output_CIFAR.txt', 'a')
	print('CIFAR Benchmarking', file=f)
	print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (Random Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	mset_0_idx = icarl.construct_exemplar_set(X_0, y_0) #get memory set indices for class 0
	mset_1_idx = icarl.construct_exemplar_set(X_1, y_1) #get memory set indices for class 1
	mset_0_rand = X_0[mset_0_idx] #get memory set for class 0
	mset_1_rand = X_1[mset_1_idx] #get memory set for class 1

	memory_x = torch.cat((mset_0_rand, mset_1_rand)) #make full memory set
	memory_y = torch.Tensor(np.array([0] * num_exemplars + [1] * num_exemplars)).long() #make full memory set
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)

	# Benchmark model performance on memory sets
	# f = open('output_CIFAR.txt', 'a')
	print('CIFAR Benchmarking', file=f)
	print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (iCaRL Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	memory_x, memory_y = icarl.create_memory_set(X, y, loss_type='icarl')
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)

	# Benchmark model performance on memory sets
	# f = open('output_CIFAR.txt', 'a')
	print('CIFAR Benchmarking', file=f)
	print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (replay Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	memory_x, memory_y = icarl.create_memory_set(X, y, loss_type='replay')
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)
	f.close()

	return None

# (iCaRL)(non-Incremental) Memory set on MNIST  data
def iCaRL_MNIST_benchmarking():
	PATH = './mnist_net.pth'
	batch_size = 10
	epochs = 10
	classes = 2
	channels = 3
	feature_dim = 2028
	input_shape = (28, 28)
	input_dim = 16 * 4 * 4
	p = 0.05
	max_data_size = 1000

	# Generate MNIST training data
	transform = transforms.Compose([transforms.ToTensor(), 
									transforms.Lambda(gray_to_rgb), 
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.MNIST(root='./data', train=True,
	                                        download=True, transform=transform)
	testset = torchvision.datasets.MNIST(root='./data', train=False,
	                                        download=True, transform=transform)
	
	# Select two classes
	first_two_classes_idx = np.where((np.array(trainset.targets) == 0) | (np.array(trainset.targets) == 1))[0][:max_data_size]
	imgs, labels = zip(*trainset)

	X = torch.stack(imgs, 0) #turn images into 3-D tensors
	y = torch.Tensor(labels).long() #turn labels into tensors
	X = X[first_two_classes_idx] #take subset of images for the first two classes
	y = y[first_two_classes_idx] #take subset of labels for the first two classes

	X_0, y_0, X_1, y_1 = subset_data(X, y, (0, 1))

	num_exemplars = int(p * max_data_size / 2.) #number of images exemplar set

	# Make training and test data loaders
	trainsubset = torch.utils.data.Subset(trainset, first_two_classes_idx)
	trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	first_two_classes_idx = np.where((np.array(testset.targets) == 0) | (np.array(testset.targets) == 1))[0]
	testsubset = torch.utils.data.Subset(testset, first_two_classes_idx)
	testloader = torch.utils.data.DataLoader(testsubset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	# Load cnn classifier (trained on full dataset) from path
	classifier = CNNnet(input_dim, classes, channels)
	classifier.load_state_dict(torch.load(PATH))
	# Get test accuracy of classifier (trained on full dataset)
	full_accuracy = evaluate_classifier(testloader, classifier, print_output=False)

	# Benchmark model performance on memory sets
	f = open('output_MNIST.txt', 'a')
	print('MNIST Benchmarking', file=f)
	print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (Random Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	mset_0_idx = icarl.construct_exemplar_set(X_0, y_0) #get memory set indices for class 0
	mset_1_idx = icarl.construct_exemplar_set(X_1, y_1) #get memory set indices for class 1
	mset_0_rand = X_0[mset_0_idx] #get memory set for class 0
	mset_1_rand = X_1[mset_1_idx] #get memory set for class 1

	memory_x = torch.cat((mset_0_rand, mset_1_rand)) #make full memory set
	memory_y = torch.Tensor(np.array([0] * num_exemplars + [1] * num_exemplars)).long() #make full memory set
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)

	# Benchmark model performance on memory sets
	# f = open('output_CIFAR.txt', 'a')
	# print('MNIST Benchmarking', file=f)
	# print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (iCaRL Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	memory_x, memory_y = icarl.create_memory_set(X, y, loss_type='icarl')
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)

	# Benchmark model performance on memory sets
	# f = open('output_CIFAR.txt', 'a')
	# print('MNIST Benchmarking', file=f)
	# print('Memory set size:', int(p * X.shape[0]), file=f)
	print(f'\n\nBenchmarking iCaRL (replay Features)', file=f)
	print(f'************************', file=f)
	print(f'Test accuracy of the network trained on full data: {full_accuracy}%', file=f)

	# Compute memory sets using random features
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='cnn') #instantiate icarl model
	memory_x, memory_y = icarl.create_memory_set(X, y, loss_type='replay')
	
	# Make memory set loader
	memory_set = torch.utils.data.TensorDataset(memory_x, memory_y)
	memory_loader = torch.utils.data.DataLoader(memory_set, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Train and evaluate classifier on memory set
	memory_classifier = train_classifier(input_dim, classes, channels, memory_loader, epochs=epochs, architecture='cnn', check_point=20)
	memory_accuracy = evaluate_classifier(testloader, memory_classifier, print_output=False)
	
	print(f'Test accuracy of the network trained on memory set: {memory_accuracy}%', file=f)
	f.close()

	return None

def main():
	iCaRL_MNIST_benchmarking()

if __name__ == "__main__":
	main()