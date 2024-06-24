from data import *

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import pdb

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


#######
####### Util for training, evaluating, computing gradients for NN classifiers
#######

# CNN for MNIST & CIFAR
class CNNnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train NN classifier
def train_classifier(input_dim, output_dim, trainloader, epochs=10, architecture='fc'):
	if architecture == 'fc':
		classifier = FCnet(input_dim, output_dim)
	if architecture == 'cnn':
		classifier = CNNnet(input_dim, output_dim)

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
	        if i % 10 == 9:    
	            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
	            running_loss = 0.0

	return classifier

# Evaluate trained classifier
def evaluat_classifier(loader, classifier):
	correct = 0
	total = 0

	with torch.no_grad():
	    for data in loader:
	        X, y = data
	        outputs = classifier(X)
	        _, predicted = torch.max(outputs.data, 1)
	        total += y.size(0)
	        correct += (predicted == y).sum().item()

	print(f'Accuracy of the network: {100 * correct // total} %')

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
def compute_sample_grads(classifier, data, targets):
    """ manually process each sample with per sample gradient """
    batch_size = data.shape[0]
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
    outputs = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))
    _, labels_predicted = torch.max(outputs, 1)
    # plot decision boundary in region of interest
    z = labels_predicted.detach().numpy().reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=cm, alpha=0.2)
    return ax

# Plot memory set against data for toy
def plot_memory_set(mset_manager, X, y, cm, title, ax, classifier, rand, method=None):
	memory_x, memory_y = mset_manager.create_memory_set(torch.from_numpy(X), torch.from_numpy(y))
	X = torch.from_numpy(X).float()
	y = torch.from_numpy(y).long()

	if method == 'lambda':
		output = classifier(X)
		memory_x, memory_y = mset_manager.update_memory_lambda(memory_x, memory_y, X, y, output)

	if method == 'GSS':
		# 1. compute per sample gradient for a subset of the memory set (we use the whole set):
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
																				   grad_sample, grad_batch)
			
			if i > mset_manager.memory_set_size:
				diff = np.linalg.norm(similarity_scores - prv_scores)
				if diff > 0:
					replacement_counter += 1
		print('GSS replacements:', replacement_counter)
	
	memory_x = memory_x.cpu().detach().numpy()
	memory_y = memory_y.cpu().detach().numpy()

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
####### Memory set selection comparison on toy data
#######

# Memory set on toy data: 2D classification, 2 classes, multiple clusters per class
def main():
	classes = 2
	input_dim = 2
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

	# Train a nn classifier
	batch_size = 30
	traindata = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
	trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
	classifier = train_classifier(2, classes, trainloader, epochs=20)
	evaluat_classifier(trainloader, classifier)

	# Compute memory sets
	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=10)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)


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



if __name__ == "__main__":
	main()