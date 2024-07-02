from data import *
from models import *
from tasks_training import *

from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

# Simple feedforward model for toy classification
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
def plot_decision_boundary(model, ax, cm, xlim, ylim, classes, steps=1000):
    # define region of interest by data limits
    xmin = xlim[0]
    xmax = xlim[1]
    ymin = ylim[0]
    ymax = ylim[1]
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # make predictions across region of interest
    model.eval()
    outputs = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])))[:, :classes]
    _, labels_predicted = torch.max(outputs, 1)
    # plot decision boundary in region of interest
    z = labels_predicted.detach().numpy().reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=cm, alpha=0.2)
    return ax

# Make data for tasks
def make_tasks_data(n_samples, n_classes, tasks=2):

	# Generate data for Task 1
	X_t1, y_t1 = mixture_of_gaussians(RandomState(10), n_samples=n_samples, n_classes=n_classes, 
								n_clusters_per_class=2, cluster_sep=2., 
								cluster_var=1.)
	# Generate data for Task 2
	X_t2, y_t2 = mixture_of_gaussians(RandomState(24), n_samples=n_samples, n_classes=n_classes, 
								n_clusters_per_class=2, cluster_sep=3., 
								cluster_var=1.)
	# Adjust the cluster centers of data in Task 2
	X_t2 += np.array([[-13, 6]])
	y_t2 += 2

	# Define the tasks
	tasks_data = {0: (torch.from_numpy(X_t1), torch.from_numpy(y_t1).long()), 
				  1: (torch.from_numpy(X_t2), torch.from_numpy(y_t2).long())}

	X = np.vstack((X_t1, X_t2))
	xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
	ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

	return tasks_data, (xmin, xmax), (ymin, ymax)

def main():
	# Define experiment parameters
	n_classes = 2
	n_samples = 400
	input_dim = 2
	feature_dim = 120
	model_training_epoch = 20
	classes_per_task = 2
	check_point = 1
	random_seed = 1
	early_stopping_threhold = 0.05
	kmeans_max_iter = 20
	p = 0.2
	num_centroids = 2
	device = 'cpu'
	num_exemplars = int(p * n_samples / 2.)
	output_dim = 4
	icarl_loss_type = 'replay'

	# Define memory sets
	#random memory set
	random_mset = RandomMemorySetManager(p)
	#kmeans memory set
	kmeans_mset = KMeansMemorySetManager(p, num_centroids, device, max_iter=kmeans_max_iter)
	#lambda memory set
	lambda_mset = LambdaMemorySetManager(p)
	#GSS memory set
	GSS_mset = GSSMemorySetManager(p)
	#icarl memory set
	icarl = iCaRL(input_dim, feature_dim, num_exemplars, p, architecture='fc')
	
	method = f'iCaRL ({icarl_loss_type} loss)'
	mset_manager = icarl

	# Make data for two tasks, two classes each
	tasks_data, xlim, ylim = make_tasks_data(n_samples, n_classes)

	# Plot memory sets against data
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	cm = {0: plt.cm.PRGn, 1: plt.cm.bwr} #color map for data
	memory_cm = {0: ['purple', 'green'], 1: ['blue', 'red']} #colors for memory sets

	# Scatter plot the data for the tasks
	for t in tasks_data.keys():
		X = tasks_data[t][0]
		y = tasks_data[t][1]
		classes = np.unique(y)
		X_0 = X[y == classes[0]]
		X_1 = X[y == classes[1]]

		ax[t].scatter(X_0[:, 0], X_0[:, 1], s=40, marker='o', c=memory_cm[t][0], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')
		ax[t].scatter(X_1[:, 0], X_1[:, 1], s=40, marker='o', c=memory_cm[t][1], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')
		
		if t == 0:
			ax[t + 1].scatter(X_0[:, 0], X_0[:, 1], s=40, marker='o', c=memory_cm[t][0], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')
			ax[t + 1].scatter(X_1[:, 0], X_1[:, 1], s=40, marker='o', c=memory_cm[t][1], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')

	# Instantiate feedforward model & loss
	model = FCnet(input_dim, output_dim)
	criterion = nn.CrossEntropyLoss()

	# Define arguments for the CL task
	kwargs = {'model_training_epoch': model_training_epoch, 
			  'check_point': check_point, 
			  'early_stopping_threhold': early_stopping_threhold,
			  'icarl_loss_type': icarl_loss_type}
	# Train and evaluate CL task
	performances, backward_transfer, memory_sets, models = CL_tasks(tasks_data, model, criterion, mset_manager, random_seed=random_seed, return_models=True, **kwargs)

	# Compute and plot memory sets for each task
	for t in tasks_data.keys():

		memory_x = memory_sets[t][0]
		memory_y = memory_sets[t][1]

		model = models[t]

		classes = np.unique(memory_y)
		current_classes = classes_per_task * (t + 1)

		mset_0 = memory_x[memory_y == classes[0]]
		mset_1 = memory_x[memory_y == classes[1]]
		
		if t == 0:
			task_cm = ListedColormap(memory_cm[t])
			ax[t] = plot_decision_boundary(model, ax[t], task_cm, xlim, ylim, current_classes, steps=1000)
		else:
			complete_cm = ListedColormap(memory_cm[t - 1] + memory_cm[t])
			ax[t] = plot_decision_boundary(model, ax[t], complete_cm, xlim, ylim, current_classes, steps=1000)

		ax[t].scatter(mset_0[:, 0], mset_0[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
		ax[t].scatter(mset_1[:, 0], mset_1[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')
		if t == 0:
			ax[t + 1].scatter(mset_0[:, 0], mset_0[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
			ax[t + 1].scatter(mset_1[:, 0], mset_1[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')

		if t == 1:
			legend = ax[t].legend(loc='best', bbox_to_anchor=(1.1, 1.05))
			for lh in legend.legend_handles: 
				lh.set_alpha(1)

		ax[t].set_title(f'Memory Set Selection: {method} (Task {t})')

	# Print performances
	print(f'{method} performance: {performances}')
	fig.savefig(f'toy_{method}.png', bbox_extra_artists=(legend,), bbox_inches='tight', dpi=fig.dpi)
	plt.tight_layout()
	plt.show()	

if __name__ == "__main__":
	main()
