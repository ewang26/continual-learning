from data import *
from models import *
from train_task import *

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


	# Generate data for Task 3
	X_t3, y_t3 = mixture_of_gaussians(RandomState(30), n_samples=n_samples, n_classes=n_classes, 
								n_clusters_per_class=2, cluster_sep=3., 
								cluster_var=1.)
	# Adjust the cluster centers of data in Task 2
	X_t3 += np.array([[16, 2]])
	y_t3 += 4


	# Define the tasks
	tasks_data = {
		0: (torch.from_numpy(X_t1), torch.from_numpy(y_t1).long()), 
		1: (torch.from_numpy(X_t2), torch.from_numpy(y_t2).long()),
		2: (torch.from_numpy(X_t3), torch.from_numpy(y_t3).long()),
	}

	X = np.vstack((X_t1, X_t2, X_t3))
	xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
	ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

	return tasks_data, (xmin, xmax), (ymin, ymax)

def main():
	# Define pipelin parameters
	model_PATH = './toy'
	train_full_only = False

	# Define experiment parameters
	n_classes = 2
	n_samples = 400
	input_dim = 2
	output_dim = 6
	feature_dim = 120
	classes_per_task = 2
	model_training_epoch = 5
	lr = 0.001
	check_point = 1
	random_seed = 1
	early_stopping_threshold = 0.8
	kmeans_max_iter = 20
	p = 0.1
	num_centroids = 2
	device = 'cpu'
	num_exemplars = int(p * n_samples / 2.)
	

	# Make data for two tasks, two classes each
	tasks_data, xlim, ylim = make_tasks_data(n_samples, n_classes)

	# Instantiate feedforward model & loss
	# Instantiate models
	model_keys = ['M1', 'M2', 'M3']
	models = {}
	for key in model_keys:
		torch.manual_seed(random_seed)
		models[key] = FCnet(input_dim, output_dim)
	
	# Define training loss
	criterion = nn.CrossEntropyLoss(reduction='none') #no reduction, for per-sample weights


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
	if train_full_only:
		_, _, _, _ = CL_tasks(
			tasks_data, 
			tasks_data, 
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
			iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='icarl', architecture='fc'), #icarl memory set
			iCaRL(input_dim, feature_dim, num_exemplars, p, loss_type='replay', architecture='fc'), #icarl memory set,
		]

		# Iterate through all memory managers
		for memory_set_manager in managers:
			memory_set_type = memory_set_manager.__class__.__name__

			if memory_set_type == 'iCaRL':
				kwargs['icarl_loss_type'] = memory_set_manager.loss_type
				memory_set_type += f' ({memory_set_manager.loss_type}) '

			performances, grad_similarities, models, memory_sets = CL_tasks(
				tasks_data, 
				tasks_data, 
				models, 
				criterion, 
				memory_set_manager=memory_set_manager, 
				use_memory_sets=True, 
				random_seed=1, 
				**kwargs,
			)

			# Plot memory sets against data
			fig, ax = plt.subplots(1, 2, figsize=(10, 5))
			cm = {0: plt.cm.PRGn, 1: plt.cm.bwr} #color map for data
			memory_cm = {0: ['purple', 'green'], 1: ['blue', 'red'], 2: ['cyan', 'brown']} #colors for memory sets

			# Scatter plot the data for the tasks
			for t in tasks_data.keys():
				X = tasks_data[t][0]
				y = tasks_data[t][1]
				classes = np.unique(y)
				X_0 = X[y == classes[0]]
				X_1 = X[y == classes[1]]

				ax[0].scatter(X_0[:, 0], X_0[:, 1], s=40, marker='o', c=memory_cm[t][0], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')
				ax[0].scatter(X_1[:, 0], X_1[:, 1], s=40, marker='o', c=memory_cm[t][1], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')

				ax[1].scatter(X_0[:, 0], X_0[:, 1], s=40, marker='o', c=memory_cm[t][0], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')
				ax[1].scatter(X_1[:, 0], X_1[:, 1], s=40, marker='o', c=memory_cm[t][1], edgecolors='none', alpha=0.1, label=f'Data (Task {t})')

			# Plot memory sets for each task
			for t in memory_sets.keys():
				memory_x = memory_sets[t][0]
				memory_y = memory_sets[t][1]

				classes = np.unique(memory_y)
				current_classes = classes_per_task * (t + 1)

				mset_0 = memory_x[memory_y == classes[0]]
				mset_1 = memory_x[memory_y == classes[1]]

				ax[0].scatter(mset_0[:, 0], mset_0[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
				ax[0].scatter(mset_1[:, 0], mset_1[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')
				
				ax[1].scatter(mset_0[:, 0], mset_0[:, 1], s=10, marker='^', c=memory_cm[t][0], alpha=0.7, label=f'Memory Set (Task {t})')
				ax[1].scatter(mset_1[:, 0], mset_1[:, 1], s=10, marker='^', c=memory_cm[t][1], alpha=0.7, label=f'Memory Set (Task {t})')


			
			complete_cm = ListedColormap(['purple', 'green', 'blue', 'red', 'cyan', 'brown'])
			ax[0] = plot_decision_boundary(models['M2'], ax[0], complete_cm, xlim, ylim, 6, steps=1000)
			ax[0].set_title(f'Memory Set Selection: {memory_set_type}\n\n M2 Decision Boundary')

			ax[1] = plot_decision_boundary(models['M3'], ax[1], complete_cm, xlim, ylim, 6, steps=1000)
			ax[1].set_title(f'Memory Set Selection: {memory_set_type}\n\n M3 Decision Boundary')

			legend = ax[1].legend(loc='best', bbox_to_anchor=(1.1, 1.05))
			for lh in legend.legend_handles: 
				lh.set_alpha(1)

			# Evaluate M2 on train data
			task_performances = evaluate(
				models['M2'], 
				criterion, 
				tasks_data, 
				batch_size=10, 
			)

			# Evaluate M3 on train data
			task_performances = evaluate(
				models['M3'], 
				criterion, 
				tasks_data, 
				batch_size=10, 
			)

			print('M2 per-task performance')
			print(task_performances)

			print('M3 per-task performance')
			print(task_performances)

			print('Gradient similarities')
			print(grad_similarities)

			fig.savefig(f'toy_{memory_set_type}.png', bbox_extra_artists=(legend,), bbox_inches='tight', dpi=fig.dpi)
			# plt.tight_layout()
			# plt.show()	

if __name__ == "__main__":
	main()
