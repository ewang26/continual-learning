from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
torch.set_default_dtype(torch.float64)
from torch import linalg as LA
from torch.autograd import Variable
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torchvision.models import resnet18

from PIL import Image
import numpy as np
from numpy.random import RandomState
from jaxtyping import Float
from sklearn.cluster import KMeans
import pdb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Check for M1 Mac MPS (Apple Silicon GPU) support
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Using M1 Mac")
    DEVICE = torch.device("mps")
# Check for CUDA support (NVIDIA GPU)
elif torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
# Default to CPU if neither is available
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")

# DEVICE = torch.device("mps") # change this before cluster!
torch.autograd.set_detect_anomaly(True)


class MemorySetManager(ABC):
    @abstractmethod
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]):
        """Creates a memory set from the given dataset. The memory set is a subset of the dataset."""
        pass

# CNN for MNIST & CIFAR
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_size, n_heads=2):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        # Linear layer: one head for each classes
        self.fc = nn.Linear(feature_size, n_heads) #initially n number of heads


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def extract_features(self, x):
        x = self.feature_extractor(x)
        return x

# Simple feedforward for toy
class FCFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_size):
        super().__init__()
        self.feature_extractor_1 = nn.Linear(input_dim, 80)
        self.feature_extractor_2 = nn.Linear(80, feature_size)
        # Linear layer: one head for each classes
        self.fc = nn.Linear(feature_size, 2) #initially only two heads

    def forward(self, x):
        x = self.feature_extractor_1(x)
        x = self.feature_extractor_2(x)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def extract_features(self, x):
        x = self.feature_extractor_1(x)
        x = self.feature_extractor_2(x)
        return x

# WP: checked
# WP: why don't we ever store the memory sets? i.e. the memory set managers do not manage memory sets, 
# they only create them. "Task" stores memory set.
class RandomMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)

    # Randomly select elements from the dataset to be in the memory set.
    @torch.no_grad()
    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"], class_balanced=True,
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Creates random memory set.

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        memory_x = torch.empty(0)
        memory_y = torch.empty(0)

        if self.p == 1:
            return x, y

        else:
            memory_set_size = int(x.shape[0] * self.p)

            if class_balanced:
                memory_set_size = int(x.shape[0] * self.p)
                classes = torch.unique(y)
                num_classes = len(classes)
                num_exemplars = int(memory_set_size / num_classes)

                memory_x = []
                memory_y = []

                for c in classes:
                    x_c = x[y == c]
                    y_c = y[y == c]

                    memory_set_indices = torch.randperm(x_c.shape[0], generator=self.generator)[:num_exemplars]

                    memory_x.append(x_c[memory_set_indices])
                    memory_y.append(y_c[memory_set_indices])

                memory_x = torch.cat(memory_x, dim=0)
                memory_y = torch.cat(memory_y, dim=0)

            else:
                # Select memeory set random elements from x and y, without replacement
                memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[:memory_set_size]

                memory_x = x[memory_set_indices]
                memory_y = y[memory_set_indices]

        return memory_x, memory_y

# WP: checked
# WP: need to batch the kmeans algo
# WP: why are we passing device in init?
#THEODORA K-MEANS
class KMeansMemorySetManager(MemorySetManager):
    def __init__(self, p: float, num_centroids: int, device: torch.device, random_seed=42, max_iter=100):
        """
        Args:
            p: The percentage of samples to retain in the memory set.
            num_centroids: The number of centroids to use for K-Means clustering.
            device: The device to use for computations (e.g., torch.device("cuda")).
            random_seed: The random seed for reproducibility.
        """
        self.p = p
        self.num_centroids = num_centroids
        self.max_iter = max_iter
        self.device = device
        
        # Set the random seed for reproducibility
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)
    
    @torch.no_grad()
    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m 1"]]:
        """Creates memory set using K-Means clustering for each class.
        Args:
            x: x data.
            y: y data.
        
        Returns:
            (memory_x, memory_y) tuple, where memory_x and memory_y are tensors.
        """
        memory_x = torch.empty(0)
        memory_y = torch.empty(0)

        if self.p == 1:
            return x, y

        else:
            device = x.device
            n = x.shape[0]
            f_shape = list(x.shape[1:])
            f = np.prod(f_shape)
            memory_set_size = int(n * self.p)
            
            # Get unique classes
            classes = torch.unique(y).tolist()
            num_classes = len(classes)
            
            # Calculate the memory size per class
            mset_size_per_class = memory_set_size // num_classes


            # Initialize dictionaries to store centroids, cluster counters, and memory sets for each class
            cluster_counters = {}
            centroids = {}
            
            # Initialize memory set related dictionaries
            memory_x = {} #dictionary -- key: class, value: memory set for class
            memory_y = {} #dictionary -- key: class, value: class labels
            memory_distances = {} #dictionary -- key: class, value: distance of memory set elements from mean
            
            # Iterate over each class: contruct the memory set for each class
            for label in classes:

                # Initialize memory set related objects for current class
                memory_x[label] = torch.zeros([mset_size_per_class] + f_shape, dtype=torch.float64, device=device)
                memory_y[label] = torch.zeros((mset_size_per_class, 1), dtype=torch.long, device=device)
                memory_distances[label] = torch.full((mset_size_per_class,), float("inf"), device=device)

                # Get the set of elements labeled with the current class from the data
                class_mask = (y == label).squeeze()
                class_x = x[class_mask]
                class_x = torch.flatten(class_x, 1, -1)
                
                # Initialize centroids and cluster counters for the current class
                centroids[label] = torch.randn((self.num_centroids, f), dtype=torch.float64, device=device, generator=self.generator)
                cluster_counters[label] = torch.zeros(self.num_centroids, dtype=torch.float64, device=device)
                
                # Do kmeans on elements of the current class
                # i.e. find the closest centroid for each element, cluster the element 
                # and update the corresponding centroid
                for _ in range(self.max_iter):
                    # Find the closest centroid for each element
                    distances = torch.cdist(class_x, centroids[label])
                    closest_centroids = torch.argmin(distances, dim=1)
                    centroid_idx, centroid_addcount = closest_centroids.unique(return_counts=True)
                    # breakpoint()
                    for i in range(len(centroid_idx)):
                        idx = centroid_idx[i]
                        # KMEANS update: cluster each element to the closest centroid and update the centroid (cluster mean)
                        curr_centroid = centroids[label][idx] # current cluster means
                        curr_cluster_size = cluster_counters[label][idx] # current cluster sizes
                        curr_cluster_sum = curr_centroid * curr_cluster_size # sum of the elements in current clusters

                        cluster_counters[label][idx] += centroid_addcount[i] # increment the cluster size for the closest centroid
                        centroids[label][idx] = (curr_cluster_sum + torch.sum(class_x[closest_centroids == idx], dim=0)) / cluster_counters[label][idx]
                    
                # Create memory set from clusters
                for i in range(class_x.shape[0]):  

                    # Find the closest centroid for each element
                    element = class_x[i]
                    distances = LA.norm(centroids[label] - element, dim=1) # get the distance from current element to all centroids
                    closest_centroid_idx = torch.argmin(distances).item() # find the closest centroid
                    distance_to_centroid = distances[closest_centroid_idx] # get distance to the closest centroid
                                   
                    # Update the memory set for the current class:
                    # (1) If we haven't filled up the memory set quota for the current class,
                    # set the i-th element of the memory set for the current class to the current element
                    if i < mset_size_per_class:
                        memory_x[label][i] = element.reshape(f_shape)
                        memory_y[label][i] = label
                        memory_distances[label][i] = distance_to_centroid
                    # (2) otherwise, if the distance between the current element and its closest centroid is less than the 
                    # max radius of an existing cluster, replace the element farthest from its corresponding centroid with 
                    # the current element
                    else:
                        max_idx = torch.argmax(memory_distances[label]) # find the memory set element farthest from its closest centroid
                        if distance_to_centroid < memory_distances[label][max_idx]:
                            memory_x[label][max_idx] = element.reshape(f_shape)
                            memory_distances[label][max_idx] = distance_to_centroid
            
            # Concatenate memory sets (just values, without keys) from all classes
            memory_x = torch.cat(list(memory_x.values()), dim=0)
            memory_y = torch.cat(list(memory_y.values()), dim=0).view(-1)
        
        return memory_x, memory_y

# WP: need to verify the mset selection look good on toy
# Jonathan Lambda Method
class LambdaMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p

        # Set the random seed for reproducibility
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)

    @torch.no_grad()
    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m 1"]]:
        
        if self.p == 1:
            return x, y

        return torch.empty(0), torch.empty(0)

    @torch.no_grad()
    def update_memory(self, memory_x,  memory_y, X, y, outputs, num_classes, class_balanced=True):
        """
        Function to update the memory buffer in Lambda Memory Selection.

        Args:
            memory_x and memory_y: the existing memory datasets.
            X and y: the full data from the terminal task.
            outputs: tensor of size [n x k] where n is number of samples in X or y, and k is number of classes to classify into.
                Outputs of forward pass through the network of all data in X.
        
        Returns:
            memory_x and memory_y.long(): new memory datasets including the memory dataset for the existing task.
        """
        
        terminal_task_size = outputs.shape[0] #number of data points in current task
        memory_set_size = int(terminal_task_size * self.p) #size of memory set
        mset_size_per_class = memory_set_size // num_classes #size of memory set per class, for class-balanced selection
        traces = []

        # WP: can vectorize below
        # Approximates Hessian of L(x, w), fixing current w, and for each x in the input set
        for i in range(terminal_task_size):
            # take output layer and apply softmax to get probabilities of classification for each output
            class_prob = torch.softmax(outputs[i], dim=0)

            # create a matrix of p @ (1-p).T to represent decision uncertainty at each class
            decision_uncertainty = torch.outer(class_prob, (1 - class_prob))
            # calculate the trace of this matrix to assess the uncertainty in classification across multiple classes
            # the trace equivalent to the hessian of the loss wrt the output layer
            decision_trace = torch.trace(decision_uncertainty)
            traces.append(decision_trace.item())
        traces = np.array(traces)

        # Class balanced selection: enforces equal number of memory set elements per class
        if class_balanced:
            for k in range(num_classes):
                traces_k = traces[y == k]
                X_k = X[y == k]
                y_k = y[y == k]

                # getting indexes of the highest trace 
                argsorted_indx = sorted(range(len(traces_k)), key=lambda idx: traces_k[idx], reverse=True)
                desired_indx = argsorted_indx[:mset_size_per_class]

                # add to memory set
                memory_x = torch.cat((memory_x, X_k[desired_indx]))
                memory_y = torch.cat((memory_y, y_k[desired_indx]))

        # Class agnostic selection: enforces total number of memory set elements over all classes
        else: 
            # getting indexes of the highest trace 
            argsorted_indx = sorted(range(len(traces)), key=lambda x: traces[x], reverse=True)
            desired_indx = argsorted_indx[:memory_set_size]
            idx = desired_indx[0]

            # add to memory set
            memory_x = torch.cat((memory_x, X[desired_indx]))
            memory_y = torch.cat((memory_y, y[desired_indx]))


        return memory_x, memory_y.long()

# WP: checked
# WP: needs documentation
# Alan Gradient Sample Selection (GSS)
class GSSMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: fraction of task dataset to be included in replay buffer.
        """
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)
        self.rand = RandomState(random_seed)
        self.memory_set_size = 0
        self.mset_size_per_class = 0

    @torch.no_grad()
    def create_memory_set(self, x, y):
        """Initializes an empty memory replay buffer if training, called when task objects are created
        Else, use ideal model to generate GSS memory set

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        
        self.memory_set_size = int(x.shape[0] * self.p)
        classes = torch.unique(y).tolist()
        num_classes = len(classes)
        self.mset_size_per_class = self.memory_set_size // num_classes

        if self.p == 1:
            return x, y

        return torch.empty(0), torch.empty(0)

    @torch.no_grad()
    def update_GSS_greedy(self, memory_x, memory_y, similarity_scores, sample_x, sample_y, grad_sample, grad_batch, class_balanced=True):
        '''
        memory_x, y: current memory set for the task (to be used in later tasks)
        similarity_scores: current list of corresponding scores of each element in memory set
        sample_x, y: single new sample
        grad_sample: gradient of new sample
        grad_batch: gradient of random batch of memory_x,y
        '''
        # Class balanced selection: enforces equal number of memory set elements per class
        sample_shape = sample_x.shape
        sample_x = torch.flatten(sample_x, 1, -1)

        if class_balanced:
            indices_k = (memory_y == sample_y).numpy()
            if len(indices_k) > 0:
                memory_x_k = memory_x[indices_k]
                memory_y_k = memory_y[indices_k]
                grad_batch_k = grad_batch[indices_k]
                similarity_scores_k = similarity_scores[indices_k]

                
                sample_norm, batch_norm = np.linalg.norm(grad_sample), np.linalg.norm(grad_batch_k, axis=1)
                norm_product = sample_norm * batch_norm

                if self.memory_set_size == 0:
                    cosine_similarity = 0
                elif np.sum(norm_product) > 0:
                    cosine_similarity = (np.dot(grad_batch_k, grad_sample.T) / (sample_norm * batch_norm)) + 1
                    cosine_similarity = np.max(cosine_similarity)
                else:
                    cosine_similarity = 1 

                if memory_x_k.shape[0] < self.mset_size_per_class:
                    memory_x = torch.cat((memory_x, sample_x.reshape(sample_shape)), 0)
                    memory_y = torch.cat((memory_y, sample_y), 0)
                    similarity_scores = np.concatenate((similarity_scores, np.array([cosine_similarity])), 0)
                else:
                    probabilities = similarity_scores_k / np.sum(similarity_scores_k)
                    idx_k = self.rand.choice(np.arange(self.mset_size_per_class), p=probabilities)
                    r = self.rand.rand()
                    if r < (similarity_scores_k[idx_k] / (similarity_scores_k[idx_k] + cosine_similarity)):
                        diff = torch.flatten(memory_x - memory_x_k[idx_k], 1, -1)
                        diff = torch.sum(diff, dim=1)
                        idx = np.where(diff == 0)[0]
                        # assert len(idx) == 1
                        if len(idx) > 1:
                            print(len(idx))
                            idx = idx[:1]
                        memory_x[idx] = sample_x.reshape(sample_shape)
                        memory_y[idx] = sample_y
                        similarity_scores[idx] = cosine_similarity
                    assert len(memory_y[memory_y == sample_y]) == self.mset_size_per_class
            else:
                memory_x = torch.cat((memory_x, sample_x.reshape(sample_shape)), 0)
                memory_y = torch.cat((memory_y, sample_y), 0)
                similarity_scores = np.concatenate((similarity_scores, np.array([1])), 0)

        # Class agnostic selection: enforces total number of memory set elements over all classes
        else:
            sample_norm, batch_norm = np.linalg.norm(grad_sample), np.linalg.norm(grad_batch, axis=1)
            norm_product = sample_norm * batch_norm

            if self.memory_set_size == 0:
                cosine_similarity = 0
            elif np.sum(norm_product) > 0:
                cosine_similarity = (np.dot(grad_batch, grad_sample.T) / (sample_norm * batch_norm)) + 1
                cosine_similarity = np.max(cosine_similarity)
            else:
                cosine_similarity = 1 

            if memory_x.shape[0] < self.memory_set_size:
                memory_x = torch.cat((memory_x, sample_x.reshape(sample_shape)), 0)
                memory_y = torch.cat((memory_y, sample_y), 0)
                similarity_scores = np.concatenate((similarity_scores, np.array([cosine_similarity])), 0)
            else:
                probabilities = similarity_scores / np.sum(similarity_scores)
                idx = self.rand.choice(np.arange(self.memory_set_size), p=probabilities)
                r = self.rand.rand()
                if r < (similarity_scores[idx] / (similarity_scores[idx] + cosine_similarity)):
                    memory_x[idx] = sample_x.reshape(sample_shape)
                    memory_y[idx] = sample_y
                    similarity_scores[idx] = cosine_similarity

        return memory_x, memory_y.long(), similarity_scores

# WP: Checked for single task and incremental setting
class iCaRL(nn.Module):
    def __init__(self, input_dim, feature_size, num_exemplars, p, 
                 classes_per_task=2, 
                 random_seed=42, num_epochs=10, batch_size=64, learning_rate=0.002, 
                 loss_type='icarl', architecture='cnn'):
        super(iCaRL, self).__init__()

        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)
        self.loss_type = loss_type

        self.architecture = architecture
        # Feature Extractor architecture for image data
        if architecture == 'cnn':
            torch.manual_seed(random_seed)
            self.network = ResnetFeatureExtractor(input_dim, feature_size, classes_per_task)
        # Simple feature extractor architecture for toy data
        else:
            torch.manual_seed(random_seed)
            self.network = FCFeatureExtractor(input_dim, feature_size)

        # Exemplar data
        self.n_classes = 0 #number of classes
        self.exemplar_sets = {} #exemplars, keyed by class label
        self.exemplar_set_labels = {}
        self.memory_set_size = 0 #size of memory set across all classes
        self.mset_size_per_class = num_exemplars #size of each exemplar set (per class)
        
        # Loss functions and optimizers
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def create_memory_set(self, X, y):
        """ Create or update memory set for new tasks """

        # Update the feature representation with new data
        self.update_representation(X, y)

        memory_x = torch.empty(0)
        memory_y = torch.empty(0)

        if self.p == 1:
            return x, y
        else:
            # Construct exmemplar set for each class
            classes = torch.unique(y)
            memory_x = {}
            memory_y = []

            for c in classes:
                class_X = X[y == c]
                class_y = y[y == c]

                mset_indices = self.construct_exemplar_set(class_X, class_y)
                
                memory_x[c] = class_X[mset_indices]
                memory_y.append(class_y[mset_indices])

            memory_x = torch.cat(list(memory_x.values()), dim=0)
            memory_y = torch.cat(memory_y)
        
        return memory_x, memory_y

    @torch.no_grad() 
    def extract_features(self, x):
        self.network.eval()
        x = self.network.extract_features(x)
        return x

    @torch.no_grad() 
    def increment_classes(self, n):
        '''
        Add n number of heads for new classes in the final linear layer
        '''
        #replace the old linear output layer with a new layer t
        #hat has n number of extra head for n number of new classes
        in_features = self.network.fc.in_features
        out_features = self.network.fc.out_features
        weight = self.network.fc.weight.data
        bias = self.network.fc.bias.data

        self.network.fc = nn.Linear(in_features, out_features + n) #the linear layer constructor initializes all weights
        self.network.fc.weight.data[:out_features] = weight #set weights for existing classes to those in old layer
        self.network.fc.bias.data[:out_features] = bias #set biases for existing classes to those in old layer

        self.n_classes += n

    @torch.no_grad() 
    def construct_exemplar_set(self, X, y):
        """Construct an exemplar set for image set

        Args:
            images: torch tensor containing data of a class
        """
        assert len(torch.unique(y)) == 1
        current_class = torch.unique(y)[0]

        exemplar_indices = []
        exemplar_set = []
        exemplar_set_labels = []

        # Get features for input
        self.network.eval()
        features = self.network.extract_features(X)

        feature_shape = features.shape
        norms = torch.norm(features, dim=1, keepdim=True)  #get norm per feature vector
        assert norms.shape == (features.shape[0], 1)
        features = features / norms  #normalize feature vectors
        assert features.shape == feature_shape

        # Compute class mean
        class_mean = torch.mean(features, axis=0, keepdim=True) #compute the mean of the normalized feature vectors
        assert class_mean.shape == (1, features.shape[1])
        class_mean_norm = LA.norm(class_mean) #compute the norm of the mean feature vector
        class_mean = class_mean / class_mean_norm # normalize the feature vector
        assert class_mean.shape == (1, features.shape[1])

        # Construct exemplar set
        for k in range(self.mset_size_per_class):
            const = torch.Tensor([1. / (k + 1)])
            if k > 0:
                diffs_from_mean = class_mean -  const * (features + torch.sum(features[exemplar_indices, :], dim=0, keepdim=True))
            else:
                diffs_from_mean = class_mean - const * features
            assert diffs_from_mean.shape == features.shape

            dists_from_mean = LA.norm(diffs_from_mean, axis=1, keepdim=True)
            assert dists_from_mean.shape == (features.shape[0], 1)
            dists_from_mean.index_fill_(0, torch.Tensor(exemplar_indices).long(), float("Inf"))
            argmin_idx = torch.argmin(dists_from_mean)

            exemplar_set.append(X[argmin_idx])
            exemplar_set_labels.append(current_class)
            exemplar_indices.append(argmin_idx)

        # Updating the class variables
        self.exemplar_sets[current_class] = torch.stack(exemplar_set) # self.exemplar_sets should be a dictionay of tensors
        self.exemplar_set_labels[current_class] = torch.stack(exemplar_set_labels)
        return torch.tensor(exemplar_indices) # only need to return the indices of selected elements from the class data

    def update_representation(self, X, y):
        """Update the network representation using a new data (x) and their corresponding labels (y)."""
        #X, y = X.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            # Identify new classes
            new_classes = torch.unique(y)
            prev_classes = torch.Tensor(list(self.exemplar_sets.keys())).long()

            if self.n_classes > 0: #if the exemplar sets are not empty, concatenate exemplar sets with new data
                assert self.network.fc.out_features == self.n_classes #check: number of output heads is equal to the number of classes seen
                assert len(np.intersect1d(new_classes, prev_classes)) == 0 #check: y contains only new classes
                # Increment classes in feature extractor network
                self.increment_classes(len(new_classes))

                # concatenate memory and task data
                exemplar_xs = torch.cat(list(self.exemplar_sets.values()), dim=0)
                exemplar_ys = torch.cat(list(self.exemplar_set_labels.values()), dim=0)
                all_xs = torch.cat((exemplar_xs, X), dim=0)
                all_ys = torch.cat((exemplar_ys, y), dim=0)

            else: #otherwise use just the new data
                new_class_added = len(new_classes) - self.network.fc.out_features
                assert new_class_added >= 0 #check that the first set of classes added is greater than the initial number of heads
                if new_class_added > 0: 
                    # Increment classes in feature extractor network
                    self.increment_classes(new_class_added)
                else:
                    self.n_classes = self.network.fc.out_features

                all_xs = X
                all_ys = y
        
            combined_dataset = torch.utils.data.TensorDataset(all_xs, all_ys)
            combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)          

        # Train iCARL feature extraction network 
        # Train using Replay loss
        if self.loss_type == "replay": 
            print('training representation using replay loss')

            # We use cross entropy loss for replay
            criterion = nn.CrossEntropyLoss()

            # If exemplar sets exist train with exemplars and new data
            if self.n_classes > len(new_classes):
                with torch.no_grad():
                    # Dataloader for exemplar data
                    exemplar_dataset = TensorDataset(exemplar_xs, exemplar_ys)
                    exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=self.batch_size, shuffle=True)
                    # Dataloader for new data
                    new_dataset = TensorDataset(X, y)
                    new_dataloader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=True)

                # Train feature extractor
                for epoch in range(self.num_epochs):
                    for (batch_ex_xs, batch_ex_ys), (batch_xs, batch_ys) in zip(exemplar_dataloader, new_dataloader):
                        self.optimizer.zero_grad()  # Clear gradients before each backward pass

                        g_new = self.network.forward(batch_xs)  # Predictions on new data
                        loss_new = criterion(g_new, batch_ys)  

                        g_exemplar = self.network.forward(batch_ex_xs) # Predictions on exemplars
                        loss_exemplar = 1. / self.p * criterion(g_exemplar, batch_ex_ys)

                        loss = loss_new + loss_exemplar
                        loss.backward()  # Backpropagate the loss
                        self.optimizer.step()  # Update weights

            # Otherwise train with just the new data
            else:
                for epoch in range(self.num_epochs):
                    for batch_xs, batch_ys in combined_loader:
                        self.optimizer.zero_grad()  # Clear gradients before each backward pass
        
                        g_new = self.network.forward(batch_xs)  # Get predictions
                        loss_new = criterion(g_new, batch_ys)  # Compute loss
                        loss_new.backward()  # Backpropagate the loss
                        self.optimizer.step()  # Update weights
        
        # Train using the icarl loss
        else: 
            print('training representation using icarl loss')
            # We use a binary cross entropy loss for distillation and feature extraction loss
            criterion = nn.BCELoss()

            # Train with distillation
            if self.n_classes > len(new_classes):
                prev_feature_extractor = copy.deepcopy(self.network) 
                for epoch in range(self.num_epochs):
                    for batch_xs, batch_ys in combined_loader:
                        # Compute q values using old network
                        prev_feature_extractor.eval()
                        with torch.no_grad():
                            q_batch = torch.sigmoid(prev_feature_extractor.forward(batch_xs))
                        
                        # Clear gradients    
                        self.optimizer.zero_grad()
                        # Compute output using current network
                        g = torch.sigmoid(self.network.forward(batch_xs))
                        # Compute feature extraction loss for new classes
                        loss_new = 0
                        for i in range(len(new_classes)):
                            one_hot_labels = (batch_ys == new_classes[i]).double()
                            loss_new += criterion(g[:, i + len(prev_classes)], one_hot_labels)
                        # Compute distillation loss
                        loss_old = 0
                        for i in range(len(prev_classes)):
                            loss_old += criterion(g[:, i], q_batch[:, i])
                        # Define iCARL loss as the sum of feature extraction and distillation loss
                        loss_total = loss_new + loss_old
                        # Compute gradients in backward pass
                        loss_total.backward()
                        # Update weights
                        self.optimizer.step()
            # Train without distillation
            else:
                for epoch in range(self.num_epochs):
                    for batch_xs, batch_ys in combined_loader:
                        # Clear gradients    
                        self.optimizer.zero_grad()
                        # Compute output using current network
                        g = torch.sigmoid(self.network.forward(batch_xs))
                        # Compute feature extraction loss for new classes
                        loss_new = 0
                        for c in new_classes:
                            one_hot_labels = (batch_ys == c).double()
                            loss_new += criterion(g[:, c], one_hot_labels)
                        # Compute gradients in backward pass
                        loss_new.backward()
                        # Update weights
                        self.optimizer.step()


class GCRMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: fraction of task dataset to be included in replay buffer.
        """
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 1.5
        self.lambda_val = 1  # need to figure out this hyperparameter

    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Initializes an empty memory replay buffer if training, called when task objects are created
        Else, use GCR to generate memory set

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        self.memory_set_size = int(x.shape[0] * self.p)
        if self.p == 1:
            return x, y
        
        return torch.empty(0), torch.empty(0)
