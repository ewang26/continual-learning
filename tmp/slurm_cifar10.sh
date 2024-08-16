#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 3-00:00 # Runtime in D-HH:MM
#SBATCH -p gpu # GPU partition
#SBATCH --gres=gpu:1 # Request 1 GPU
<<<<<<< HEAD
#SBATCH --mem=100GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o test/cifar10/out_%j.txt # File to which STDOUT will be written
#SBATCH -e test/cifar10/err_%j.txt # File to which STDERR will be written
=======
#SBATCH --mem=1000GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o gss_mnist_cifar_zero_grad/cifar10/out_%j.txt # File to which STDOUT will be written
#SBATCH -e gss_mnist_cifar_zero_grad/cifar10/err_%j.txt # File to which STDERR will be written
>>>>>>> a14bb0a6f8b9fae3bd66eae6724ad56071a75c4a

module load cuda/12.2.0-fasrc01
module load python/3.10.9-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MAIN_PATH="/n/home12/thb489/new_continual_learning/continual-learning"
VENV_PATH="/n/home12/thb489/new_continual_learning/continual-learning/myenv"

cd ${MAIN_PATH}
source "${VENV_PATH}/bin/activate"

<<<<<<< HEAD
python -u ${MAIN_PATH}/run.py test/cifar10 cifar10 '{"p": 0.5, "T": 5, "learning_rate": 0.5, "batch_size": 10, "num_centroids": 4, "model_training_epoch": 1, "early_stopping_threshold": 5.0, "random_seed": 1, "class_balanced": true, "max_data_size": 100, "execute_early_stopping": false}'
=======
python -u ${MAIN_PATH}/run.py gss_mnist_cifar_zero_grad/cifar10 cifar10 '{"p": 0.9, "T": 5, "learning_rate": 0.001, "batch_size": 50, "num_centroids": 4, "model_training_epoch": 50, "early_stopping_threshold": 1000000, "random_seed": 4, "class_balanced": true, "execute_early_stopping": false}'
>>>>>>> a14bb0a6f8b9fae3bd66eae6724ad56071a75c4a
