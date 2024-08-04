#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p shared 
#SBATCH --mem=100GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o mnist_official/mnist/out_%j.txt # File to which STDOUT will be written
#SBATCH -e mnist_official/mnist/err_%j.txt # File to which STDERR will be written

module load cuda/12.2.0-fasrc01
module load python/3.10.9-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MAIN_PATH="/n/home12/thb489/new_continual_learning/continual-learning"
VENV_PATH="/n/home12/thb489/new_continual_learning/continual-learning/myenv"

cd ${MAIN_PATH}
source "${VENV_PATH}/bin/activate"

python -u ${MAIN_PATH}/run.py mnist_official/mnist mnist '{"p": 0.5, "T": 5, "learning_rate": 0.5, "batch_size": 10, "num_centroids": 4, "model_training_epoch": 1, "early_stopping_threshold": 1.0, "random_seed": 0, "class_balanced": true, "max_data_size": 100}'
