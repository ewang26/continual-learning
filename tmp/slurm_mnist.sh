#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p shared 
#SBATCH --mem=100GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o mnist_official_gss/mnist/out_%j.txt # File to which STDOUT will be written
#SBATCH -e mnist_official_gss/mnist/err_%j.txt # File to which STDERR will be written

module load cuda/12.2.0-fasrc01
module load python/3.10.9-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MAIN_PATH="/n/home12/thb489/new_continual_learning/continual-learning"
VENV_PATH="/n/home12/thb489/new_continual_learning/continual-learning/myenv"

cd ${MAIN_PATH}
source "${VENV_PATH}/bin/activate"

python -u ${MAIN_PATH}/run.py mnist_official_gss/mnist mnist '{"p": 0.9, "T": 5, "learning_rate": 0.001, "batch_size": 50, "num_centroids": 4, "model_training_epoch": 30, "early_stopping_threshold": 1000000, "random_seed": 4, "class_balanced": true, "execute_early_stopping": false}'
