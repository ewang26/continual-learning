#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -p test # odyssey partition
#SBATCH --mem=10000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/mnist/out_%j.txt # File to which STDOUT will be written
#SBATCH -e output/mnist/err_%j.txt # File to which STDERR will be written

python -u run.py output/mnist mnist '{"p": 0.01, "T": 2, "learning_rate": 0.001, "batch_size": 10, "num_centroids": 2, "model_training_epoch": 10, "early_stopping_threshold": 0.1, "random_seed": 0, "class_balanced": true}'

