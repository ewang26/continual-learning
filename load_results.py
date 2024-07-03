import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pickle
import pandas as pd
import glob
from run import OUTPUT_DIR


def main():
    
    results = []
    for fname in glob.glob(f'{OUTPUT_DIR}/*/*'):
        print('Reading from {}'.format(fname))
        with open(fname, 'rb') as f:
            result = pickle.load(f)
            results.append(result)
          
    df = pd.DataFrame(results)

if __name__ == '__main__':
	main()


