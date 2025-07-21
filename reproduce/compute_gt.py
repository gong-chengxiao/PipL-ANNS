import os
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OPENBLAS_NUM_THREADS'] = '64'

import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.io import fvecs_read, fvecs_write
from settings import datasets_dir, datasets, TOPK

def compute_gt(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=64)
    nbrs.fit(base)
    distances, indices = nbrs.kneighbors(query)
    
    return np.array([indices[i] for i in range(len(indices))])

if __name__ == "__main__":
    for DATASET in datasets:
        base = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_query.fvecs")

        gt = compute_gt(base, query, TOPK)
        fvecs_write(f"{datasets_dir}/{DATASET}/{DATASET}_groundtruth.fvecs", gt)