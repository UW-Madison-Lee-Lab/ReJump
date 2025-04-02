"""
Preprocess dataset for blobs task - a synthetic classification task with n-dimensional data points
"""

import numpy as np
from sklearn.datasets import make_blobs
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, classification_reward_fn, flip_label, prepare_dataset
import pdb
import json

def gen_dataset(
    num_samples: int,
    noise_level: float = 1.0,
    seed_value: int = 42,
    label_flip_rate: float = 0.0,
    random: bool = False,
) -> List[Tuple]:
    """Generate synthetic blob dataset for classification task.
    
    Args:
        num_samples: Number of samples to generate
        noise_level: Standard deviation of the clusters
        seed_value: Random seed for reproducibility
        label_flip_rate: Label flip rate
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    if random: 
        centers = None
    else:
        # default centers
        centers = [
            [-2.50919762,  9.01428613],
            [ 4.63987884,  1.97316968],
            [-6.87962719, -6.88010959]
        ]
        
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=2,
        centers=centers,
        cluster_std=noise_level,
        random_state=seed_value,
    )
    
    y = flip_label(y, label_flip_rate, 3)
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        label = int(y[i])
        samples.append((features, label))
    
    return samples



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--noise_level', type=float, default=1.0)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=10)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--label_flip_rate', type=float, default=0.0)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default", "grid", "mixed"])
    args = parser.parse_args()
    set_seed(42)
    
    data_source = 'blobs'
    n_classes = 3
    
    
    datasets = prepare_dataset(args, gen_dataset)
    
    save_data(
        datasets['dataset_dict'],
        datasets['in_context_dataset_dict'],
        data_source,
        args,
        n_classes,
        datasets['TRAIN_SIZE'],
        datasets['TEST_SIZE'],
        data_mode = args.data_mode
    )


def blobs_reward_fn(solution_str, ground_truth):
    return classification_reward_fn(solution_str, ground_truth)
