"""
Preprocess dataset for piecewise regression task - a synthetic regression task with n-dimensional data points
"""

import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, prepare_dataset

def gen_dataset(
    num_samples: int,
    feature_noise: float = 1.0,
    seed_value: int = 42,
    n_features: int = 2,
    label_noise: float = 0.0,
    random: bool = False,
) -> List[Tuple]:
    """Generate synthetic piecewise regression dataset.
    
    Args:
        num_samples: Number of samples to generate
        feature_noise: Standard deviation of the noise
        seed_value: Random seed for reproducibility
        n_features: Number of features in the dataset
        random: Whether to use random coefficients
        label_noise: Standard deviation of the noise added to labels
    Returns:
        List of tuples containing (features, target)
    """
    np.random.seed(seed_value)
    
    if random:
        coef = np.random.uniform(-1, 1, n_features)
    else:
        coef = np.ones(n_features) / n_features
    
    # Generate random feature matrix
    X = np.random.uniform(-1, 1, (num_samples, n_features))
    
    # Apply piecewise function to each feature
    def piecewise_func(x, j):
        if x < -coef[j]:
            return x - coef[j]
        elif x < coef[j]:
            return 0
        else:
            return x + coef[j]
    
    # Apply the piecewise function to each feature and sum them
    y = np.zeros(num_samples)
    for i in range(num_samples):
        feature_sum = 0
        for j in range(n_features):
            feature_sum += piecewise_func(X[i, j], j)
        y[i] = feature_sum / n_features  # Normalize by number of features
    
    # Add noise to features if specified
    if feature_noise > 0:
        X = X + np.random.normal(0, feature_noise, size=X.shape)
    
    # Add additional noise to labels if specified
    if label_noise > 0:
        # Scale label noise by the standard deviation of y
        y_std = np.std(y)
        label_noise = label_noise * y_std
        y = y + np.random.normal(0, label_noise, size=y.shape)
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        target = float(y[i])
        samples.append((features, target))
    
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--feature_noise', type=float, default=1.0)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=10)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default", "grid", "mixed"])
    parser.add_argument('--label_noise', type=float, default=0.0)
    args = parser.parse_args()
    set_seed(42)
    
    data_source = 'pwreg'
    n_classes = None  # Not applicable for regression
    
    datasets = prepare_dataset(args, gen_dataset)
    
    save_data(
        datasets['dataset_dict'],
        datasets['in_context_dataset_dict'],
        data_source,
        args,
        n_classes,
        datasets['TRAIN_SIZE'],
        datasets['TEST_SIZE'],
        data_mode=args.data_mode
    )
