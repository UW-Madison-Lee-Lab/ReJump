"""
Preprocess dataset for cosine regression task - a synthetic regression task with n-dimensional data points
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
    """Generate synthetic cosine regression dataset.
    
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
        # Generate random coefficients between -2 and 2
        coef = np.random.uniform(-2, 2, n_features)
        p = n_features
        intercept = np.random.uniform(-2, 2)
    else:
        if n_features != 2: raise ValueError(f"n_features must be 2 for default coefficients, got {n_features}")
        # default coefficients for 2D case
        coef = np.ones(2)
        intercept = 0.0
        p = n_features
    
    # Generate random feature matrix
    X = np.random.uniform(-1, 1, (num_samples, n_features))
    
    # Generate target values using cosine function
    y = np.sum(coef * np.cos(2 * np.pi * X), axis=1) / p + intercept
    
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
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default", "grid", "mixed"])
    parser.add_argument('--label_noise', type=float, default=0.0)
    args = parser.parse_args()
    set_seed(42)
    
    data_source = 'cosreg'
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
