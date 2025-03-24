"""
Preprocess dataset for linear classification task - a synthetic binary classification task based on linear boundary
"""

import numpy as np
from typing import List, Tuple
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, classification_reward_fn, flip_label

def gen_dataset(
    num_samples: int,
    noise: float = 0.1,
    seed_value: int = 42,
    coefficients: List[float] = [3.55, -0.3],
    intercept: float = 2.0,
    label_flip_rate: float = 0.0,
) -> List[Tuple]:
    """Generate synthetic linear binary classification dataset.
    
    Args:
        num_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        seed_value: Random seed for reproducibility
        coefficients: Coefficients for the linear boundary
        intercept: Intercept for the linear boundary
        
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    # To ensure a balanced dataset, we'll generate samples for each class separately
    samples_per_class = num_samples // 2
    
    X, y = [], []
    
    # Generate positive class samples (y > 0)
    count_positive = 0
    while count_positive < samples_per_class:
        # Generate random features
        feature_vector = np.random.uniform(-5, 5, len(coefficients))
        
        # Calculate raw value using linear function
        raw_value = np.dot(feature_vector, coefficients) + intercept
        
        # Add noise
        noisy_value = raw_value + np.random.normal(0, noise)
        
        # Check if it belongs to the positive class
        if noisy_value > 0:
            X.append(feature_vector.tolist())
            y.append(1)
            count_positive += 1
    
    # Generate negative class samples (y ≤ 0)
    count_negative = 0
    while count_negative < samples_per_class:
        # Generate random features
        feature_vector = np.random.uniform(-5, 5, len(coefficients))
        
        # Calculate raw value using linear function
        raw_value = np.dot(feature_vector, coefficients) + intercept
        
        # Add noise
        noisy_value = raw_value + np.random.normal(0, noise)
        
        # Check if it belongs to the negative class
        if noisy_value <= 0:
            X.append(feature_vector.tolist())
            y.append(0)
            count_negative += 1
    
    y = flip_label(y, label_flip_rate, 2)
    
    samples = list(zip(X, y))
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--label_flip_rate', type=float, default=0.0)
    args = parser.parse_args()
    set_seed(42)

    data_source = 'linear'
    TEST_SIZE = int(args.num_samples * args.test_ratio)
    TRAIN_SIZE = args.num_samples - TEST_SIZE
    n_classes = 2  # Binary classification task
    
    # Generate synthetic dataset
    samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise_level,
        seed_value=12
    )
    
    in_context_samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise_level,
        label_flip_rate=args.label_flip_rate,
        seed_value=34
    )
    
    dataset_dict = {
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    }
    
    in_context_dataset_dict = {
        'features': [sample[0] for sample in in_context_samples],
        'label': [sample[1] for sample in in_context_samples]
    }
    
    save_data(
        dataset_dict,
        in_context_dataset_dict,
        data_source,
        args,
        n_classes,
        TRAIN_SIZE,
        TEST_SIZE
    )
def linear_reward_fn(solution_str, ground_truth):
    return classification_reward_fn(solution_str, ground_truth)
