"""
Preprocess dataset for blobs task - a synthetic classification task with n-dimensional data points
"""

import numpy as np
from sklearn.datasets import make_blobs
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, classification_reward_fn, flip_label
import pdb
import json
def gen_dataset(
    num_samples: int,
    cluster_std: float = 1.0,
    seed_value: int = 42,
    label_flip_rate: float = 0.0,
    random: bool = False,
) -> List[Tuple]:
    """Generate synthetic blob dataset for classification task.
    
    Args:
        num_samples: Number of samples to generate
        cluster_std: Standard deviation of the clusters
        seed_value: Random seed for reproducibility
        label_flip_rate: Label flip rate
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    if random: 
        # default centers
        centers = [
            [-2.50919762,  9.01428613],
            [ 4.63987884,  1.97316968],
            [-6.87962719, -6.88010959]
        ]
    else:
        centers = None
        
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed_value,
    )
    
    y = flip_label(y, label_flip_rate, 3)
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        label = int(y[i])
        samples.append((features, label))
    
    return samples


def gen_grid_dataset(grid_size=100, x_range=(-10, 10), y_range=(-10, 10)):
    """Generate a grid of points for visualization and testing.
    
    Args:
        grid_size: Number of points in each dimension
        x_range: Range of x values
        y_range: Range of y values
    
    Returns:
        List of tuples containing (features, dummy_label)
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    
    samples = []
    for i in range(grid_size):
        for j in range(grid_size):
            features = [xx[i, j], yy[i, j]]
            # Dummy label, will be predicted by the model
            label = 0
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
    
    if args.n_shot <= 0:
        raise ValueError("n_shot must be greater than 0")
    
    samples, in_context_samples = [], []
    if args.data_mode == "grid":
        # For plotting mode, generate a grid dataset for testing
        samples = gen_grid_dataset(grid_size = int(args.num_samples ** 0.5))
        TEST_SIZE = len(samples)
        TRAIN_SIZE = 0
        
        # Generate a small set of in-context examples
        in_context_samples = gen_dataset(
            num_samples=args.n_shot,
            cluster_std=args.noise_level,
            seed_value=34
        )
        
    elif args.data_mode == "default":
        # Normal mode - generate regular training and test sets
        TEST_SIZE = int(args.num_samples * args.test_ratio)
        TRAIN_SIZE = args.num_samples - TEST_SIZE
        
        # Generate synthetic dataset
        samples = gen_dataset(
            num_samples=args.num_samples,
            cluster_std=args.noise_level,
            random = args.random,
            seed_value=12
        )
        
        in_context_samples = gen_dataset(
            num_samples=args.num_samples,
            cluster_std=args.noise_level,
            label_flip_rate=args.label_flip_rate,
            random = args.random,
            seed_value=34
        )
        
    elif args.data_mode == "mixed":
        # Mixed mode - generate a small set of in-context examples and a large set of test samples
        TEST_SIZE = int(args.num_samples * args.test_ratio)
        TRAIN_SIZE = args.num_samples - TEST_SIZE
        
        for i in range(args.num_samples):
            data = gen_dataset(
                num_samples=args.n_shot+1,
                cluster_std=args.noise_level,
                label_flip_rate=args.label_flip_rate,
                random = True,
                seed_value=i
            )
            samples.append(data[0])
            in_context_samples.append(data[1:])
    else:
        raise ValueError(f"Invalid data mode: {args.data_mode}")
    
    dataset_dict = {
        'features': [sample[0] for sample in samples],
        'label': [sample[1] for sample in samples]
    }
    
    pdb.set_trace()
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
        TEST_SIZE,
        plot=args.plot
    )


def blobs_reward_fn(solution_str, ground_truth):
    return classification_reward_fn(solution_str, ground_truth)
