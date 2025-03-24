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
def gen_dataset(
    num_samples: int,
    cluster_std: float = 1.0,
    seed_value: int = 42,
    label_flip_rate: float = 0.0,
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
        cluster_std=cluster_std,
        random_state=seed_value,
    )
    
    y = flip_label(y, label_flip_rate, 3)
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        label = int(y[i])
        samples.append((features, label))
    
    pdb.set_trace()
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--noise_level', type=float, default=1.0)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--label_flip_rate', type=float, default=0.0)
    args = parser.parse_args()
    set_seed(42)
    
    data_source = 'blobs'
    TEST_SIZE = int(args.num_samples * args.test_ratio)
    TRAIN_SIZE = args.num_samples - TEST_SIZE
    
    # Generate synthetic dataset
    samples = gen_dataset(
        num_samples=args.num_samples,
        cluster_std=args.noise_level,
        seed_value=12
    )
    
    in_context_samples = gen_dataset(
        num_samples=args.num_samples,
        cluster_std=args.noise_level,
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
        3,
        TRAIN_SIZE,
        TEST_SIZE,
    )
    


def blobs_reward_fn(solution_str, ground_truth):
    return classification_reward_fn(solution_str, ground_truth)
