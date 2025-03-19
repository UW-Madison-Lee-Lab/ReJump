"""
Preprocess dataset for moons task - a synthetic classification task with two interleaving half circles
"""

import numpy as np
from sklearn.datasets import make_moons
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, classification_reward_fn

def gen_dataset(
    num_samples: int,
    noise: float = 0.1,
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate synthetic moons dataset for classification task.
    
    Args:
        num_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (features, label)
    """
    np.random.seed(seed_value)
    
    # Generate synthetic moons data
    X, y = make_moons(
        n_samples=num_samples,
        noise=noise,
        random_state=seed_value
    )
    
    samples = []
    for i in tqdm(range(num_samples)):
        features = X[i].tolist()
        label = int(y[i])
        samples.append((features, label))
    
    return samples



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()
    set_seed(42)
    data_source = 'moons'
    TEST_SIZE = int(args.num_samples * args.test_ratio)
    TRAIN_SIZE = args.num_samples - TEST_SIZE
    n_classes = 2  # Moons dataset always has 2 classes
    
    # Generate synthetic dataset
    samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise,
        seed_value=42
    )
    
    in_context_samples = gen_dataset(
        num_samples=args.num_samples,
        noise=args.noise,
        seed_value=42
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
    
def moons_reward_fn(solution_str, ground_truth):
    return classification_reward_fn(solution_str, ground_truth)
