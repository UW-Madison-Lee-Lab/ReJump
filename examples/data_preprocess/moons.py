"""
Preprocess dataset for moons task - a synthetic classification task with two interleaving half circles
"""

import numpy as np
from sklearn.datasets import make_moons
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import set_seed
from examples.data_preprocess.helper import save_data, flip_label, prepare_dataset

def gen_dataset(
    num_samples: int,
    feature_noise: float = 0.1,
    seed_value: int = 42,
    label_noise: float = 0.0,
    random: bool = False,
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
        noise=feature_noise,
        random_state=seed_value
    )
    
    if random:
        # Randomly shift the data points
        shift_x1 = np.random.uniform(-2, 2)
        shift_x2 = np.random.uniform(-2, 2)
        X[:, 0] += shift_x1
        X[:, 1] += shift_x2
    
    y = flip_label(y, label_noise, 2)
    
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
    parser.add_argument('--feature_noise', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--n_shot', type=int, default=10)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--label_noise', type=float, default=0.0)
    parser.add_argument('--data_mode', type=str, default="default", choices=["default", "grid", "mixed"])
    args = parser.parse_args()
    set_seed(42)
    
    data_source = 'moons'

    n_classes = 2  # Moons dataset always has 2 classes
    
    
    datasets = prepare_dataset(args, gen_dataset)
    
    save_data(
        datasets['dataset_dict'],
        datasets['in_context_dataset_dict'],
        data_source,
        args,
        n_classes,
        datasets['TRAIN_SIZE'],
        datasets['TEST_SIZE'],
        data_mode = args.data_mode,
    )
    
