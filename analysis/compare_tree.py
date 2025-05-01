
from analysis.tree_vis_math import parse_json, get_result_dir
import zss
import argparse
from constants import supported_datasets
import itertools
import pandas as pd
from tqdm import tqdm
import numpy as np
import pdb
from utils import load_json
import wandb
from environment import WANDB_INFO

def get_compare_prompt(str1, str2):
    return f"""
Do the following two descriptions refer to the same concept or task? Answer only with 'Yes' or 'No'.

Description 1: "{str1}"
Description 2: "{str2}"
"""

def get_distance(str1, str2):
    if str1 == "" or str2 == "": return 1
    return 0


# Define a simple Node class compatible with the zss library
# Each node needs a label and a way to access its children
class SimpleNode:
    """
    A simple node structure for representing trees.
    Required by the zss library.
    """
    def __init__(self, label, children=None, description=""):
        self.label = label
        self.children = children if children is not None else []
        self.description = description

    # --- Methods required by the zss library ---

    @staticmethod
    def get_children(node):
        """Returns the list of children of a node."""
        return node.children

    @staticmethod
    def get_label(node):
        """Returns the label of a node."""
        return node.label
    
    @staticmethod
    def get_description(node):
        """Returns the description of a node."""
        return node.description

    # --- Helper method for printing the tree (optional) ---
    def __str__(self):
        """Simple string representation for debugging."""
        return f"Node({self.label})"

    @classmethod
    def print_tree(cls, node, indent="", last=True):
        """Prints the tree structure."""
        print(indent, "+- " if last else "|- ", node, sep="")
        indent += "   " if last else "|  "
        child_count = len(node.children)
        print(node.children)
        for i, child in enumerate(node.children):
            print(i, child_count-1)
            cls.print_tree(child, indent, i == child_count - 1)


# Function to compute the tree edit distance using zss.simple_distance
def compute_tree_edit_distance(tree1_root, tree2_root):
    """
    Computes the tree edit distance between two trees using the zss library,
    which implements the Zhang-Shasha algorithm.

    Args:
        tree1_root (SimpleNode): The root node of the first tree.
        tree2_root (SimpleNode): The root node of the second tree.

    Returns:
        int: The tree edit distance based on the default costs (insert=1, delete=1, update=1 if labels differ).
             This represents the minimum number of edit operations (insert, delete, update)
             to transform tree1 into tree2.
    """
    # The zss.simple_distance function requires the root nodes and access methods.
    # It uses default costs and does not accept insert_cost, remove_cost, or update_cost keywords.
    # We provide our SimpleNode methods directly.
    distance = zss.simple_distance(
        tree1_root,
        tree2_root,
        get_children=SimpleNode.get_children,
        get_label=SimpleNode.get_description,
        label_dist=get_distance,
        # No cost functions passed here; simple_distance uses defaults:
        # insert_cost=1, remove_cost=1, update_cost=1 if labels differ, 0 otherwise.
    )
    return distance


def build_tree_recursive(node_label, flow_dict, node_map):
    """
    Recursively builds a tree of SimpleNode objects from a flow dictionary.

    Args:
        node_label (str): The label of the current node to build.
        flow_dict (dict): The dictionary defining parent-child relationships.
                          e.g., {'parent': ['child1', 'child2']}
        node_map (dict): A dictionary to cache created nodes by label.

    Returns:
        SimpleNode: The root node of the subtree starting at node_label.
    """
    
    logical_flow = flow_dict["visualization"]["logical_flow"]
    clustering_results = flow_dict["visualization"]["clustering_results"]
    
    
    # If node already exists in our map, return the cached version
    if node_label in node_map:
        return node_map[node_label]

    # Create the node for the current label
    current_node = SimpleNode(node_label)
    node_map[node_label] = current_node # Add to cache before processing children
    if node_label in clustering_results:
        current_node.description = clustering_results[node_label]["description"]
    else:
        current_node.description = current_node.label

    if node_label == "solution":
        return current_node
        
    children_labels = logical_flow[node_label]

    # Recursively build each child and add it to the current node's children
    for child_label in children_labels:
        child_node = build_tree_recursive(child_label, flow_dict, node_map)
        current_node.children.append(child_node)

    return current_node

def get_root_node(flow_dict):
    root_node = build_tree_recursive("root", flow_dict, {})
    return root_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs='+', default=["deepseek-ai/deepseek-reasoner", "xai/grok-3-mini-beta", "alibaba/qwen-turbo-2025-04-28-thinking"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["math500", "gpqa-diamond"])
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    models = args.model
    datasets = args.dataset
    
    results_dirs = {}
    
    for model in models:
        for dataset in datasets:
            results_dirs[(model, dataset)] = get_result_dir(
                dataset_name = dataset,
                model_name = model,
                shot = 0,
                template_type = "reasoning_api",
                response_length = 404,
                num_samples = -1,
                feature_noise = supported_datasets[dataset]["feature_noise"],
                label_noise = 0.0,
                data_mode = "default",
                n_query = 1,
            )
            
    model_pairs = list(itertools.combinations(models, 2))
    
    for model_pair in model_pairs:
        for dataset in datasets:
            if args.wandb:
                wandb.init(
                    project=f"{WANDB_INFO['project']}-tree-compare",
                    entity=WANDB_INFO["entity"],
                    config={
                        "model_pair": sorted(model_pair),
                        "dataset": dataset,
                    }
                )
            
            results_dir1 = results_dirs[(model_pair[0], dataset)]
            results_dir2 = results_dirs[(model_pair[1], dataset)]
            
            n_samples = len(pd.read_parquet(f"{results_dir1}/test_default.parquet"))
            
            distances = []
            
            print("--------------------------------"*2)
            print(f"|{model_pair[0]}|{model_pair[1]}|{dataset}|")
            print("--------------------------------"*2)
            
            pbar = tqdm(range(n_samples))
            for i in pbar:
                flow_dict1 = load_json(f"{results_dir1}/tree_vis/{i}.json")
                flow_dict2 = load_json(f"{results_dir2}/tree_vis/{i}.json")
                tree1 = get_root_node(flow_dict1)
                tree2 = get_root_node(flow_dict2)
                distance = compute_tree_edit_distance(tree1, tree2)
                distances.append(distance)
                # Calculate running average and update tqdm postfix

                current_avg_distance = np.mean(distances)
                pbar.set_description(f'Avg Dist: {current_avg_distance:.4f}')
                
            print(f"Average distance between {model_pair[0]} and {model_pair[1]} on {dataset}: {np.mean(distances)}")
    
            if args.wandb:
                wandb.log({"distance": np.mean(distances)})
                wandb.finish()
    