from TTT.tree_vis_math_v3 import parse_json, get_result_dir
import zss
import argparse
from constants import supported_datasets, supported_llms
import itertools
import pandas as pd
from tqdm import tqdm
import numpy as np
import pdb
from utils import load_json, wandb_init
import wandb
from environment import WANDB_INFO
from utils import set_seed
import signal

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
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Tree edit distance computation timed out")
    
    # Set the signal handler and a 20-second alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2)
    
    try:
        distance = zss.simple_distance(
            tree1_root,
            tree2_root,
            get_children=SimpleNode.get_children,
            get_label=SimpleNode.get_description,
            label_dist=get_distance,
            # No cost functions passed here; simple_distance uses defaults:
            # insert_cost=1, remove_cost=1, update_cost=1 if labels differ, 0 otherwise.
        )
        # Cancel the alarm if computation completes
        signal.alarm(0)
    except TimeoutError:
        # Return None if timeout occurs
        print(f"Debug: TimeoutError occurred during tree edit distance computation.")
        distance = None
        signal.alarm(0)
    return distance


def build_tree_from_flat_format(json_data):
    """
    Builds a tree of SimpleNode objects from the flat JSON format.

    Args:
        json_data (dict): The loaded JSON data (e.g., content of 0.json).

    Returns:
        SimpleNode: The root node of the tree, or None if a root cannot be determined.
    """
    if "tree" not in json_data or not isinstance(json_data["tree"], dict):
        print("Error: JSON data does not contain a 'tree' dictionary.")
        return None

    tree_nodes_data = json_data["tree"]
    nodes_map = {}  # To store SimpleNode instances, mapping node_id to SimpleNode object

    # First pass: create all nodes
    for node_id, node_data in tree_nodes_data.items():
        if not isinstance(node_data, dict) or "Problem" not in node_data:
            print(f"Warning: Skipping node '{node_id}' due to missing 'Problem' field or incorrect format.")
            continue
        # Use node_id as label, and "Problem" field as description
        description = node_data.get("Problem", "") 
        nodes_map[node_id] = SimpleNode(label=node_id, description=description)

    # Second pass: connect nodes and identify the root
    actual_root = None
    
    processed_nodes_for_parenting = set()

    for node_id, node_data in tree_nodes_data.items():
        if node_id not in nodes_map: # Node might have been skipped or is malformed
            continue
        
        current_node_obj = nodes_map[node_id]
        parent_id = node_data.get("parent")

        if parent_id == "none":
            if actual_root is not None and actual_root.label != node_id : # Check if a different root was already set
                print(f"Warning: Multiple roots defined with 'parent: none'. Previous: '{actual_root.label}', New: '{node_id}'. Using the first one encountered: '{actual_root.label}'.")
            elif actual_root is None:
                 actual_root = current_node_obj
        elif parent_id and parent_id in nodes_map:
            parent_node_obj = nodes_map[parent_id]
            # Avoid adding the same child multiple times if source data is redundant
            if current_node_obj not in parent_node_obj.children:
                 parent_node_obj.children.append(current_node_obj)
        elif parent_id: # Parent ID exists but parent node is not in nodes_map (e.g., skipped or malformed)
            print(f"Warning: Parent ID '{parent_id}' for node '{node_id}' not found in valid nodes. Node '{node_id}' will be an orphan unless it's the root.")
            # This node could still be the 'actual_root' if it's marked 'parent: none'
            if parent_id == "none" and actual_root is None: # Redundant check, but for safety
                actual_root = current_node_obj

        processed_nodes_for_parenting.add(node_id)


    if actual_root is None:
        # Try to find a root among nodes not parented, if no explicit "none" parent was found
        all_nodes_with_children = set()
        for node_id_iter in nodes_map:
            for child_node in nodes_map[node_id_iter].children:
                all_nodes_with_children.add(child_node.label)
        
        potential_roots = [nodes_map[nid] for nid in nodes_map if nid not in all_nodes_with_children]

        if len(potential_roots) == 1:
            actual_root = potential_roots[0]
            print(f"Info: No explicit root (parent: 'none') found. Using the single unparented node '{actual_root.label}' as root.")
        elif len(potential_roots) > 1:
            print(f"Error: No explicit root (parent: 'none') and multiple unparented nodes found: {[r.label for r in potential_roots]}. Cannot determine a unique root.")
            return None
        elif not nodes_map:
             print("Error: No nodes were parsed from the tree data.")
             return None
        else: # No explicit root, no unparented nodes (e.g. cycle, or all nodes malformed for parenting)
            print("Error: No explicit root (parent: 'none') and no clear unparented candidate node found.")
            # Fallback or error: if nodes_map is not empty, could pick the first node as a desperate measure.
            if nodes_map:
                 # first_node_key = list(nodes_map.keys())[0]
                 # actual_root = nodes_map[first_node_key]
                 # print(f"Warning: Desperate fallback - using node '{actual_root.label}' as root.")
                 return None # Safer to return None
            return None


    # Sanity check for unique children
    for node_id_iter in nodes_map:
        node = nodes_map[node_id_iter]
        if len(node.children) != len(set(node.children)):
            unique_children = []
            seen_children_labels = set()
            for child in node.children:
                if child.label not in seen_children_labels:
                    unique_children.append(child)
                    seen_children_labels.add(child.label)
            if len(node.children) != len(unique_children):
                # print(f"Warning: Node '{node.label}' had duplicate children objects pointing to the same labels. Corrected.")
                node.children = unique_children
                
    return actual_root


def get_root_node(json_data): # Changed flow_dict to json_data
    root_node = build_tree_from_flat_format(json_data)
    return root_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, nargs='+', default=["deepseek-ai/deepseek-reasoner", "xai/grok-3-mini-beta", "alibaba/qwen-turbo-2025-04-28-thinking"])
    parser.add_argument("--model", type=str, nargs='+', default=["deepseek-ai/deepseek-reasoner", "xai/grok-3-mini-beta"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["math500"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--idx", type=int, nargs='+', default=None, help="List of sample indices to process")
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
                template_type = supported_llms[model]["template_type"],
                response_length = 404,
                num_samples = -1,
                feature_noise = supported_datasets[dataset]["feature_noise"],
                label_noise = 0.0,
                data_mode = "default",
                n_query = 1,
                temperature = 0.00,
            )
            
    model_pairs = list(itertools.combinations(models, 2))
    
    for model_pair in model_pairs:
        for dataset in datasets:
            if args.wandb:
                model1, model2 = sorted(model_pair)
                wandb_config = {
                    "model1": model1,
                    "model2": model2,
                    "dataset": dataset,
                }
                if not wandb_init(f"{WANDB_INFO['project']}-tree-compare", WANDB_INFO["entity"], wandb_config):
                    continue
                
            set_seed(234)
            
            results_dir1 = results_dirs[(model_pair[0], dataset)]
            results_dir2 = results_dirs[(model_pair[1], dataset)]
            
            n_samples = len(pd.read_parquet(f"{results_dir1}/test_default.parquet"))
            
            distances = []
            
            print("--------------------------------"*2)
            print(f"|{model_pair[0]}|{model_pair[1]}|{dataset}|")
            print("--------------------------------"*2)
            
            sample_indices = args.idx if args.idx is not None else range(n_samples)
            
            pbar = tqdm(sample_indices)
            for i in pbar:
                flow_dict1 = load_json(f"{results_dir1}/tree_vis_v3/{i}.json")
                flow_dict2 = load_json(f"{results_dir2}/tree_vis_v3/{i}.json")
                
                if flow_dict1 is None or flow_dict2 is None:
                    print(f"Debug: Failed to load JSON for sample {i}. Skipping.")
                    distance = None # Ensure distance is None so it's skipped
                    continue

                tree1, tree2 = None, None # Initialize
                try:
                    tree1 = get_root_node(flow_dict1)
                    tree2 = get_root_node(flow_dict2)

                    if tree1 is None or tree2 is None:
                        print(f"Debug: Failed to build one or both trees for sample {i}. Tree1: {'OK' if tree1 else 'Failed'}, Tree2: {'OK' if tree2 else 'Failed'}. Skipping distance computation.")
                        distance = None
                    else:
                        distance = compute_tree_edit_distance(tree1, tree2)
                
                except Exception as e: # Catch any other unexpected error during tree processing or distance calc
                    print(f"Debug: An unexpected error occurred for sample {i} while building trees or computing distance. Error: {e}")
                    import traceback
                    traceback.print_exc()
                    distance = None
                
                if distance is None:
                    print(f"Debug: Distance is None for sample {i}. Skipping.")
                    continue
                
                print(f"Sample {i}: Distance = {distance}") # Print individual distance
                distances.append(distance)
                # Calculate running average and update tqdm postfix

                current_avg_distance = np.mean(distances)
                pbar.set_description(f'Avg Dist: {current_avg_distance:.4f}')
                
            print(f"Average distance between {model_pair[0]} and {model_pair[1]} on {dataset}: {np.mean(distances):.2f}")
    
            if args.wandb:
                wandb.log({"distance": np.mean(distances)})
                wandb.finish()
    