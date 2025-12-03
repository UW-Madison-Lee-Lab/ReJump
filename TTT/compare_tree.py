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
from scipy.spatial.distance import jensenshannon

# Module-level constants for action flow similarity
_ACTION_TYPES = ['calculation/derivation', 'verification', 'backtracking']
_ACTION_MAP = {action: i for i, action in enumerate(_ACTION_TYPES)}
_NUM_ACTIONS = len(_ACTION_TYPES)
# Small constant for smoothing probabilities to avoid zeros and ensure valid distributions
_SMOOTHING_ALPHA = 1e-9 


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


def count_nodes(node: SimpleNode) -> int:
    
    if node is None:
        return 0
    
    count = 1  # Count the current node
    for child in SimpleNode.get_children(node):
        count += count_nodes(child)
    return count


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

def compute_tree_similarity(tree1_root, tree2_root):
    try:
        return 1 - compute_tree_edit_distance(tree1_root, tree2_root)/max(count_nodes(tree1_root), count_nodes(tree2_root))
    except: 
        return 0

def _get_transition_matrix(walk):
    """
    Builds a 3x3 transition-probability matrix from a walk.
    P_ab is the empirical probability of moving from action type a to b.
    Uses smoothing to ensure all probabilities are > 0 and rows sum to 1.
    """
    counts = np.zeros((_NUM_ACTIONS, _NUM_ACTIONS))

    # Ensure walk is a list-like sequence of actions and has at least one transition
    if walk and hasattr(walk, '__iter__') and not isinstance(walk, str) and len(walk) >= 2:
        for i in range(len(walk) - 1):
            action_from_str = walk[i]
            action_to_str = walk[i+1]
            
            # Only consider transitions between known action types
            if action_from_str["category"] in _ACTION_MAP and action_to_str["category"] in _ACTION_MAP:
                idx_from = _ACTION_MAP[action_from_str["category"]]
                idx_to = _ACTION_MAP[action_to_str["category"]]
                counts[idx_from, idx_to] += 1

    prob_matrix = np.zeros((_NUM_ACTIONS, _NUM_ACTIONS))
    for i in range(_NUM_ACTIONS):
        row_sum = np.sum(counts[i, :])
        # Denominator is guaranteed positive if _SMOOTHING_ALPHA > 0
        denominator = row_sum + _NUM_ACTIONS * _SMOOTHING_ALPHA
        prob_matrix[i, :] = (counts[i, :] + _SMOOTHING_ALPHA) / denominator
            
    return prob_matrix

def compute_walk_similarity(walk1, walk2):
    """
    Computes the Action-Flow Similarity between two walks (sequences of actions).
    Similarity = 1 - JS(P1 || P2), where P1 and P2 are transition probability matrices,
    and JS is the Jensen-Shannon divergence, averaged over states.
    The JSD is calculated using log base 2, so its value is in [0,1].
    The final similarity score is also in [0,1].
    """
    P1 = _get_transition_matrix(walk1)
    P2 = _get_transition_matrix(walk2)

    js_divergences_per_state = []
    for i in range(_NUM_ACTIONS):
        p_s = P1[i, :]
        q_s = P2[i, :]
        
        # scipy.spatial.distance.jensenshannon returns sqrt(JSD).
        # We use base=2.0 for JSD to be in [0,1].
        jsd_s_sqrt = jensenshannon(p_s, q_s, base=2.0)
        
        # With smoothing in _get_transition_matrix, p_s and q_s should always be
        # valid probability distributions, so jsd_s_sqrt should not be NaN.
        # This check is a safeguard.
        if np.isnan(jsd_s_sqrt):
            jsd_s = 1.0  # Assign maximal divergence if NaN occurs
        else:
            jsd_s = jsd_s_sqrt**2  # Square to get JSD in [0,1]
        
        js_divergences_per_state.append(jsd_s)

    # _NUM_ACTIONS is expected to be > 0 (it's 3).
    # If js_divergences_per_state were empty, mean would result in NaN or error.
    if not js_divergences_per_state: # Should not be hit for _NUM_ACTIONS = 3
        # This case implies _NUM_ACTIONS was 0, which is an invalid setup.
        # Return lowest similarity or handle as error.
        return 0.0 

    mean_jsd = np.mean(js_divergences_per_state) # mean_jsd will be in [0,1]
    
    similarity = 1.0 - mean_jsd  # similarity will be in [0,1]
    
    return similarity

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
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0])
    parser.add_argument("--mode", type=str, nargs='+', default=["default"])
    parser.add_argument("--dataset", type=str, nargs='+', default=["math500", "game24"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--idx", type=int, nargs='+', default=None, help="List of sample indices to process")
    parser.add_argument("--analysis_model", type=str, nargs='+', default=["google/gemini-2.5-pro-preview-03-25"])
    args = parser.parse_args()
    
    models = args.model
    datasets = args.dataset
    modes = args.mode
    analysis_model = args.analysis_model
    results_dirs = {}
    
    if len(args.temperature) == 1:
        temperatures = [args.temperature[0] for _ in models]
    else:
        if len(args.temperature) != len(models):
            raise ValueError(f"Number of temperatures ({len(args.temperature)}) must match number of models ({len(models)})")
        temperatures = args.temperature
        
    if len(args.mode) == 1:
        modes = [args.mode[0] for _ in models]
    else:
        if len(args.mode) != len(models):
            raise ValueError(f"Number of modes ({len(args.mode)}) must match number of models ({len(models)})")
        modes = args.mode
        
    if len(args.analysis_model) == 1:
        analysis_models = [args.analysis_model[0] for _ in models]
    else:
        if len(args.analysis_model) != len(models):
            raise ValueError(f"Number of analysis models ({len(args.analysis_model)}) must match number of models ({len(models)})")
        analysis_models = args.analysis_model
    
    for model, temperature, mode, analysis_model in zip(models, temperatures, modes, analysis_models):
        for dataset in datasets:
            if mode == "default":
                template_type = supported_llms[model]["template_type"] 
            else:
                template_type = supported_llms[model]["template_type"] + "_" + mode
                
            results_dirs[(model, dataset, temperature, mode, analysis_model)] = get_result_dir(
                dataset_name = dataset,
                model_name = model,
                shot = 0,
                template_type = template_type,
                response_length = 404,
                num_samples = -1 if dataset == "math500" else 100,
                feature_noise = supported_datasets[dataset]["feature_noise"],
                label_noise = 0.0,
                data_mode = "default",
                n_query = 1,
                temperature = temperature,
                replicate_id = 0,
            )
            
    model_indices = list(range(len(models)))
    model_index_pairs = list(itertools.combinations(model_indices, 2))
    
    for model_index_pair in model_index_pairs:
        for dataset in datasets:
            if args.wandb:
                idx1, idx2 = sorted(model_index_pair, key=lambda x: models[x])
                model1, model2 = models[idx1], models[idx2]
                temperature1, temperature2 = temperatures[idx1], temperatures[idx2]
                mode1, mode2 = modes[idx1], modes[idx2]
                analysis_model1, analysis_model2 = analysis_models[idx1], analysis_models[idx2]
                wandb_config = {
                    "model1": model1,
                    "model2": model2,
                    "dataset": dataset,
                    "temperature1": temperature1,
                    "temperature2": temperature2,
                    "mode1": mode1,
                    "mode2": mode2,
                    "analysis_model1": analysis_model1,
                    "analysis_model2": analysis_model2,
                }
                if not wandb_init(f"{WANDB_INFO['project']}-tree-compare", WANDB_INFO["entity"], wandb_config):
                    continue
                
            set_seed(234)
            
            results_dir1 = results_dirs[(model1, dataset, temperature1, mode1, analysis_model1)]
            results_dir2 = results_dirs[(model2, dataset, temperature2, mode2, analysis_model2)]
            
            n_samples = len(pd.read_parquet(f"{results_dir1}/test_default.parquet"))
            
            tree_similarities = []
            walk_similarities = []
            
            print("--------------------------------"*2)
            print(f"|{model1}|{model2}|{dataset}|{analysis_model1}|{analysis_model2}|")
            print("--------------------------------"*2)
            
            sample_indices = args.idx if args.idx is not None else range(n_samples)
            
            pbar = tqdm(sample_indices)
            for i in pbar:
                flow_dict1 = load_json(f"{results_dir1}/tree_vis_{analysis_model1}/{i}.json")
                flow_dict2 = load_json(f"{results_dir2}/tree_vis_{analysis_model2}/{i}.json")
                
                if flow_dict1 is None or flow_dict2 is None:
                    print(f"Debug: Failed to load JSON for sample {i}. Skipping.")
                    tree_similarity = None # Ensure distance is None so it's skipped
                    continue

                tree1, tree2 = None, None # Initialize
                tree1 = get_root_node(flow_dict1)
                tree2 = get_root_node(flow_dict2)
                walk1, walk2 = flow_dict1["walk"], flow_dict2["walk"]

                if tree1 is None or tree2 is None:
                    print(f"Debug: Failed to build one or both trees for sample {i}. Tree1: {'OK' if tree1 else 'Failed'}, Tree2: {'OK' if tree2 else 'Failed'}. Skipping distance computation.")
                    tree_similarity = None
                    walk_similarity = None
                else:
                    tree_similarity = compute_tree_similarity(tree1, tree2)
                    walk_similarity = compute_walk_similarity(walk1, walk2)

                
                if tree_similarity is None or walk_similarity is None:
                    print(f"Debug: Distance is None for sample {i}. Skipping.")
                    continue
                
                print(f"Sample {i}: Tree Similarity = {tree_similarity}, Walk Similarity = {walk_similarity}") # Print individual distance
                tree_similarities.append(tree_similarity)
                walk_similarities.append(walk_similarity)
                # Calculate running average and update tqdm postfix

                current_avg_tree_similarity = np.mean(tree_similarities)
                current_avg_walk_similarity = np.mean(walk_similarities)
                pbar.set_description(f'Avg Tree Similarity: {current_avg_tree_similarity:.4f}, Avg Walk Similarity: {current_avg_walk_similarity:.4f}')
                
            print(f"Average tree similarity between {model1}({temperature1}) and {model2}({temperature2}) on {dataset}: {np.mean(tree_similarities):.2f}")
            print(f"Average walk similarity between {model1}({temperature1}) and {model2}({temperature2}) on {dataset}: {np.mean(walk_similarities):.2f}")
    
            if args.wandb:
                wandb.log({
                    "tree_similarity": np.mean(tree_similarities), 
                    "tree_similarity_stderr": np.std(tree_similarities) / np.sqrt(len(tree_similarities)),
                    "walk_similarity": np.mean(walk_similarities),
                    "walk_similarity_stderr": np.std(walk_similarities) / np.sqrt(len(walk_similarities)),
                })
                wandb.finish()
    