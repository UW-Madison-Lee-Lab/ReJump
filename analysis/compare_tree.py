
from analysis.tree_vis_math import parse_json
import zss
import argparse


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
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    flow_dict1 = parse_json(args.file1)