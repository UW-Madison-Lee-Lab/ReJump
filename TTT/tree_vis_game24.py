import json
import re
import graphviz
import textwrap
import argparse
import pandas as pd
import os
from tqdm import tqdm
from constants import get_result_dir, supported_datasets
from utils import save_json, load_json, wandb_init
import pdb
import random

from verl.utils.llm_api import LLMAPI
from constants import supported_llms
import wandb
from environment import WANDB_INFO, root_dir

import numpy as np
from collections import deque
# Note: The following import might need to be moved to the top of the file
from sklearn.model_selection import train_test_split
from collections import Counter
import xgboost as xgb
# compare_answer will be imported dynamically based on dataset_name
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

def get_tree_prompt_sudoku(input_str, output_str):
    return f"""
Given the problem statement and reasoning process below. Your task is to analyze a detailed thinking process for solving a Sudoku puzzle (provided below) and convert it into a reasoning tree. **Do not try to solve the problem yourself, fully use the given reasoning process and just convert it!**

---
**BEGIN ORIGINAL PROBLEM STATEMENT**
---
{input_str}
---
**END ORIGINAL PROBLEM STATEMENT**
---

---
**BEGIN INPUT REASONING PROCESS**
---
{output_str}
---
**END INPUT REASONING PROCESS**
---

Here are some instructions:

**Node Object Structure:**

Each node object must contain: `Problem`, `parent`, `Result`.

1. **`Problem` (String):** A description of the current state of the Sudoku grid or the specific cell(s) being filled.

* **`node1` (Root):** Must describe the initial Sudoku puzzle state with empty cells marked as 0. For example, "Initial grid: 0204\\n3401\\n0012\\n2043"

* **Non-leaf Nodes:** Each node describes a partial solution being explored. For example, "Filled row 2, column 3 with 2" or "Considering values for row 1, column 1". Give all these nodes index numbers to keep tracking (after node1).

* **Leaf node:** **This node represents a complete or attempted complete solution of the Sudoku grid.** For example, "Complete grid: 1234\\n3421\\n4312\\n2143". Use an index number for each one (after node1).

Pay attention that the problem statement of each node should be unique. If two nodes have the same description (i.e., the same partial grid state), merge them into one.

2. **`parent` (String):**

* **`node1` (root):** Must be None.

* **Other nodes:** Must be the previous partial solution that the current node builds on. Use the index number to indicate the index of its parent node.

3. **`Result` (String):**

* **`root`:** None.

* **Intermediate Nodes:** None.

* **Leaf node:** Must be the **complete solution grid**. For example, "1234\\n3421\\n4312\\n2143".

Please generate a single JSON output. This output must be a **single JSON object** where keys are unique node IDs (e.g., "node1", "node2", corresponding to the index numbers assigned to track the nodes) and values are the node objects (containing 'Problem', 'parent', 'Result') as detailed above.

    """


def get_tree_prompt(input_str, output_str):
    return f"""
Given the problem statement and reasoning process below. Your task is to analyze a detailed thinking process for solving a math problem (provided below) and convert it into a reasoning tree. **Do not try to solve the problem yourself, fully use the given reasoning process and just convert it!**

---
**BEGIN ORIGINAL PROBLEM STATEMENT**
---
{input_str}
---
**END ORIGINAL PROBLEM STATEMENT**
---

---
**BEGIN INPUT REASONING PROCESS**
---
{output_str}
---
**END INPUT REASONING PROCESS**
---

Here are some instructions:

**Node Object Structure:**

Each node object must contain: `Problem`, `parent`, `Result`.

1. **`Problem` (String): A partial solution containing the four numbers and any calculation has been tried. Only use numbers, + - * / and parentheses.

* **`node1` (Root):** Must be exactly the four initial numbers in the problem. For example, "9,3,12,8".

* **Non-leaf Nodes:** Each node describes the partial solution being explored. For example, for problem 9,3,12,8, an intermediate node "9-3, 12, 8" means that we have tried (9-3), and need to try 2 more calculations with numbers 12 and 8 to get 24. Give all these nodes indexes number to keep tracking (after node1).

* **Leaf node:** **This node represents the very last calculation step that produces the final answer after three calculation steps.** For example, for problem 9,3,12,8, this could be "9-3+128", which is a leaf node that is unsuccessful. Another successful leaf node could be "(9-3)*(128)". Also use an index number for each one (after node1).

Pay attention that the problem statement of each node should be unique. If two nodes have the same description (i.e., the same partial calculation and the numbers not calculated so far), merge them into one.

2. **`parent` (String):

* **`node1` (root):** Must be None.

* **Other nodes:** Must be the previous partial solution that the current node builds on. For example, the parent of the node "9-3, 12, 8" is "9,3,12,8". But here just use the index number to indicate the index of its parent node.

3. **`Result` (String):

* **`root`:** None.

* **Intermediate Nodes:** None.

* **Leaf node** Must be the **final answer**. For example, the result of node "9-3+12-8" is 10. Written in latex.

Please generate a single JSON output. This output must be a **single JSON object** where keys are unique node IDs (e.g., "node1", "node2", corresponding to the index numbers assigned to track the nodes) and values are the node objects (containing 'Problem', 'parent', 'Result') as detailed above.

    """
    
def get_walk_prompt_sudoku(input_str, output_str, tree_json):
    return f"""
You are an AI assistant specialized in analyzing Sudoku solving reasoning processes. Your task is to trace the provided reasoning text against a structured reasoning tree and generate a "walk" representing the trajectory of the thought process.

**Inputs:**

1.  **Problem Description:**
    ```
    {input_str}
    ```
2.  **Reasoning Text:** A step-by-step textual explanation of how the Sudoku puzzle was solved, including potential errors, corrections, explorations of different cell values, and verifications.
    ```text
    {output_str}
    ```
3.  **Reasoning Tree:** A JSON object representing the structured steps and dependencies of the solution(s). Each key is a node ID, and the value contains information about that step, including its parent node and specifically a "Problem" field describing the state or action at that node.
    ```json
    {tree_json}
    ```

**Task:**

Analyze the `Reasoning Text` to determine the sequence in which the solver mentally visited or considered the steps represented by the nodes in the `Reasoning Tree`. Identify the transitions between these nodes and categorize each transition.

**Output Format:**

Generate a JSON list of dictionaries, where each dictionary represents a single step in the reasoning walk. Each dictionary must have the following keys:

* `from`: The ID (string) of the node the reasoning is moving *from*.
* `to`: The ID (string) of the node the reasoning is moving *to*.
* `category`: A string indicating the type of transition. Must be one of:
    * `calculation/derivation`: Represents forward progress in the reasoning, filling in cells or making logical deductions.
    * `backtracking`: Represents realizing an error and returning to a previous state to try a different value.
    * `verification`: Represents checking or confirming filled cells by re-checking row/column/box constraints.

**Instructions:**

1.  Read the `Reasoning Text` carefully, paying attention to the flow, changes in direction, cell filling decisions, and verification steps.
2.  Map segments of the `Reasoning Text` to the corresponding nodes in the `Reasoning Tree`.
3.  Identify the sequence of nodes visited based on the flow of the `Reasoning Text`.
4.  For each transition, determine the appropriate `category` based on the definitions above.
5.  The walk should reflect the *actual* path taken in the `Reasoning Text`, including explorations of incorrect values and subsequent backtracking.
6.  Ensure the output is strictly the JSON list as specified, with no additional explanatory text.
7.  The output MUST be perfectly valid JSON, parseable by standard libraries.
8.  The walk must always start at node1: The first transition in your output should always be `"from": "node1"`, `"to": ...`.

**Final Output Request:**

Now, analyze the provided inputs and generate the reasoning walk as a JSON list. Output *only* the JSON list.
    """


def get_walk_prompt(input_str, output_str, tree_json):
    return f"""
You are an AI assistant specialized in analyzing mathematical reasoning processes. Your task is to trace the provided reasoning text against a structured reasoning tree and generate a "walk" representing the trajectory of the thought process.

**Inputs:**

1.  **Problem Description:**
    ```
    {input_str}
    ```
2.  **Reasoning Text:** A step-by-step textual explanation of how the problem was solved, including potential errors, corrections, explorations of different paths, and verifications.
    ```text
    {output_str}
    ```
3.  **Reasoning Tree:** A JSON object representing the structured steps and dependencies of the solution(s). Each key is a node ID, and the value contains information about that step, including its parent node and specifically a "Problem" field describing the task of that node.
    ```json
    {tree_json}
    ```

**Task:**

Analyze the `Reasoning Text` to determine the sequence in which the solver mentally visited or considered the steps represented by the nodes in the `Reasoning Tree`. Identify the transitions between these nodes and categorize each transition. **Crucially, for verification steps, visiting a node X implies the text shows evidence of re-doing the specific task described in the "Problem" field of node X.**

**Output Format:**

Generate a JSON list of dictionaries, where each dictionary represents a single step in the reasoning walk. Each dictionary must have the following keys:

* `from`: The ID (string) of the node the reasoning is moving *from*.
* `to`: The ID (string) of the node the reasoning is moving *to*.
* `category`: A string indicating the type of transition. Must be one of:
    * `calculation/derivation`: Represents forward progress in the reasoning, moving from one step to the next logical step (often parent to child in the tree) to derive new information or explore a solution path.
    * `backtracking`: Represents abandoning a current line of thought or calculation (often because it's incorrect, inefficient, or a dead end) and returning to a previous state (node) to try a different approach. This is typically a move from a node to one of its ancestors (not necessarily the direct parent).
    * `verification`: Represents checking or confirming a result or step **by re-doing the work associated with previous nodes**. This is determined based on the text:
        * **Specific Re-work:** If the text explicitly describes actions that precisely match the **problem description** defined within an intermediate node (e.g., node X) as part of checking a later result (node Z), trace the path reflecting that specific re-work (e.g., Z -> X -> Z). This requires clear evidence in the text of **re-solving the problem defined in node X**.
        * **General Check:** If the text indicates verification of a result (node Z) but ***does not*** show actions matching the specific **problem description** of any intermediate node, interpret this as checking consistency with the initial problem statement/conditions (node 1). Represent this path as Z -> 1 -> Z. ***Note: Simply using a formula or result from a previous node (e.g., node X) without showing the steps to re-solve the problem defined in node X does NOT count as re-doing the work of node X.***

**Instructions:**

1.  Read the `Reasoning Text` carefully, paying attention to the flow, changes in direction, calculations, statements of intent (e.g., "Let me try...", "No, that's wrong...", "Let me verify..."), and results.
2.  Map segments of the `Reasoning Text` to the corresponding nodes in the `Reasoning Tree`. Use the "Problem" and "Result" fields in the tree nodes to help with mapping *initial* derivations.
3.  Identify the sequence of nodes visited or considered based on the flow of the `Reasoning Text`.
4.  For each transition from one node (`from`) to the next (`to`) in the sequence, determine the appropriate `category` based the definitions above.
5.  Pay close attention to parts of the reasoning text that indicate:
    * Starting a calculation or derivation (maps to `calculation/derivation`).
    * Realizing an error or deciding a path is not fruitful and returning to an earlier idea (maps to `backtracking`).
    * Re-checking results (maps to `verification`). **When mapping `verification`:** First, check if the text describes actions that precisely match the **problem description** of an intermediate node (Node X), essentially re-doing the work defined in that node. If yes, trace the walk through the node being re-worked (e.g., Z -> X -> Z). If the text indicates verification but ***does not*** show such a specific re-work of a prior node's problem, assume it implies checking against the initial problem conditions (node 1) and represent the path as Z -> 1 -> Z. Remember: Simply *using* a result or formula from node X does not qualify as re-doing the problem of node X according to this definition.
6.  The walk should reflect the *actual* path taken in the `Reasoning Text`, including explorations of dead ends (like `node2` in the example) and subsequent backtracking.

    **Mandatory Backtracking Rule:**  
    Only when the reasoning process explicitly abandons or gives up on the current approach at node A and then starts a new, distinct attempt at node B must you include a backtracking transition from A to the parent of B, followed by a calculation/derivation transition from the parent of B to B. Never allow a direct calculation/derivation transition from A to B in these cases. Do not include backtracking transitions except in such abandonment cases.

7.  Ensure the output is strictly the JSON list as specified, with no additional explanatory text.
8. The output MUST be perfectly valid JSON, parseable by standard libraries.
9. The walk must always start at node1: The first transition in your output should always be `"from": "node1"`, `"to": ...`. Never use `"from": "none"`, `"from": null`, or any other alternative. Assume reasoning always conceptually begins at node1.

**Example Analysis (Based on Provided Inputs with Stricter Verification Logic):**

* Reasoning starts, defining the problem (maps to `node1`).
* Text explores calculating AB with specific points (maps to `node2`). `node1` -> `node2` (`calculation/derivation`).
* Text says "That seems messy... Let me think differently." and abandons the `node2` approach, returning to the setup phase (conceptually `node1`). `node2` -> `node1` (`backtracking`).
* Text introduces symmetry and points B(x,y), C(-x,y) (maps to `node3`). `node1` -> `node3` (`calculation/derivation`). This step involves *doing* the problem in `node3` (calculating distances).
* Text derives relationship between AB and BC, sets them equal (maps to `node4`). `node3` -> `node4` (`calculation/derivation`).
* Text solves for x and y using parabola equation (maps to `node5`). `node4` -> `node5` (`calculation/derivation`).
* Text calculates final side length (maps to `node6`). `node5` -> `node6` (`calculation/derivation`).
* Text says "Let me verify with the distance." It then shows:
    1.  `AB = sqrt(x^2 + y^2) = ...` This ***uses*** the formula derived in `node3` and values from `node5`. It does ***not*** show a re-derivation of the distance formula as described in `node3`'s problem ("Calculate the distances...").
    2.  `BC is 2x = ...` This ***uses*** the formula derived in `node3` and value from `node5`. It does ***not*** show a re-derivation.
* **Applying the strict verification rule:** Does the text show actions matching the *problem description* of an intermediate node (like re-deriving the formulas as defined in `node3`'s problem, or re-solving for x,y as defined in `node5`'s problem)? **No**, the text only shows the *application* of results from previous nodes.
* Therefore, according to the rule, since no specific re-work of a prior node's **problem** is detailed, we default to the **General Check** case. The path should be represented as checking the final result (`node6`) against the initial state (`node1`).
* The expected verification path for this text, under this strict interpretation, would be: `node6` -> `node1` (`verification`), potentially followed by `node1` -> `node6` (`verification`) or repeated. A simple `node6 -> node1 -> node6` sequence for the overall verification check is likely.

**Final Output Request:**

Now, analyze the provided inputs (`{{problem_description}}`, `{{reasoning_text}}`, `{{reasoning_tree_json}}`) using **this strict interpretation of verification** (visiting a node requires re-doing its specific "Problem") and generate the reasoning walk as a JSON list. Output *only* the JSON list.
    """


def parse_json(json_prompt):
    """Parse JSON content from a prompt string.
    
    Args:
        json_prompt: A string containing JSON data between ```json and ``` markers
        
    Returns:
        The parsed JSON data as a Python dictionary
    """
    # Find content between ```json and ``` markers
    json_match = re.search(r'```json\s*(.*?)\s*```', json_prompt, re.DOTALL)
    
    if not json_match:
        return {}
    
    json_content = json_match.group(1) 
    json_content = re.sub(
        r'(?<!\\)\\([a-zA-Z{}()\[\]\|\$%_#&\s])',
        r'\\\\\1',
        json_content
    )
    # Parse the JSON content
    try: 
        data = json.loads(json_content)
    except json.decoder.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}
    
    return data 

def visualize_tree_walk(tree_data, walk_data, filename="tree_walk_visualization_v5", format="png", wrap_width=35):
    """
    Visualizes a problem-solving tree and walk path using Graphviz (Version 5).
    Uses colors for walk edge categories.
    Adds step index number as a small label to each walk edge.
    Includes an HTML-based legend.

    Args:
        tree_data (dict): Dictionary of tree nodes.
        walk_data (list): List of walk steps.
        filename (str): Base name for the output file.
        format (str): Output format ('png', 'svg', 'pdf').
        wrap_width (int): Approx width for text wrapping in nodes.

    Returns:
        str: Full path to the rendered output file, or None if rendering fails.
    """
    dot = graphviz.Digraph(comment='Problem Solving Tree Walk V5', engine='dot')

    # --- Graph Attributes ---
    dot.attr(rankdir='TB', overlap='scale', ranksep='0.8', nodesep='0.6')

    # --- Global Node Attributes ---
    dot.node_attr.update(shape='box', style='rounded,filled',
                         fillcolor='lightcyan', fontsize='10')

    # --- Add Nodes --- (Same as V4)
    for node_id, node_info in tree_data.items():
        problem_text = node_info.get('Problem', 'N/A')
        result_text = node_info.get('Result', '')
        node_title = f"Node: {node_id}"
        if 'Initial State' in problem_text: node_title = "Initial State"
        else:
            is_last_target = node_id == walk_data[-1]['to']
            has_outgoing_calc = any(step['from'] == node_id and step['category'] == 'calculation/derivation' for step in walk_data)
            if is_last_target and not has_outgoing_calc and result_text: node_title = "Final Solution / State"
            elif node_info.get('parent') != 'none': node_title = f"Step/State: {node_id}"
        wrapped_problem = f"Problem: {textwrap.fill(problem_text, width=wrap_width)}"
        wrapped_result = f"Result: {textwrap.fill(result_text, width=wrap_width)}" if result_text else ""
        label_parts = [node_title, wrapped_problem]
        if wrapped_result: label_parts.append(wrapped_result)
        full_label = "\n".join(label_parts)
        dot.node(node_id, label=full_label)

    # --- Add Tree Edges (Parent links - No Label) --- (Same as V4)
    tree_edge_attr = {'color': 'black', 'arrowhead': 'none', 'constraint': 'true', 'penwidth': '1.5'}
    for node_id, node_info in tree_data.items():
        parent_id = node_info.get('parent')
        if parent_id and parent_id != 'none' and parent_id in tree_data:
            dot.edge(parent_id, node_id, **tree_edge_attr)

    # --- Add Walk Edges (Colored, Dashed, WITH Index Number Label) ---
    walk_colors = {
        'calculation/derivation': 'blue',
        'verification': 'red',
        'backtracking': 'darkgreen',
    }
    default_walk_color = 'purple'

    for i, step in enumerate(walk_data):
        from_node = step.get('from')
        to_node = step.get('to')
        category = step.get('category', 'unknown') # Still needed for color

        if from_node in tree_data and to_node in tree_data:
            edge_color = walk_colors.get(category, default_walk_color)
            step_number = str(i + 1) # Get the step number (1-based)

            # Add the step number as a label. Using standard 'label' places it mid-edge.
            # 'headlabel' or 'taillabel' could place it near nodes if preferred.
            dot.edge(from_node, to_node,
                     label=step_number,       # ADDED: Step number label
                     style='dashed',
                     color=edge_color,
                     fontcolor=edge_color,   # Match label color to edge color
                     fontsize='8',           # Small font size for the number
                     arrowhead='normal',
                     constraint='false',
                     penwidth='1.0')

    # --- Add Legend using HTML-like Label --- (Same as V4)
    legend_items = [
        (walk_colors.get('calculation/derivation', 'blue'), 'dashed', 'Calculation/Derivation'),
        (walk_colors.get('verification', 'red'), 'dashed', 'Verification'),
        (walk_colors.get('backtracking', 'darkgreen'), 'dashed', 'Backtracking'),
        ('black', 'solid', 'Tree Structure')
    ]
    html_rows = []
    html_rows.append('<TR><TD COLSPAN="2" ALIGN="CENTER" CELLPADDING="5"><B>Legend</B></TD></TR>') # Title row
    for color, style, text in legend_items:
        style_desc = f" ({style})"
        html_rows.append(f'<TR><TD BGCOLOR="{color}" FIXEDSIZE="TRUE" WIDTH="25" HEIGHT="5"></TD>' \
                         f'<TD ALIGN="LEFT" VALIGN="MIDDLE"><FONT POINT-SIZE="9">{text}{style_desc}</FONT></TD></TR>')
    html_label = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="5" CELLPADDING="4">\n{"".join(html_rows)}\n</TABLE>>'
    dot.node('legend_node', label=html_label, shape='box', style='', penwidth='0.5', margin='0.1')


    # --- Render Graph ---
    try:
        output_path = dot.render(filename, format=format, view=False, cleanup=True)
        print(f"Graph saved to: {output_path}")
        return output_path
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n--- Graphviz Error ---")
        print("Graphviz executable not found. Visualization failed.")
        print("Please install Graphviz (https://graphviz.org/download/)")
        print("and ensure its 'bin' directory is in your system's PATH.")
        print("----------------------\n")
        return None
    except Exception as e:
        print(f"An error occurred during rendering: {e}")
        return None

# Helper function to calculate distance (remains the same)
def _get_distance(node1_id, node2_id, parents, depths):
    """Calculates the distance (number of edges) between two nodes in a tree."""
    if node1_id not in depths or node2_id not in depths:
        return float('inf')
    if node1_id == node2_id:
        return 0
    path1_ancestors = {}
    curr = node1_id
    while curr is not None:
        path1_ancestors[curr] = depths[curr]
        curr = parents.get(curr)
    lca = None
    curr = node2_id
    while curr is not None:
        if curr in path1_ancestors:
            lca = curr
            break
        curr = parents.get(curr)
    if lca is None:
         return float('inf')
    distance = depths[node1_id] + depths[node2_id] - 2 * depths[lca]
    return distance

# Helper to build tree info (remains the same)
def _build_tree_info_from_parent_links(tree_data):
    """Parses tree data where nodes specify their parent and computes necessary info."""
    # --- (Code from previous answer - unchanged) ---
    if not isinstance(tree_data, dict) or not tree_data:
        print("Error: Tree data must be a non-empty dictionary.")
        return None, None, None, None, None
    parents = {}
    children = {node_id: [] for node_id in tree_data}
    root_id = None
    parent_nodes = set()
    for node_id, node_info in tree_data.items():
        if not isinstance(node_info, dict) or "parent" not in node_info:
            print(f"Error: Invalid format for node '{node_id}'. Missing 'parent' key.")
            return None, None, None, None, None
        parent = node_info["parent"]
        if parent == "none" or parent is None:
            if root_id is not None and root_id != node_id:
                print(f"Error: Multiple root nodes detected ('{root_id}' and '{node_id}').")
                return None, None, None, None, None
            root_id = node_id
            parents[node_id] = None
        else:
            if parent not in tree_data:
                 print(f"Error: Parent node '{parent}' listed for node '{node_id}' is not defined in the tree data.")
                 return None, None, None, None, None
            parents[node_id] = parent
            if parent in children:
                 children[parent].append(node_id)
            parent_nodes.add(parent)
    if root_id is None and tree_data:
        print("Error: No root node found (no node with parent='none').")
        return None, None, None, None, None
    elif root_id is None and not tree_data:
         print("Warning: Tree data is empty.")
         return {}, {}, set(), {}, None
    depths = {}
    if root_id is not None:
        queue = deque([(root_id, 0)])
        visited_for_depth = {root_id}
        depths[root_id] = 0
        while queue:
            current_node, current_depth = queue.popleft()
            for child_id in children.get(current_node, []):
                 if child_id not in visited_for_depth:
                     if child_id not in tree_data:
                         print(f"Warning: Child '{child_id}' (parent: '{current_node}') not found in tree_data keys. Skipping.")
                         continue
                     visited_for_depth.add(child_id)
                     depths[child_id] = current_depth + 1
                     # Ensure parent link is correct based on BFS traversal if needed
                     # parents[child_id] = current_node # Careful not to overwrite if structure allows multiple paths
                     queue.append((child_id, current_depth + 1))
    all_nodes = set(tree_data.keys())
    reachable_nodes = set(depths.keys())
    if all_nodes != reachable_nodes:
        print(f"Warning: Nodes exist but are not reachable from root '{root_id}': {all_nodes - reachable_nodes}")
    leaves = {node_id for node_id in reachable_nodes if not children.get(node_id)}
    return parents, depths, leaves, children, root_id
    # --- (End of unchanged code) ---


# Helper function to reconstruct the walk sequence (remains the same)
def _reconstruct_walk_sequence(walk_steps_list):
    """Reconstructs the sequence of visited nodes from a list of 'from'-'to' steps."""
     # --- (Code from previous answer - unchanged) ---
    if not walk_steps_list or not isinstance(walk_steps_list, list):
        return []
    walk_sequence = []
    first_step = walk_steps_list[0]
    if not isinstance(first_step, dict) or 'from' not in first_step or 'to' not in first_step:
         print("Error: Invalid format for the first step in walk_steps_list.")
         return None
    walk_sequence.append(first_step['from'])
    walk_sequence.append(first_step['to'])
    last_node = first_step['to']
    for i in range(1, len(walk_steps_list)):
        step = walk_steps_list[i]
        if not isinstance(step, dict) or 'from' not in step or 'to' not in step:
             print(f"Error: Invalid format for step at index {i} in walk_steps_list.")
             return None
        if step['from'] != last_node:
             print(f"Warning: Discontinuity in walk steps at index {i}. Expected from '{last_node}', got from '{step['from']}'. Following sequence.")
        walk_sequence.append(step['to'])
        last_node = step['to']
    return walk_sequence
     # --- (End of unchanged code) ---

# **** NEW Function to filter leaf visits based on verification steps ****
def _filter_leaf_visits(full_walk_sequence, walk_steps_list, leaves, depths):
    """
    Filters the sequence of leaf visits based on the verification rule.

    Args:
        full_walk_sequence (list): The complete sequence of visited node IDs.
        walk_steps_list (list): The original list of step dictionaries.
        leaves (set): Set of leaf node IDs.
        depths (dict): Dictionary mapping node IDs to their depth (used to check reachability).

    Returns:
        list: The filtered sequence of leaf node IDs.
    """
    all_leaf_visits_with_indices = []
    for i, node_id in enumerate(full_walk_sequence):
        # Ensure node is reachable and is a leaf
        if node_id in depths and node_id in leaves:
            all_leaf_visits_with_indices.append({"node": node_id, "index": i})

    if not all_leaf_visits_with_indices:
        return []

    filtered_visits = [all_leaf_visits_with_indices[0]] # Start with the first leaf visit

    for i in range(1, len(all_leaf_visits_with_indices)):
        current_visit = all_leaf_visits_with_indices[i]
        last_accepted_visit = filtered_visits[-1]

        # Check if the current leaf node is the same as the last accepted one
        if current_visit["node"] == last_accepted_visit["node"]:
            # If same node, check the intermediate steps
            start_step_index = last_accepted_visit["index"] # Index in walk_steps corresponds to sequence index
            end_step_index = current_visit["index"] - 1

            all_intermediate_verification = True
            if start_step_index > end_step_index: # Should not happen if indices are correct
                 print(f"Warning: Logic error in step indices between leaf visits at sequence indices {last_accepted_visit['index']} and {current_visit['index']}.")
                 all_intermediate_verification = False # Treat as non-verification path
            else:
                # Iterate through the relevant steps in walk_steps_list
                for k in range(start_step_index, end_step_index + 1):
                    # Check bounds for safety
                    if k >= len(walk_steps_list):
                        print(f"Warning: Step index {k} out of bounds for walk_steps_list (len {len(walk_steps_list)}).")
                        all_intermediate_verification = False
                        break
                    step = walk_steps_list[k]
                    if step.get("category") != "verification":
                        all_intermediate_verification = False
                        break

            # If all intermediate steps were verification, this visit is redundant
            if all_intermediate_verification:
                continue # Skip adding this visit to filtered_visits
            else:
                # If even one step wasn't verification, this is a significant visit
                filtered_visits.append(current_visit)
        else:
            # If it's a different leaf node, it's always significant
            filtered_visits.append(current_visit)

    # Return just the sequence of node IDs from the filtered visits
    final_leaf_sequence = [visit["node"] for visit in filtered_visits]
    return final_leaf_sequence


# **** NEW Function to compute average solution count ****
def compute_average_solution_count(tree_data, walk_steps_list):
    """
    Computes the number of leaf nodes in the tree.

    Args:
        tree_data (dict): Dict representing tree {node_id: {"parent": parent_id,...}}
        walk_steps_list (list): List of dicts representing steps (unused in this function but kept for consistency).

    Returns:
        int or None: The number of leaf nodes, or None if tree data is invalid.
    """
    parents, depths, leaves, children, root_id = _build_tree_info_from_parent_links(tree_data)

    if parents is None:
        print("Error: Could not process tree data for solution count.")
        return None

    if not leaves:
        # This could mean no reachable leaf nodes or an empty tree.
        # Depending on definition, 0 might be more appropriate than None if tree is valid but has no leaves.
        print("Info: No leaf nodes found or tree is empty. Returning 0 solutions.")
        return 0

    return len(leaves)


# **** Main function updated to use the filtering ****
def compute_filtered_average_jump_distance(tree_data, walk_steps_list):
    """
    Computes the Average Jump Distance (AJD) based on a *filtered* sequence
    of leaf visits, where visits to the same leaf are ignored if all
    intermediate steps were 'verification'.

    Args:
        tree_data (dict): Dict representing tree {node_id: {"parent": parent_id,...}}
        walk_steps_list (list): List of dicts representing steps [{'from': 'n1', 'to': 'n2', 'category': 'cat'}, ...]

    Returns:
        float or None: The computed filtered AJD, or None if undefined.
    """

    # --- Step 1: Build tree information ---
    parents, depths, leaves, children, root_id = _build_tree_info_from_parent_links(tree_data)
    if parents is None:
        print("Error: Could not process tree data.")
        return None
    if not leaves:
        print("Warning: No leaf nodes found reachable from root. Filtered AJD is undefined.")
        return None

    # --- Step 2: Reconstruct the full walk sequence ---
    full_walk_sequence = _reconstruct_walk_sequence(walk_steps_list)
    if full_walk_sequence is None:
        print("Error: Could not reconstruct walk sequence.")
        return None
    if not full_walk_sequence:
         print("Info: Walk sequence is empty.")
         return None

    # --- Step 3: Filter the leaf visits based on the verification rule ---
    print("Original full walk sequence:", full_walk_sequence) # Debug
    filtered_leaf_sequence = _filter_leaf_visits(full_walk_sequence, walk_steps_list, leaves, depths)
    print("Filtered leaf visit sequence:", filtered_leaf_sequence) # Debug

    # --- Step 4: Calculate average jump distance on the filtered sequence ---
    M_filtered = len(filtered_leaf_sequence)

    if M_filtered < 2:
        return 0

    sum_distances = 0
    jumps_with_errors = 0
    num_jumps = M_filtered - 1

    for i in range(num_jumps):
        u = filtered_leaf_sequence[i]
        v = filtered_leaf_sequence[i+1]

        distance = _get_distance(u, v, parents, depths)

        if distance == float('inf'):
             print(f"Warning: Skipping jump ({u} -> {v}) in filtered sequence due to distance calculation error.")
             jumps_with_errors += 1
             # Skip this jump's contribution
             continue
        sum_distances += distance

    if jumps_with_errors > 0:
         print(f"Warning: {jumps_with_errors} jumps in the filtered sequence could not be calculated. Result may be inaccurate.")
         # Return None if strict error handling is needed for any failed jump
         # return None

    # Calculate the average based on the number of jumps in the filtered sequence
    if num_jumps == 0: # Should be caught by M_filtered < 2, but safety
        return None

    # Average the sum of successfully calculated distances over the total number of jumps attempted in filtered sequence
    filtered_ajd = sum_distances / num_jumps
    return filtered_ajd

def get_analysis(idx, results, results_dir, overwrite=False, corr_constraint=None, dataset_name="game24"):
    result_path = f"{results_dir}/tree_vis_{analysis_model}/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        if "<answer>" in output_str:
            # Find all <answer></answer> pairs
            answer_matches = re.findall(r'<answer>(.*?)</answer>', output_str, re.DOTALL)
            if answer_matches:
                answer_str = answer_matches[-1]  # Use the first match
        # Call compare_answer based on dataset type
        if dataset_name == "game24":
            corr = compare_answer(answer_str)
        elif dataset_name == "sudoku":
            # For sudoku, compare_answer signature is different
            # We need ground_truth, but we'll just check format here
            corr = compare_answer(answer_str, results.iloc[idx]["reward_model"]["ground_truth"]["label"][0])
        else:
            corr = compare_answer(answer_str)
        
        # Choose prompt based on dataset type
        if dataset_name == "sudoku":
            tree_prompt = get_tree_prompt_sudoku(input_str, output_str)
        else:
            tree_prompt = get_tree_prompt(input_str, output_str)
        
        tree_json = llm.generate([{
            "role": "user",
            "content": tree_prompt
        }])[2]
        
        # Choose walk prompt based on dataset type
        if dataset_name == "sudoku":
            walk_prompt = get_walk_prompt_sudoku(input_str, output_str, tree_json)
        else:
            walk_prompt = get_walk_prompt(input_str, output_str, tree_json)
        walk_json = llm.generate([{
            "role": "user",
            "content": walk_prompt
        }])[2]
        json_data = {
            "tree": parse_json(tree_json),
            "walk": parse_json(walk_json),
            "corr": corr,
        }
        save_json(json_data, result_path)
    else:
        json_data = load_json(result_path)
        output_str = results.iloc[idx]["responses"][0]
        if "<answer>" in output_str:
            # Find all <answer></answer> pairs
            answer_matches = re.findall(r'<answer>(.*?)</answer>', output_str, re.DOTALL)
            if answer_matches:
                answer_str = answer_matches[-1]  # Use the first match
        # Call compare_answer based on dataset type
        if dataset_name == "game24":
            corr = compare_answer(answer_str)
        elif dataset_name == "sudoku":
            corr = compare_answer(answer_str, results.iloc[idx]["reward_model"]["ground_truth"]["label"][0])
        else:
            corr = compare_answer(answer_str)
        json_data["corr"] = corr
        
    if corr_constraint is not None:
        if json_data["corr"] != corr_constraint:
            return None
        

    vis_path = visualize_tree_walk(json_data["tree"], json_data["walk"], filename=f"{results_dir}/tree_vis_{analysis_model}/{idx}", format="pdf")
    filtered_ajd = compute_filtered_average_jump_distance(json_data["tree"], json_data["walk"])
    print(f"Index {idx}: Filtered AJD = {filtered_ajd}")
    
    average_solution_count = compute_average_solution_count(json_data["tree"], json_data["walk"])
    print(f"Index {idx}: Solution Count = {average_solution_count}")
    
    # Count arrow types
    calculation_count = 0
    verification_count = 0
    backtracking_count = 0
    walk_steps = json_data.get("walk", [])
    if isinstance(walk_steps, list):
        for step in walk_steps:
            if isinstance(step, dict) and "category" in step:
                category = step.get("category")
                if category == "calculation/derivation":
                    calculation_count += 1
                elif category == "verification":
                    verification_count += 1
                elif category == "backtracking":
                    backtracking_count += 1
    
    print(f"Index {idx}: Calculation Arrows = {calculation_count}")
    print(f"Index {idx}: Verification Arrows = {verification_count}")
    print(f"Index {idx}: Backtracking Arrows = {backtracking_count}")

    # Count total nodes
    total_node_count = len(json_data.get("tree", {}))
    print(f"Index {idx}: Total Node Count = {total_node_count}")

    # Calculate Forgetting Rate
    if calculation_count < total_node_count:
        forgetting_rate = 0
    else:
        forgetting_rate = 1
    print(f"Index {idx}: Forgetting Rate = {forgetting_rate}")

    # Calculate Average Verification Rate
    total_arrows = calculation_count + verification_count + backtracking_count
    if total_arrows > 0:
        average_verification_rate = verification_count / total_arrows
    else:
        average_verification_rate = 0 # Or None, depending on desired behavior for no arrows
    print(f"Index {idx}: Verification Rate = {average_verification_rate:.4f}")

    # --- Call compare_answer for each node in the Filtered leaf visit sequence (using Problem field) and print the output ---
    tree_data_for_leaf_check = json_data.get("tree", {})
    walk_data_for_leaf_check = json_data.get("walk", [])

    num_filtered_leaves_for_sr = 0 # Initialize for Success Rate calculation
    num_successful_filtered_leaves_for_sr = 0 # Initialize for Success Rate calculation
    filtered_leaf_is_correct_list_for_or = [] # Initialize for Overthinking Rate

    if not tree_data_for_leaf_check or not isinstance(tree_data_for_leaf_check, dict):
        print(f"Index {idx}: Tree data is missing or invalid for leaf answer check.")
    else:
        parents, depths, leaves, children, root_id = _build_tree_info_from_parent_links(tree_data_for_leaf_check)
        if parents is None:
            print(f"Index {idx}: Could not build tree info for leaf answer check.")
        elif not leaves:
            print(f"Index {idx}: No leaf nodes found in the tree for leaf answer check.")
        else:
            full_walk_sequence = _reconstruct_walk_sequence(walk_data_for_leaf_check)
            if full_walk_sequence is None:
                print(f"Index {idx}: Could not reconstruct walk sequence for leaf answer check.")
            elif not full_walk_sequence:
                print(f"Index {idx}: Walk sequence is empty for leaf answer check.")
            else:
                filtered_leaf_sequence_for_check = _filter_leaf_visits(full_walk_sequence, walk_data_for_leaf_check, leaves, depths)
                
                if filtered_leaf_sequence_for_check:
                    print(f"Index {idx}: Checking answers for Filtered Leaf Visit Sequence (using Problem field): {filtered_leaf_sequence_for_check}")
                    num_filtered_leaves_for_sr = len(filtered_leaf_sequence_for_check) # Total number of filtered leaves

                    for node_id in filtered_leaf_sequence_for_check:
                        if node_id in tree_data_for_leaf_check and isinstance(tree_data_for_leaf_check[node_id], dict):
                            leaf_node_info = tree_data_for_leaf_check[node_id]
                            # Get the expression from the 'Problem' field.
                            expression_from_leaf_problem = leaf_node_info.get("Problem")

                            if expression_from_leaf_problem is not None:
                                # Evaluate based on dataset type
                                if dataset_name == "game24":
                                    # Add "=24" to the expression string from the Problem field
                                    # so that the compare_answer function can evaluate it correctly.
                                    expression_to_evaluate = str(expression_from_leaf_problem) + "=24"
                                    is_correct_for_leaf = compare_answer(expression_to_evaluate)
                                elif dataset_name == "sudoku":
                                    # For sudoku, compare with ground truth
                                    ground_truth = results.iloc[idx]["reward_model"]["ground_truth"]["label"][0]
                                    is_correct_for_leaf = compare_answer(str(expression_from_leaf_problem), ground_truth)
                                else:
                                    expression_to_evaluate = str(expression_from_leaf_problem)
                                    is_correct_for_leaf = compare_answer(expression_to_evaluate)
                                print(f"  Leaf Node {node_id}: Problem='{expression_from_leaf_problem}', compare_answer output={is_correct_for_leaf}")
                                if is_correct_for_leaf == 1: # Count successful ones
                                    num_successful_filtered_leaves_for_sr += 1
                                filtered_leaf_is_correct_list_for_or.append(is_correct_for_leaf == 1)
                            else:
                                print(f"  Leaf Node {node_id}: 'Problem' field is missing or None.")
                                filtered_leaf_is_correct_list_for_or.append(False)
                        else:
                            print(f"  Leaf Node {node_id}: Not found in tree_data or invalid format.")
                            filtered_leaf_is_correct_list_for_or.append(False)
                else:
                    print(f"Index {idx}: Filtered leaf visit sequence is empty for answer check.")
    # --- End of modified logic ---

    sample_success_rate = 0.0
    if num_filtered_leaves_for_sr > 0:
        sample_success_rate = num_successful_filtered_leaves_for_sr / num_filtered_leaves_for_sr
    print(f"Index {idx}: Success Rate = {sample_success_rate:.4f}")

    # Calculate Overthinking Rate
    sample_overthinking_rate = 0.0
    if num_filtered_leaves_for_sr > 0: # Denominator must be greater than 0
        first_success_idx_for_or = -1
        for i, is_correct in enumerate(filtered_leaf_is_correct_list_for_or):
            if is_correct:
                first_success_idx_for_or = i
                break
        
        if first_success_idx_for_or != -1: # If a success was found
            nodes_after_first_success = num_filtered_leaves_for_sr - (first_success_idx_for_or + 1)
            sample_overthinking_rate = nodes_after_first_success / num_filtered_leaves_for_sr
    print(f"Index {idx}: Overthinking Rate = {sample_overthinking_rate:.4f}")

    return {
        "graph": vis_path,
        "filtered_ajd": filtered_ajd,
        "answer": answer_str,
        "average_solution_count": average_solution_count,
        "calculation_count": calculation_count,
        "verification_count": verification_count,
        "backtracking_count": backtracking_count,
        "total_node_count": total_node_count,
        "forgetting_rate": forgetting_rate,
        "average_verification_rate": average_verification_rate,
        "corr": json_data["corr"],
        "success_rate": sample_success_rate,
        "overthinking_rate": sample_overthinking_rate, # Added overthinking rate
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs='+', default=[])
    parser.add_argument("--dataset_name", type=str, default="game24", choices=["game24", "sudoku"])
    parser.add_argument("--model_name", type=str, nargs='+', default=["deepseek-ai/deepseek-reasoner"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.00])
    parser.add_argument("--mode", type=str, default="default", choices=["default", "ricl_1", "ricl_2", "ricl_3", "ricl_4", "ricl_5", "ricl_6", "ricl_7", "ricl_8", "ricl_9", "ricl_10", "instructiona", "instructionb", "instructionc", "instructiond"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--corr_constraint", type=lambda x: None if x == "None" else int(x), default=None, choices=[None, 0, 1])
    parser.add_argument("--replicate_id", type=int, default=0)
    parser.add_argument("--analysis_model", type=str, default="google/gemini-2.5-pro-preview-03-25")
    parser.add_argument("--response_length", type=int, default=404, help="Response length used in model generation")
    args = parser.parse_args()
    
    # Dynamically import compare_answer based on dataset_name
    if args.dataset_name == "game24":
        from verl.utils.reward_score.game24 import compare_answer
    elif args.dataset_name == "sudoku":
        from verl.utils.reward_score.sudoku import compare_answer
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    # model = "xai/grok-3-mini-beta"
    # analysis_model = "claude/claude-3-7-sonnet-20250219-thinking"
    # model = "google/gemini-2.5-pro-preview-03-25"
    
    analysis_model = args.analysis_model
    llm = LLMAPI(
        api_key=supported_llms[analysis_model]["api_key"],
        model_name=analysis_model,
        template_type="reasoning_api"
    )
    
    if len(args.temperature) == 1:
        temperatures = [args.temperature[0] for _ in args.model_name]
    else:
        if len(args.temperature) != len(args.model_name):
            raise ValueError(f"Number of temperatures ({len(args.temperature)}) must match number of models ({len(args.model_name)})")
        temperatures = args.temperature
        
    all_metrics = {}
    for model_name, temperature in zip(args.model_name, temperatures):
        if args.wandb:
            wandb_config = {
                "dataset_name": args.dataset_name,
                "model_name": model_name,
                "num_samples": args.num_samples,
                "temperature": temperature,
                "mode": args.mode,
                "replicate_id": args.replicate_id,
                "analysis_model": analysis_model,
            }
            project_name = f"{WANDB_INFO['project']}-tree-vis-v3"
            
            if not wandb_init(project_name, WANDB_INFO["entity"], wandb_config):
                exit()
                
            
        if "ricl" in args.mode:
            template_type = f"{supported_llms[model_name]['template_type']}_{args.mode}"
        else: 
            template_type = supported_llms[model_name]["template_type"]
            
        if "instruction" in args.mode:
            data_mode = args.mode
        else:
            data_mode = "default"

        results_dir = get_result_dir(
            dataset_name = args.dataset_name,
            model_name = model_name,
            shot = 0,
            template_type = template_type,
            response_length = args.response_length,
            num_samples = args.num_samples,
            feature_noise = supported_datasets[args.dataset_name]["feature_noise"],
            label_noise = 0.0,
            data_mode = data_mode,
            n_query = 1,
            temperature = temperature,
            replicate_id = args.replicate_id,
        )
        results = pd.read_parquet(f"{results_dir}/test_default.parquet")
        
        if len(args.idx) == 0:
            idxs = list(range(len(results)))
            random.shuffle(idxs)
        else:
            idxs = args.idx
        
        filtered_ajds = []
        answers = []
        average_solution_counts = [] # Initialize list for average solution counts
        calculation_arrow_counts = []
        verification_arrow_counts = []
        backtracking_arrow_counts = []
        total_node_counts = [] # Initialize list for total node counts
        forgetting_rates = [] # Initialize list for forgetting rates
        average_verification_rates_list = [] # Initialize list for average_verification_rate
        corrs = []
        success_rates_list = [] # Initialize list for success rates
        overthinking_rates_list = [] # Initialize list for overthinking rates

        for idx in tqdm(idxs):
            attempts, success, overwrite, skip = 0, False, args.overwrite, False
            while attempts < 5 and not success:
                try:
                    graph_metric = get_analysis(idx, results, results_dir, overwrite, args.corr_constraint, args.dataset_name)
                    success = True
                    if args.corr_constraint is not None:
                        if graph_metric is None:
                            skip = True
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except pdb.bdb.BdbQuit:
                    raise pdb.bdb.BdbQuit
                except Exception as e:
                    print(f"Error: {type(e)} {e}")
                    print(f"Attempt {attempts} failed")
                    attempts += 1
                    overwrite = True
                    continue
            
            if skip: continue
            
            filtered_ajds.append(graph_metric["filtered_ajd"])
            answers.append(graph_metric["answer"])
            average_solution_counts.append(graph_metric["average_solution_count"]) # Append average solution count
            calculation_arrow_counts.append(graph_metric["calculation_count"])
            verification_arrow_counts.append(graph_metric["verification_count"])
            backtracking_arrow_counts.append(graph_metric["backtracking_count"])
            total_node_counts.append(graph_metric["total_node_count"]) # Append total node count
            forgetting_rates.append(graph_metric["forgetting_rate"]) # Append forgetting rate
            average_verification_rates_list.append(graph_metric["average_verification_rate"]) # Append average_verification_rate
            corrs.append(graph_metric["corr"])  
            if graph_metric.get("success_rate") is not None:
                success_rates_list.append(graph_metric["success_rate"])
            else:
                success_rates_list.append(0.0) # Default if None, though it should always be float
            
            if graph_metric.get("overthinking_rate") is not None:
                overthinking_rates_list.append(graph_metric["overthinking_rate"])
            else:
                overthinking_rates_list.append(0.0) # Default if None

        metric_dict = {
            "filtered_ajd": filtered_ajds,
            "answers": answers,
            "average_solution_count": average_solution_counts,
            "calculation_arrow_counts": calculation_arrow_counts,
            "verification_arrow_counts": verification_arrow_counts,
            "backtracking_arrow_counts": backtracking_arrow_counts,
            "total_node_counts": total_node_counts,
            "forgetting_rates": forgetting_rates,
            "average_verification_rates": average_verification_rates_list,
            "corrs": corrs,
            "success_rates": success_rates_list,
            "overthinking_rates": overthinking_rates_list, # Added to metric_dict
        }
        
        metric_df = pd.DataFrame(metric_dict)
        metric_df = metric_df.dropna(how='any')
        metric_df.to_csv(f"{results_dir}/tree_vis_{analysis_model}/metric_df.csv")

        filtered_ajd = np.mean(metric_df["filtered_ajd"])
        print(f"Filtered AJD: {filtered_ajd}")
        
        avg_sol_count = np.mean(metric_df["average_solution_count"])
        print(f"Average Solution Count: {avg_sol_count}") # Print average solution count
        
        avg_calc_arrows = np.mean(metric_df["calculation_arrow_counts"])
        avg_ver_arrows = np.mean(metric_df["verification_arrow_counts"])
        avg_back_arrows = np.mean(metric_df["backtracking_arrow_counts"])

        print(f"Average Calculation Arrows: {avg_calc_arrows}")
        print(f"Average Verification Arrows: {avg_ver_arrows}")
        print(f"Average Backtracking Arrows: {avg_back_arrows}")
            
        avg_total_nodes = np.mean(metric_df["total_node_counts"])
        print(f"Average Total Node Count: {avg_total_nodes}") # Print average total_node_count
        
        avg_forgetting_rate = np.mean(metric_df["forgetting_rates"])
        print(f"Average Forgetting Rate: {avg_forgetting_rate}") # Print average forgetting_rate
        
        overall_avg_verification_rate = np.mean(metric_df["average_verification_rates"])
        print(f"Average Verification Rate: {overall_avg_verification_rate:.4f}" if overall_avg_verification_rate is not None else "Average Verification Rate: None")
            
        avg_corr = np.mean(metric_df["corrs"])
        print(f"Average Correlation: {avg_corr}")
        
        avg_success_rate = np.mean(metric_df["success_rates"]) if "success_rates" in metric_df.columns and not metric_df["success_rates"].empty else 0.0
        print(f"Average Success Rate: {avg_success_rate:.4f}")
        
        avg_overthinking_rate = np.mean(metric_df["overthinking_rates"]) if "overthinking_rates" in metric_df.columns and not metric_df["overthinking_rates"].empty else 0.0
        print(f"Average Overthinking Rate: {avg_overthinking_rate:.4f}")
        
        
        if args.wandb:
            wandb.log({
                "filtered_ajd": filtered_ajd,
                "average_solution_count": avg_sol_count, # Log average solution count
                "average_calculation_arrows": avg_calc_arrows,
                "average_verification_arrows": avg_ver_arrows,
                "average_backtracking_arrows": avg_back_arrows,
                "average_total_node_count": avg_total_nodes, # Log average total_node_count
                "average_forgetting_rate": avg_forgetting_rate, # Log average forgetting_rate
                "overall_average_verification_rate": overall_avg_verification_rate, # Log overall_average_verification_rate
                "average_correlation": avg_corr,
                "average_success_rate": avg_success_rate,
                "average_overthinking_rate": avg_overthinking_rate, # Log average overthinking rate
            })
            wandb.finish()
            
        for metric in metric_dict:
            if not metric in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].extend(metric_dict[metric])
            
    # Check the importance of each feature using XGBoost regressor
    feature_columns = [
        "filtered_ajd",
        "forgetting_rates",
        "average_verification_rates",
        "average_solution_count",
        "success_rates",
        "overthinking_rates"
    ]
    target_column = "corrs"

    all_metrics_df = pd.DataFrame(all_metrics)
    # Prepare X and y, dropping rows with NaN in any feature or target
    valid_rows = all_metrics_df[feature_columns + [target_column]].dropna()
    X = valid_rows[feature_columns].values
    y = valid_rows[target_column].values

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Since 'corr' is a binary variable, use a classifier instead of regressor
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Classifier Accuracy: {accuracy:.4f}")

        # Check the accuracy of a majority classifier (predicts the most common class)
        if len(y_test) > 0:
            majority_class = Counter(y_train).most_common(1)[0][0]
            majority_pred = np.full_like(y_test, fill_value=majority_class)
            majority_accuracy = np.mean(majority_pred == y_test)
            print(f"Majority Classifier Accuracy: {majority_accuracy:.4f}")
        else:
            print("Not enough test data to compute majority classifier accuracy.")
            majority_accuracy = None

        # Check feature importances
        importances = model.feature_importances_
        # Optionally, print sorted feature importances
        sorted_features = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
        print("Features ranked by importance:")
        feature_importance_dict = {}
        for name, importance in sorted_features:
            print(f"  {name}: {importance:.4f}")
            feature_importance_dict[name] = float(importance)

        # Perform K-means clustering on all samples
        X_incorr, y_incorr = X[y == 0], y[y == 0]
        if len(X_incorr) > 1:  # K-means requires at least 2 samples
            # Normalize features for K-means clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_incorr_normalized = scaler.fit_transform(X_incorr)
            
            print("Feature normalization applied for K-means clustering")
            print("Original feature ranges:")
            for i, feature_name in enumerate(feature_columns):
                feature_min, feature_max = X_incorr[:, i].min(), X_incorr[:, i].max()
                print(f"  {feature_name}: [{feature_min:.4f}, {feature_max:.4f}]")
            
            # Directly use k=2 for clustering
            k = 3
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_incorr_normalized)  # Use normalized data
            cluster_labels = kmeans.labels_
            print(f"K-means clustering on normalized data with k={k} completed.")
            print(f"Number of clusters: {kmeans.n_clusters}")
            print(f"Cluster sizes: {np.bincount(cluster_labels)}")
            
            # Analyze feature means per cluster (using original scale for interpretability)
            print("Feature means per cluster (original scale):")
            for cluster in range(kmeans.n_clusters):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                cluster_data = X_incorr[cluster_indices]  # Use original data for interpretation
                if len(cluster_data) > 0:
                    cluster_means = np.mean(cluster_data, axis=0)
                    cluster_corrs = y_incorr[cluster_indices]
                    avg_corrs = np.mean(cluster_corrs) if len(cluster_corrs) > 0 else 0
                    print(f"  Cluster {cluster} (size: {len(cluster_data)}):")
                    for feature_name, mean_value in zip(feature_columns, cluster_means):
                        print(f"    {feature_name}: {mean_value:.4f}")
                    print(f"    Average corrs: {avg_corrs:.4f}")
        else:
            print("Not enough samples for K-means clustering (minimum 2 required).")


        # Compose the output dictionary
        summary = {
            "xgboost_classifier_accuracy": float(accuracy),
            "majority_classifier_accuracy": float(majority_accuracy) if majority_accuracy is not None else None,
            "feature_importances": feature_importance_dict,
            "sorted_feature_importances": [
                {"feature": name, "importance": float(importance)}
                for name, importance in sorted_features
            ],
        }

        # Compose the output filename
        corr_constraint_str = str(args.corr_constraint) if args.corr_constraint is not None else "None"
        dataset_name_str = str(args.dataset_name)
        output_dir =f"{root_dir}/results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"success_fail_summary_{corr_constraint_str}_{dataset_name_str}.json"
        )

        save_json(summary, output_path)
        print(f"Saved summary to {output_path}")
    else:
        print("Not enough valid data to check feature importance.")

