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

from verl.utils.llm_api import LLMAPI
from constants import supported_llms
import wandb
from environment import WANDB_INFO

import numpy as np
from collections import deque
# Note: The following import might need to be moved to the top of the file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from verl.utils.reward_score.math500 import compute_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import Counter

# model = "xai/grok-3-mini-beta"
# model = "claude/claude-3-7-sonnet-20250219-thinking"
model = "google/gemini-2.5-pro-preview-03-25"
llm = LLMAPI(
    api_key=supported_llms[model]["api_key"],
    model_name=model,
    template_type="reasoning_api"
)

def get_tree_prompt(input_str, output_str):
    return f"""
Your task is to analyze a detailed thinking process for solving a math problem (provided below) and convert it into a reasoning tree. This tree must represent the **chronological flow of solving substantive, mathematically well-posed subproblems or distinct attempts**, starting from an initial state and culminating in answering the original question.

Represent this structure as a **single JSON object** where keys are unique node IDs (e.g., "node1", "node2") and values are node objects detailing each state or subproblem attempt.

**Core Principles for Tree Generation:**

* **Chronological Flow & Dependency:** The tree follows the order of substantive steps/attempts in the reasoning. Parent links indicate the preceding step whose `Result` provides necessary mathematical input.
  **BRANCHING AND SUBSTEP RULE:** 
    - Create a new branch **if and only if** the reasoning process explicitly abandons or gives up on a previous approach and then starts a new, distinct solution plan. In other words, a new branch is created always and only when the previous line of reasoning is abandoned and a fundamentally different method is attempted. The new branch should start from the most recent shared node. Even if the solver does not immediately abandon the previous approach, we still consider it an Abandoned Attempt Node and mark it with [Path abandoned] if a different method is initiated that departs from the original direction.
    - Importantly, whenever a new branch is created, the leaf node where the previous method ended must be explicitly marked with [Path abandoned].
    - Conversely, if the current node is marked with [Path abandoned], a new branch must always be created.
    - Importantly, for all subproblems or calculations within a single uninterrupted attempt, even if subcalculations are mathematically independent, represent these steps sequentially in the order they are performed in the reasoning: each node's parent must be the immediately preceding node within that attempt.  
    That is, substeps within any one attempt always form a single chain.
* **Substantive, Well-Posed Steps Only:** Nodes must represent **major** intermediate calculations or logical deductions constituting a clear, self-contained mathematical task (like a homework sub-problem). **Aggressively filter out** setup actions, strategy descriptions, narrative, verification, and trivial calculations/manipulations. Minor algebraic steps within a larger logical step must be grouped.
* **Include Failed Attempts:** Represent distinct, substantive calculation or derivation attempts that were **explicitly abandoned** in the reasoning as separate nodes in the chronological flow. **Do not filter these out.**
* **Focus on Mathematical Task:** Intermediate `Problem` fields must state a clear mathematical objective based on **all necessary given mathematical conditions and inputs**, avoiding descriptions of the reasoner's process or assumptions *within the Problem text*.
* **Special Final Node:** The node performing the last calculation for the final answer uses the original problem statement as its `Problem`.

**Node Object Structure:**
Each node object must contain: `Problem`, `parent`, `Result`.

1.  **`Problem` (String): Defines the specific mathematical task for this node.**
    * **`node1` (Root):** Must be exactly "Initial State".
    * **Intermediate Nodes (`node2` to `node(N-1)`):** Formulates a **clear, mathematically well-posed, and self-contained task representing a substantive step or distinct attempt.** Each node represents achieving a distinct intermediate objective through calculation or deduction.
        * **Format:** Start with "Given..." listing **all essential mathematical conditions, constraints, equations, and input values** (often from parent `Result` or established context like 'point P is on curve C') needed to define and solve *this specific task*. End with a specific mathematical question/instruction (e.g., "Calculate...", "Solve...", "Derive...").
        * **Content:** The formulation must focus purely on the **mathematical task**, making it **understandable and solvable in isolation** like a homework sub-problem, using only the provided "Given..." information and general mathematical knowledge. **CRITICAL RULE:** The `Problem` text **must not** include descriptions of the reasoner's strategy, assumptions, or procedural instructions reflecting the reasoning flow. State only the necessary mathematical conditions and the objective. The task must be **substantive**. **CRITICAL FILTERING RULE:** **DO NOT** create separate nodes for individual algebraic manipulations... [rest of filtering rule stays the same - GROUP minor operations]. Also filter out narrative, setup, verification. No meta-tags or node ID references.
    * **`nodeN` (Final Calculation Node):** **This node represents the very last calculation step that produces the final answer.** Its `Problem` field **must contain the verbatim Original Problem Statement.**

2.  **`parent` (String): Identifies the immediately preceding substantive step providing necessary input.**
    * **`node1`:** Must be "none".
    * **Other Nodes (`node2` to `nodeN`):** Must be the ID of the node whose `Result` provides the direct mathematical prerequisite for the task in the current node's `Problem`. (For abandoned attempts, the parent is the node preceding the attempt).

3.  **`Result` (String): Records the mathematical outcome of completing the task.**
    * **`node1`:** "Original problem statement provided as context." (or similar).
    * **Intermediate Nodes (`node2` to `node(N-1)`):** The direct mathematical outcome of achieving the task defined in `Problem`. Summarizes the result of grouped operations.
    * **Abandoned Attempt Nodes:** Must state any partial outcome and explicitly end with "[Path abandoned]".
    * **`nodeN` (Final Calculation Node):** Must be the **final answer** to the Original Problem Statement.

**Instructions for Analysis:**
1.  **Inputs:** Use the "Original Problem Statement" and "Input Reasoning Process".
2.  **Identify & Filter Steps:** Read the reasoning chronologically. Identify **major** calculation phases, key logical deductions, or distinct attempts. **Crucially, ensure that distinct, substantive attempts explicitly marked as abandoned in the reasoning are identified and *not* filtered out.** Apply the **CRITICAL FILTERING and GROUPING RULES** aggressively: Group sequences of trivial algebraic steps into the single larger objective they serve. Filter out non-mathematical content, setup, strategy descriptions/assumptions-as-actions, and verification. Only create nodes for the remaining substantive steps and distinct abandoned attempts.
3.  **Create Nodes Sequentially:**
    * Create `node1`.
    * For each identified **substantive step/objective/attempt** *before* the final answer calculation: Create the corresponding intermediate node (`node2`, `node3`, ...). Determine `parent`. Formulate the `Problem` strictly according to Rule 1 (well-posed, self-contained task including **all necessary conditions/constraints**, no process descriptions). Record `Result`. Link abandoned attempt nodes chronologically.
    * For the **final calculation step**: Create `nodeN`. Determine `parent`. Set `Problem` to verbatim Original Problem Statement. Set `Result` to final answer.
4.  **Formatting:** Use LaTeX (`$...$`) for all math notation.
5.  **Output:** Produce a single JSON object.

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

Generate the JSON output based on these instructions.
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

        if from_node == 'none' and to_node == 'node1':
            continue

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
    """Calculates the distance (number of edges) between two nodes in a tree.
    Returns a tuple: (distance_from_node1_to_lca, distance_from_lca_to_node2, total_distance).
    Returns (None, None, float('inf')) if distance cannot be calculated.
    Returns (0, 0, 0) if node1_id == node2_id.
    """
    if node1_id not in depths or node2_id not in depths:
        return None, None, float('inf') # Error case
    if node1_id == node2_id:
        return 0, 0, 0 # Distance from a node to itself
    
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
         return None, None, float('inf') # LCA not found, implies nodes might be in disconnected parts if both in depths
    
    dist_node1_to_lca = depths[node1_id] - depths[lca]
    dist_lca_to_node2 = depths[node2_id] - depths[lca]
    total_distance = dist_node1_to_lca + dist_lca_to_node2 # This is equivalent to depths[node1_id] + depths[node2_id] - 2 * depths[lca]
    
    return dist_node1_to_lca, dist_lca_to_node2, total_distance

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
    Also prepares a list of jump distances for display, including a conceptual
    jump from the root to the first filtered leaf.

    Args:
        tree_data (dict): Dict representing tree {node_id: {"parent": parent_id,...}}
        walk_steps_list (list): List of dicts representing steps [{'from': 'n1', 'to': 'n2', 'category': 'cat'}, ...]

    Returns:
        tuple (float or None, list, list or None, list or None):
            - The computed filtered AJD (based on inter-leaf jumps only).
            - A list of individual jump distance dicts for display.
              The first element may represent root-to-first-leaf.
            - The original full walk sequence.
            - The filtered leaf visit sequence.
        Returns (None, [], None, None) if AJD cannot be computed due to major errors.
        Returns (0.0, display_jumps_list, full_walk_seq, filtered_leaf_seq) if no inter-leaf jumps.
    """

    # --- Step 1: Build tree information ---
    parents, depths, leaves, children, root_id = _build_tree_info_from_parent_links(tree_data)
    display_individual_jump_distances = [] # For display, including root jump

    if parents is None:
        print("Error: Could not process tree data.")
        return None, [], None, None
    
    # --- Step 2: Reconstruct the full walk sequence ---
    full_walk_sequence = _reconstruct_walk_sequence(walk_steps_list)
    if full_walk_sequence is None:
        print("Error: Could not reconstruct walk sequence.")
        return None, display_individual_jump_distances, None, None # display_individual_jump_distances is empty
    if not full_walk_sequence:
         print("Info: Walk sequence is empty.")
         return None, display_individual_jump_distances, [], []

    if not leaves: # No leaves in the tree itself
        print("Warning: No leaf nodes found in the tree. Filtered AJD is undefined.")
        # Add root jump info if root exists, even if no leaves in filtered_leaf_sequence later
        if root_id:
            # This case implies filtered_leaf_sequence will be empty. We add a conceptual root marker.
            # Or, if filtered_leaf_sequence might have non-leaf nodes (not current design), this is more complex.
            # Assuming filtered_leaf_sequence contains only valid leaves from the 'leaves' set.
            # If no leaves, no first_leaf to jump to. So, can't form the root jump as specified.
            pass # display_individual_jump_distances remains empty
        return None, display_individual_jump_distances, full_walk_sequence, []

    # --- Step 3: Filter the leaf visits based on the verification rule ---
    print("Original full walk sequence:", full_walk_sequence) # Debug
    filtered_leaf_sequence = _filter_leaf_visits(full_walk_sequence, walk_steps_list, leaves, depths)
    print("Filtered leaf visit sequence:", filtered_leaf_sequence) # Debug

    # --- Add conceptual jump from root to the first filtered leaf for display ---
    if root_id and filtered_leaf_sequence: # Check if filtered_leaf_sequence is not empty
        first_leaf = filtered_leaf_sequence[0]
        # _get_distance returns (dist_root_to_lca, dist_lca_to_first_leaf, total_dist_root_to_first_leaf)
        # When node1 is root, dist_root_to_lca = 0, dist_lca_to_first_leaf = total_dist_root_to_first_leaf
        _, _, dist_root_to_first_leaf = _get_distance(root_id, first_leaf, parents, depths)
        if dist_root_to_first_leaf != float('inf'):
            root_jump_info = {"From_Root": "None", "Root_To": dist_root_to_first_leaf, "JD": "None"}
        else:
            root_jump_info = {"From_Root": "Error", "Root_To": "Error", "JD": "Error"}
        display_individual_jump_distances.append(root_jump_info)
    elif not filtered_leaf_sequence and root_id: # No filtered leaves, but root exists
        # Optionally, add a marker that no leaves were reached from root for display.
        # For now, display_individual_jump_distances will be empty if filtered_leaf_sequence is empty.
        pass 

    # --- Step 4: Calculate average jump distance on the filtered sequence (inter-leaf jumps) ---
    M_filtered = len(filtered_leaf_sequence)
    actual_num_jumps = M_filtered - 1
    filtered_ajd = 0.0

    if actual_num_jumps < 1: # If M_filtered is 0 or 1, no inter-leaf jumps
        # display_individual_jump_distances might have 0 or 1 element (the root jump)
        return 0.0, display_individual_jump_distances, full_walk_sequence, filtered_leaf_sequence

    sum_distances_for_ajd = 0
    jumps_with_errors_for_ajd = 0

    for i in range(actual_num_jumps):
        u = filtered_leaf_sequence[i]
        v = filtered_leaf_sequence[i+1]

        dist_u_lca, dist_lca_v, total_jd = _get_distance(u, v, parents, depths)

        if total_jd == float('inf'):
             print(f"Warning: Skipping jump ({u} -> {v}) for AJD calculation due to distance error (total_jd is inf).")
             jumps_with_errors_for_ajd += 1
             # Still add an error marker to display list for this jump if desired, or skip
             display_individual_jump_distances.append({
                 "from_leaf_to_lca": "Error", 
                 "lca_to_to_leaf": "Error", 
                 "JD": "Error", 
                 "note": f"Error calculating jump {u}->{v}"
             })
             continue
        if dist_u_lca is None or dist_lca_v is None: # Should be caught by total_jd == inf
            print(f"Warning: Skipping jump ({u} -> {v}) for AJD calculation due to component distance error.")
            jumps_with_errors_for_ajd +=1
            display_individual_jump_distances.append({
                "from_leaf_to_lca": "Error", 
                "lca_to_to_leaf": "Error", 
                "JD": "Error", 
                "note": f"Component error calculating jump {u}->{v}"
            })
            continue
            
        sum_distances_for_ajd += total_jd
        display_individual_jump_distances.append({
            "from_leaf_to_lca": dist_u_lca,
            "lca_to_to_leaf": dist_lca_v,
            "JD": total_jd
        })

    if jumps_with_errors_for_ajd > 0:
         print(f"Warning: {jumps_with_errors_for_ajd} inter-leaf jumps could not be calculated for AJD. AJD may be inaccurate.")

    # Calculate AJD based on successfully calculated inter-leaf jumps
    # Number of jumps attempted for AJD is actual_num_jumps.
    # Number of successful jumps for AJD is actual_num_jumps - jumps_with_errors_for_ajd.
    
    # If all inter-leaf jumps errored, but there were attempts (actual_num_jumps > 0)
    if actual_num_jumps > 0 and (actual_num_jumps - jumps_with_errors_for_ajd == 0):
        filtered_ajd = 0.0 # Or None, or handle as error. Current: 0 if all fail.
    elif actual_num_jumps > 0 : # Some successful inter-leaf jumps
        # Average over successful jumps or over attempted jumps?
        # Current code averages sum_distances_for_ajd / actual_num_jumps. This means errored jumps effectively count as 0 distance if sum_distances_for_ajd wasn't updated.
        # If we only want to average successful jumps:
        # num_successful_jumps = actual_num_jumps - jumps_with_errors_for_ajd
        # if num_successful_jumps > 0: filtered_ajd = sum_distances_for_ajd / num_successful_jumps else: filtered_ajd = 0.0
        # Let's stick to original averaging logic: sum of successful distances / total number of inter-leaf segments
        filtered_ajd = sum_distances_for_ajd / actual_num_jumps
    else: # No inter-leaf jumps (actual_num_jumps == 0), AJD is 0
        filtered_ajd = 0.0
    
    return filtered_ajd, display_individual_jump_distances, full_walk_sequence, filtered_leaf_sequence

def get_analysis(idx, results, results_dir, overwrite=False):
    result_path = f"{results_dir}/tree_vis_v3/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        corr = compute_score(output_str, results.iloc[idx]["reward_model"]["ground_truth"], "box")
        tree_prompt = get_tree_prompt(input_str, output_str)
        tree_json = llm.generate([{
            "role": "user",
            "content": tree_prompt
        }])[2]
        
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
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        corr = compute_score(output_str, results.iloc[idx]["reward_model"]["ground_truth"], "box")
        json_data["corr"] = corr
        
    
    # Filter out the specific walk elements before visualization
    if "walk" in json_data and isinstance(json_data["walk"], list):
        json_data["walk"] = [
            step for step in json_data["walk"]
            if not (step.get("from") == "none" and step.get("to") == "node1")
        ]

    vis_path = visualize_tree_walk(json_data["tree"], json_data["walk"], filename=f"{results_dir}/tree_vis_v3/{idx}", format="pdf")
    filtered_ajd, display_jds_list, original_walk_seq, filtered_leaf_seq = compute_filtered_average_jump_distance(json_data["tree"], json_data["walk"])
    print(f"Index {idx}: Filtered AJD = {filtered_ajd}")
    
    print(f"Index {idx}: Individual Jump Distances (Display Format):")
    if display_jds_list:
        for jd_info in display_jds_list:
            if "From_Root" in jd_info: # Check for the root jump marker
                print(f"  - From_Root: {jd_info.get('From_Root')}, Root_To: {jd_info.get('Root_To')}, JD: {jd_info.get('JD')}")
            elif "note" in jd_info: # Check for error marker from inter-leaf jump calculation
                print(f"  - From_LCA: Error, LCA_To: Error, JD: Error (Note: {jd_info.get('note')})")
            else: # Standard inter-leaf jump
                print(f"  - From_LCA: {jd_info.get('from_leaf_to_lca')}, LCA_To: {jd_info.get('lca_to_to_leaf')}, JD: {jd_info.get('JD')}")
    else:
        print("  - No jump distances to display.")
    
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

    # ---- NEW: Calculate no_calculation_edge ----
    no_calculation_edge_value = 0
    missing_calculation_edges_list = []
    tree_nodes_data = json_data.get("tree", {})
    walk_steps_for_check = json_data.get("walk", []) # Already filtered for "none" -> "node1"

    # Create a set of calculation/derivation walks for efficient lookup: (from_node, to_node)
    calculation_walks_set = set()
    if isinstance(walk_steps_for_check, list):
        for step in walk_steps_for_check:
            if isinstance(step, dict) and step.get("category") == "calculation/derivation":
                calculation_walks_set.add((step.get("from"), step.get("to")))

    if isinstance(tree_nodes_data, dict):
        for node_id, node_info in tree_nodes_data.items():
            parent_id = node_info.get("parent")
            # Check if it's a valid tree edge (parent exists and is not 'none')
            if parent_id and parent_id != "none" and parent_id in tree_nodes_data:
                # This represents a tree edge: parent_id -> node_id
                if (parent_id, node_id) not in calculation_walks_set:
                    no_calculation_edge_value = 1
                    missing_calculation_edges_list.append(f"{parent_id} -> {node_id}")
    
    print(f"Index {idx}: no_calculation_edge = {no_calculation_edge_value}")
    if no_calculation_edge_value == 1:
        print(f"Index {idx}: Missing calculation edges: {', '.join(missing_calculation_edges_list)}")
    # ---- END NEW ----

    return {
        "graph": vis_path,
        "filtered_ajd": filtered_ajd,
        "individual_jump_distances": display_jds_list, # This now correctly refers to the display list
        "average_solution_count": average_solution_count,
        "calculation_count": calculation_count,
        "verification_count": verification_count,
        "backtracking_count": backtracking_count,
        "total_node_count": total_node_count,
        "forgetting_rate": forgetting_rate,
        "average_verification_rate": average_verification_rate,
        "corr": json_data["corr"],
        "no_calculation_edge": no_calculation_edge_value,
        "missing_edges_info": ', '.join(missing_calculation_edges_list) if no_calculation_edge_value == 1 else "",
        "original_full_walk_sequence": original_walk_seq, # Add original walk sequence
        "filtered_leaf_visit_sequence": filtered_leaf_seq, # Add filtered leaf sequence
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs='+', default=[])
    parser.add_argument("--dataset_name", type=str, default="math500", choices=["gsm8k", "math500", "gpqa-diamond"])
    parser.add_argument("--model_name", type=str, nargs='+', default=["deepseek-ai/deepseek-reasoner"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "ricl_1", "ricl_2", "ricl_3", "ricl_4", "ricl_5", "ricl_6", "ricl_7", "ricl_8", "ricl_9", "ricl_10"])
    parser.add_argument("--temperature", type=float, default=0.00)
    args = parser.parse_args()
    
    all_metrics = {}
    collected_jds_for_final_print = [] # NEW: Initialize list to collect JDs across all models and samples

    for model_name in args.model_name:
        if args.wandb:
            wandb_config = {
                "dataset_name": args.dataset_name,
                "model_name": model_name,
                "num_samples": args.num_samples,
                "mode": args.mode,
                "temperature": args.temperature,
            }
            project_name = f"{WANDB_INFO['project']}-tree-vis-v3"
            
            if not wandb_init(project_name, WANDB_INFO["entity"], wandb_config):
                exit()
                
        if args.mode == "default":
            template_type = supported_llms[model_name]["template_type"]
        else:
            template_type = f"{supported_llms[model_name]['template_type']}_{args.mode}"
        results_dir = get_result_dir(
            dataset_name = args.dataset_name,
            model_name = model_name,
            shot = 0,
            template_type = template_type,
            response_length = 404,
            num_samples = args.num_samples,
            feature_noise = supported_datasets[args.dataset_name]["feature_noise"],
            label_noise = 0.0,
            data_mode = "default",
            n_query = 1,
            temperature = args.temperature,
        )
        results = pd.read_parquet(f"{results_dir}/test_default.parquet")
        
        if len(args.idx) == 0:
            idxs = range(len(results))
        else:
            idxs = args.idx
        
        filtered_ajds = []
        average_solution_counts = [] # Initialize list for average solution counts
        calculation_arrow_counts = []
        verification_arrow_counts = []
        backtracking_arrow_counts = []
        total_node_counts = [] # Initialize list for total node counts
        forgetting_rates = [] # Initialize list for forgetting rates
        average_verification_rates_list = [] # Initialize list for average_verification_rate
        forgetting_rate_one_indices = [] # Initialize list for indices with forgetting_rate == 1
        none_ajd_indices = [] # Initialize list for indices with filtered_ajd == None
        corrs = []
        no_calculation_edge_values = [] # For storing 0 or 1 for each sample
        all_samples_missing_edges_info = [] # For storing detailed string info for samples with missing edges
        no_calculation_edge_one_indices = [] # Initialize list for indices with no_calculation_edge == 1
        
        for idx in tqdm(idxs):
            attempts, success, overwrite = 0, False, args.overwrite
            while attempts < 5 and not success:
                try:
                    graph_metric = get_analysis(idx, results, results_dir, overwrite)
                    success = True
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
            
            filtered_ajds.append(graph_metric["filtered_ajd"])
            average_solution_counts.append(graph_metric["average_solution_count"]) # Append average solution count
            calculation_arrow_counts.append(graph_metric["calculation_count"])
            verification_arrow_counts.append(graph_metric["verification_count"])
            backtracking_arrow_counts.append(graph_metric["backtracking_count"])
            total_node_counts.append(graph_metric["total_node_count"]) # Append total node count
            forgetting_rates.append(graph_metric["forgetting_rate"]) # Append forgetting rate
            average_verification_rates_list.append(graph_metric["average_verification_rate"]) # Append average_verification_rate
            corrs.append(graph_metric["corr"])
            
            no_calculation_edge_values.append(graph_metric["no_calculation_edge"])
            if graph_metric["no_calculation_edge"] == 1:
                no_calculation_edge_one_indices.append(idx)
            
            if graph_metric["forgetting_rate"] == 1: # Check if forgetting_rate is 1
                forgetting_rate_one_indices.append(idx) # Add index to the list
            
            if graph_metric["filtered_ajd"] is None: # Check if filtered_ajd is None
                none_ajd_indices.append(idx) # Add index to the list

            # NEW: Collect individual jump distances, AJD, and walk sequences for final printing
            if ("individual_jump_distances" in graph_metric and 
                "filtered_ajd" in graph_metric and
                "original_full_walk_sequence" in graph_metric and # Check for new key
                "filtered_leaf_visit_sequence" in graph_metric and # Check for new key
                "individual_jump_distances" in graph_metric):  # Ensure this key exists for collected_jds_for_final_print
                collected_jds_for_final_print.append((
                    model_name, 
                    idx, 
                    graph_metric["individual_jump_distances"], # This is the display list
                    graph_metric["filtered_ajd"], 
                    graph_metric["original_full_walk_sequence"], 
                    graph_metric["filtered_leaf_visit_sequence"]
                ))
        
        # Print indices with forgetting_rate == 1
        print(f"Indices with forgetting_rate == 1: --idx {' '.join(map(str, forgetting_rate_one_indices))}")     
        
        # Print indices with no_calculation_edge == 1
        print(f"Indices with no_calculation_edge == 1: --idx {' '.join(map(str, no_calculation_edge_one_indices))}")

        # Print indices with filtered_ajd == None
        print(f"Indices with filtered_ajd == None: --idx {' '.join(map(str, none_ajd_indices))}")
        
        # Calculate and print the union of all three lists
        all_problematic_indices = sorted(list(set(forgetting_rate_one_indices + no_calculation_edge_one_indices + none_ajd_indices)))
        print(f"Indices with all: --idx {' '.join(map(str, all_problematic_indices))}")
                
        metric_dict = {
            "filtered_ajd": filtered_ajds,
            "average_solution_count": average_solution_counts,
            "calculation_arrow_counts": calculation_arrow_counts,
            "verification_arrow_counts": verification_arrow_counts,
            "backtracking_arrow_counts": backtracking_arrow_counts,
            "total_node_counts": total_node_counts,
            "forgetting_rates": forgetting_rates,
            "average_verification_rates": average_verification_rates_list,
            "corrs": corrs,
            "no_calculation_edge": no_calculation_edge_values, # Add new metric to dict
        }
        
        metric_df = pd.DataFrame(metric_dict)
        # It's usually better to handle NaNs explicitly or ensure they are not produced for critical metrics.
        # 'no_calculation_edge' should always be 0 or 1, so it won't introduce NaNs itself.
        metric_df = metric_df.dropna(how='any') 
        
        filtered_ajd = np.mean(metric_df["filtered_ajd"]) if "filtered_ajd" in metric_df.columns and not metric_df["filtered_ajd"].empty else np.nan
        print(f"Filtered AJD: {filtered_ajd}")
        
        avg_sol_count = np.mean(metric_df["average_solution_count"]) if "average_solution_count" in metric_df.columns and not metric_df["average_solution_count"].empty else np.nan
        print(f"Average Solution Count: {avg_sol_count}") # Print average solution count
        
        avg_calc_arrows = np.mean(metric_df["calculation_arrow_counts"]) if "calculation_arrow_counts" in metric_df.columns and not metric_df["calculation_arrow_counts"].empty else np.nan
        avg_ver_arrows = np.mean(metric_df["verification_arrow_counts"]) if "verification_arrow_counts" in metric_df.columns and not metric_df["verification_arrow_counts"].empty else np.nan
        avg_back_arrows = np.mean(metric_df["backtracking_arrow_counts"]) if "backtracking_arrow_counts" in metric_df.columns and not metric_df["backtracking_arrow_counts"].empty else np.nan

        print(f"Average Calculation Arrows: {avg_calc_arrows}")
        print(f"Average Verification Arrows: {avg_ver_arrows}")
        print(f"Average Backtracking Arrows: {avg_back_arrows}")
            
        avg_total_nodes = np.mean(metric_df["total_node_counts"]) if "total_node_counts" in metric_df.columns and not metric_df["total_node_counts"].empty else np.nan
        print(f"Average Total Node Count: {avg_total_nodes}") # Print average total_node_count
        
        avg_forgetting_rate = np.mean(metric_df["forgetting_rates"]) if "forgetting_rates" in metric_df.columns and not metric_df["forgetting_rates"].empty else np.nan
        print(f"Average Forgetting Rate: {avg_forgetting_rate}") # Print average forgetting_rate
        
        overall_avg_verification_rate = np.mean(metric_df["average_verification_rates"]) if "average_verification_rates" in metric_df.columns and not metric_df["average_verification_rates"].empty else np.nan
        print(f"Average Verification Rate: {overall_avg_verification_rate:.4f}" if overall_avg_verification_rate is not None and not np.isnan(overall_avg_verification_rate) else "Average Verification Rate: N/A")
            
        avg_corr = np.mean(metric_df["corrs"]) if "corrs" in metric_df.columns and not metric_df["corrs"].empty else np.nan
        print(f"Average Correlation: {avg_corr}")
        
        # Calculate and print average for no_calculation_edge (Proportion of samples with the issue)
        avg_no_calc_edge = np.mean(metric_df["no_calculation_edge"]) if "no_calculation_edge" in metric_df.columns and not metric_df["no_calculation_edge"].empty else np.nan
        print(f"Proportion of samples with no_calculation_edge=1: {avg_no_calc_edge:.4f}" if avg_no_calc_edge is not None and not np.isnan(avg_no_calc_edge) else "Proportion of samples with no_calculation_edge=1: N/A")
                
        if args.wandb:
            wandb_log_data = {
                "filtered_ajd": filtered_ajd,
                "average_solution_count": avg_sol_count, # Log average solution count
                "average_calculation_arrows": avg_calc_arrows,
                "average_verification_arrows": avg_ver_arrows,
                "average_backtracking_arrows": avg_back_arrows,
                "average_total_node_count": avg_total_nodes, # Log average total_node_count
                "average_forgetting_rate": avg_forgetting_rate, # Log average forgetting_rate
                "overall_average_verification_rate": overall_avg_verification_rate, # Log overall_average_verification_rate
                "average_correlation": avg_corr,
                "average_no_calculation_edge": avg_no_calc_edge, # Log new metric
            }
            # Filter out NaN values before logging to wandb
            wandb_log_data = {k: v for k, v in wandb_log_data.items() if v is not None and not np.isnan(v)}
            wandb.log(wandb_log_data)
            wandb.finish()
            
        for metric in metric_dict:
            if not metric in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].extend(metric_dict[metric])
            
    # NEW LOCATION for printing collected JDs
    # Print all collected individual jump distances (length >= 0, as first entry can be root jump)
    print("\n--- Collected Individual Jump Distances (length >= 0, as first entry can be root jump) ---")
    # Adjusted condition for printing, as the display_jds_collected might contain only the root jump
    # The original request was "length >= 2" for the *inter-leaf* jumps.
    # Now, we print if display_jds_collected is not empty, and then iterate through it.
    # The old jds_list_collected (which was for inter-leaf jumps) is no longer directly used for the len check.

    found_any_jds_to_print = False
    for model_name_collected, sample_idx_collected, display_jds_collected_list, filtered_ajd_collected, original_walk_seq_collected, filtered_leaf_seq_collected in collected_jds_for_final_print:
        # Print if there's anything to display for this sample (e.g., at least the root jump or any inter-leaf jumps)
        if display_jds_collected_list: # Check if the list itself is not empty
            print(f"Model: {model_name_collected}, Sample Index: {sample_idx_collected}, Filtered AJD: {filtered_ajd_collected}")
            print(f"  Original full walk sequence: {original_walk_seq_collected}")
            print(f"  Filtered leaf visit sequence: {filtered_leaf_seq_collected}")
            print(f"  Individual Jump Distances (Display Format):")
            for jd_info in display_jds_collected_list:
                if "From_Root" in jd_info:
                    print(f"    - From_Root: {jd_info.get('From_Root')}, Root_To: {jd_info.get('Root_To')}, JD: {jd_info.get('JD')}")
                elif "note" in jd_info:
                    print(f"    - From_LCA: Error, LCA_To: Error, JD: Error (Note: {jd_info.get('note')})")    
                else:
                    print(f"    - From_LCA: {jd_info.get('from_leaf_to_lca')}, LCA_To: {jd_info.get('lca_to_to_leaf')}, JD: {jd_info.get('JD')}")
            found_any_jds_to_print = True
            print("---") # Add a separator between samples for clarity
            
    if not found_any_jds_to_print:
        print("No individual jump distances to display across all samples and models.")

    # Check the importance of each feature using XGBoost regressor
    feature_columns = [
        "filtered_ajd",
        "forgetting_rates",
        "average_verification_rates",
        "average_solution_count",
        "no_calculation_edge"  # Add new feature for model input
    ]
    target_column = "corrs"

    # Prepare X and y, dropping rows with NaN in any feature or target
    valid_rows = metric_df[feature_columns + [target_column]].dropna()
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

        # INSERT_YOUR_CODE
        # Check the accuracy of a majority classifier (predicts the most common class)
        if len(y_test) > 0:
            majority_class = Counter(y_train).most_common(1)[0][0]
            majority_pred = np.full_like(y_test, fill_value=majority_class)
            majority_accuracy = np.mean(majority_pred == y_test)
            print(f"Majority Classifier Accuracy: {majority_accuracy:.4f}")
        else:
            print("Not enough test data to compute majority classifier accuracy.")
        # Check feature importances
        importances = model.feature_importances_
        # Optionally, print sorted feature importances
        sorted_features = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
        print("Features ranked by importance:")
        for name, importance in sorted_features:
            print(f"  {name}: {importance:.4f}")
    else:
        print("Not enough valid data to check feature importance.")
