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
7.  Ensure the output is strictly the JSON list as specified, with no additional explanatory text.
8. The output MUST be perfectly valid JSON, parseable by standard libraries.

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
    json_content = re.sub(r'\\(?![\\\"/bfnrtu])', r'\\\\', json_content)
    
    # Parse the JSON content
    try: 
        data = json.loads(json_content)
    except json.decoder.JSONDecodeError as e:
        pdb.set_trace()
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

def get_analysis(idx, results, results_dir, overwrite=False):
    result_path = f"{results_dir}/tree_vis_v3/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        corr = results.iloc[idx]["answers"][0] == results.iloc[idx]["reward_model"]["ground_truth"]["label"][0]
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
    
    vis_path = visualize_tree_walk(json_data["tree"], json_data["walk"], filename=f"{results_dir}/tree_vis_v3/{idx}", format="pdf")
    filtered_ajd = compute_filtered_average_jump_distance(json_data["tree"], json_data["walk"])
    return {
        "graph": vis_path,
        "filtered_ajd": filtered_ajd,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs='+', default=[])
    parser.add_argument("--dataset_name", type=str, default="math500", choices=["gsm8k", "math500", "gpqa-diamond"])
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-reasoner")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.wandb:
        wandb_config = {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "num_samples": args.num_samples,
        }
        project_name = f"{WANDB_INFO['project']}-tree-vis-v3"
        
        if not wandb_init(project_name, WANDB_INFO["entity"], wandb_config):
            exit()
            
    results_dir = get_result_dir(
        dataset_name = args.dataset_name,
        model_name = args.model_name,
        shot = 0,
        template_type = supported_llms[args.model_name]["template_type"],
        response_length = 404,
        num_samples = args.num_samples,
        feature_noise = supported_datasets[args.dataset_name]["feature_noise"],
        label_noise = 0.0,
        data_mode = "default",
        n_query = 1,
    )
    results = pd.read_parquet(f"{results_dir}/test_default.parquet")
    
    if len(args.idx) == 0:
        idxs = range(len(results))
    else:
        idxs = args.idx
    
    filtered_ajds = []
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
        
    filtered_ajd = sum(filtered_ajds) / len(filtered_ajds)
    print(f"Filtered AJD: {filtered_ajd}")
        


    # print("\n--- XGBoost Analysis ---")
    # # Prepare data for XGBoost
    # # Ensure all lists have the same length and are not empty
    # if len(max_depths) > 0 and len(max_depths) == len(breadths) == len(avg_depths) == len(b2d_ratios) == len(validation_rates) == len(corrs):
    #     X = np.array([max_depths, breadths, avg_depths, b2d_ratios, validation_rates]).T
    #     y = np.array(corrs) # Assuming corrs contains binary correctness labels (e.g., 0 or 1)

    #     # Check if there are at least two classes in the target variable
    #     unique_classes = np.unique(y)
    #     if len(unique_classes) >= 2:
    #         # Check if there are enough samples relative to features
    #         if X.shape[0] > X.shape[1]:
    #             # Import XGBoost
    #             from xgboost import XGBClassifier
                
    #             # Instantiate the XGBoost model
    #             xgb_model = XGBClassifier(
    #                 random_state=42,
    #                 scale_pos_weight=len(y) / sum(y) - 1 if sum(y) > 0 else 1  # For imbalanced classes
    #             )

    #             # Train the model
    #             xgb_model.fit(X, y)

    #             y_pred = xgb_model.predict(X)
    #             accuracy = accuracy_score(y, y_pred)
    #             print(f"\nModel Training Accuracy: {accuracy:.4f}")
    #             print(f"Baseline Accuracy (predicting majority class): {max(np.mean(y), 1 - np.mean(y)):.4f}")
                
    #             # Feature importance
    #             importance = xgb_model.feature_importances_
    #             features = ['max_depth', 'breadth', 'avg_depth', 'b2d_ratio', 'validation_rate']
    #             print("\nFeature Importance:")
    #             for i, feat in enumerate(features):
    #                 print(f"{feat}: {importance[i]:.4f}")

    #         else:
    #             print("Skipping XGBoost: Not enough samples relative to the number of features.")
    #     else:
    #         print(f"Skipping XGBoost: Only one class ({unique_classes[0]}) found in the target variable 'corrs'.")
    # else:
    #     print("Skipping XGBoost: Data lists are empty or have inconsistent lengths.")
    # print("--- End XGBoost Analysis ---\n")
    
    if args.wandb:
        wandb.log({
            "filtered_ajd": filtered_ajd,
        })
        wandb.finish()