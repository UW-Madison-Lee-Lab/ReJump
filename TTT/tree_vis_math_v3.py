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
from environment import root_dir

import numpy as np
from collections import deque

from verl.utils.reward_score.math500 import compute_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression

# model = "xai/grok-3-mini-beta"
# model = "claude/claude-3-7-sonnet-20250219-thinking"
model_pro = "google/gemini-2.5-pro-preview-03-25"
llm_pro = LLMAPI(
    api_key=supported_llms[model_pro]["api_key"],
    model_name=model_pro,
    template_type="reasoning_api"
)

model_flash_parsing = "google/gemini-2.5-flash-preview-04-17"
llm_flash_for_parsing = LLMAPI(
    api_key=supported_llms[model_flash_parsing]["api_key"],
    model_name=model_flash_parsing,
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
    
    
def get_result_parsing_and_comparison_prompt(result_string, ground_truth_string):
    return f"""You are an expert AI assistant. Your task is to analyze a 'Result' string from a mathematical reasoning step and compare its final numerical answer to a 'Ground Truth' value.

Instructions:
1.  Extract the final numerical value(s) from the 'Result' string. 
    - If multiple numbers are present, focus on the one that seems to be the conclusive answer of that step.
    - Handle approximations (e.g., "approx 46.0", "is about 3.14").
    - If the result explicitly states abandonment (e.g., "[Path abandoned]"), extract the numerical value derived *before* abandonment, if any. If no clear numerical value was derived, use "N/A" for the parsed value.
    - If no specific numerical answer can be clearly identified, use "N/A" for the parsed value.

2.  Compare the extracted numerical value with the 'Ground Truth' value.
    - The comparison should determine if they are essentially the same, considering potential minor differences in formatting or precision (e.g., "46" vs "46.0", "1.03" vs "1.035" if context implies rounding).
    - If the parsed value is "N/A", the comparison result should be "NOT_APPLICABLE".
    - If the ground truth is empty or clearly not a comparable numerical value, and the parsed value is numerical, consider it a "MISMATCH" unless specified otherwise.

3.  Output a single JSON object with two keys:
    -   `"parsed_value"`: The extracted numerical value as a string (e.g., "46", "3.14", "N/A").
    -   `"match_status"`: A string indicating the comparison result. Must be one of: "MATCH", "MISMATCH", "NOT_APPLICABLE".

Example:
Result string: "Using the approximations, $tan x^\circ \\approx \\frac{{1.3270 + 6.3138}}{{1.3270 \\times 6.3138 - 1}} \\approx \\frac{{7.6408}}{{8.381 - 1}} \\approx \\frac{{7.6408}}{{7.381}} \\approx 1.0355$. This implies $x \\approx arctan(1.0355) \\approx 46.0^\circ$. [Path abandoned]"
Ground Truth string: "46"
Expected JSON Output: {{"parsed_value": "46.0", "match_status": "MATCH"}}

Result string: "The answer is $y=3$."
Ground Truth string: "3.0"
Expected JSON Output: {{"parsed_value": "3", "match_status": "MATCH"}}

Result string: "The calculation leads to $10/2 = 5$. However, this path is incorrect."
Ground Truth string: "7"
Expected JSON Output: {{"parsed_value": "5", "match_status": "MISMATCH"}}

Result string: "[Path abandoned] No value obtained."
Ground Truth string: "10"
Expected JSON Output: {{"parsed_value": "N/A", "match_status": "NOT_APPLICABLE"}}

---
Result string to analyze:
{result_string}

Ground Truth value:
{ground_truth_string}
---

JSON Output:"""


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
    Computes the number of leaf nodes in the tree and returns their IDs.

    Args:
        tree_data (dict): Dict representing tree {node_id: {"parent": parent_id,...}}
        walk_steps_list (list): List of dicts representing steps (unused in this function but kept for consistency).

    Returns:
        tuple (int or None, set or None): The number of leaf nodes and a set of their IDs,
                                         or (None, None) if tree data is invalid.
    """
    parents, depths, leaves, children, root_id = _build_tree_info_from_parent_links(tree_data)

    if parents is None:
        print("Error: Could not process tree data for solution count.")
        return None, None

    if not leaves:
        # This could mean no reachable leaf nodes or an empty tree.
        # Depending on definition, 0 might be more appropriate than None if tree is valid but has no leaves.
        print("Info: No leaf nodes found or tree is empty. Returning 0 solutions.")
        return 0, set()

    return len(leaves), leaves


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

def check_leaf_node(
    sorted_leaf_ids,
    ground_truth_value, 
    tree_json
):
    for leaf_id in sorted_leaf_ids:
        if leaf_id in tree_json and "Result" in tree_json[leaf_id]:
            parsed_value_text = "N/A (processing error)"
            match_corr = 0.0 # Default to 0.0 for correlation
            parsing_comparison_prompt = get_result_parsing_and_comparison_prompt(tree_json[leaf_id]["Result"], ground_truth_value)
            llm_response_raw = llm_flash_for_parsing.generate([{"role": "user", "content": parsing_comparison_prompt}])
            
            llm_output_str = llm_response_raw[2].strip() if len(llm_response_raw) > 2 and isinstance(llm_response_raw[2], str) else ""
            response_json = parse_json(llm_output_str)
            parsed_value_text = response_json.get("parsed_value", "N/A (LLM missing parsed_value)")
            match_status = response_json.get("match_status", "N/A (LLM missing match_status)")
            
            if match_status == "MATCH":
                match_corr = 1.0
            elif match_status == "MISMATCH":
                match_corr = 0.0
            else:
                match_corr = 0.0
        else:
            parsed_value_text = "N/A (processing error)"
            match_corr = 0.0
                
        tree_json[leaf_id]["parsed_value"] = parsed_value_text
        tree_json[leaf_id]["match_corr"] = match_corr
        
    return tree_json


def get_analysis(idx, results, results_dir, overwrite=False, corr_constraint = None):
    result_path = f"{results_dir}/tree_vis_v3/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        corr = compute_score(output_str, results.iloc[idx]["reward_model"]["ground_truth"], "box")
        tree_prompt = get_tree_prompt(input_str, output_str)
        tree_json = llm_pro.generate([{
            "role": "user",
            "content": tree_prompt
        }])[2]
        tree_json = parse_json(tree_json)
        
        walk_prompt = get_walk_prompt(input_str, output_str, tree_json)
        walk_json = llm_pro.generate([{
            "role": "user",
            "content": walk_prompt
        }])[2]
        walk_json = parse_json(walk_json)
        
        solution_count, leaf_node_ids = compute_average_solution_count(tree_json, walk_json)
        sorted_leaf_ids = sorted(list(leaf_node_ids)) if leaf_node_ids is not None else []
        if solution_count is not None and solution_count > 1:
            
            tree_json = check_leaf_node(
                sorted_leaf_ids, 
                output_str,  
                tree_json
            )
                
        json_data = {
            "tree": tree_json,
            "walk": walk_json,
            "corr": corr,
        }
        save_json(json_data, result_path)
    else:
        json_data = load_json(result_path)
        tree_json = json_data["tree"]
        walk_json = json_data["walk"]
        corr = json_data["corr"]
        
    if corr_constraint is not None:
        if corr != corr_constraint:
            return None
    
    # Filter out the specific walk elements before visualization
    if "walk" in json_data and isinstance(json_data["walk"], list):
        json_data["walk"] = [
            step for step in json_data["walk"]
            if not (step.get("from") == "none" and step.get("to") == "node1")
        ]

    vis_path = visualize_tree_walk(json_data["tree"], json_data["walk"], filename=f"{results_dir}/tree_vis_v3/{idx}", format="pdf")
    filtered_ajd = compute_filtered_average_jump_distance(json_data["tree"], json_data["walk"])
    print(f"Index {idx}: Filtered AJD = {filtered_ajd}")

    all_samples_leaf_node_parsed_corrs = []
    
    solution_count, leaf_node_ids = compute_average_solution_count(tree_json, walk_json)
    
    if solution_count is not None:
        sorted_leaf_ids = sorted(list(leaf_node_ids)) if leaf_node_ids is not None else []
        if solution_count == 1 and sorted_leaf_ids:
            all_samples_leaf_node_parsed_corrs.append(float(corr))

        elif solution_count > 1 and sorted_leaf_ids:
            for leaf_id in sorted_leaf_ids:
                if "match_corr" in tree_json[leaf_id]:
                    match_corr = tree_json[leaf_id]["match_corr"]
                    all_samples_leaf_node_parsed_corrs.append(float(match_corr))
                else:
                    all_samples_leaf_node_parsed_corrs.append(0.00)
                
        current_sample_success_rate = 0.0 # Default for empty or error cases
        if all_samples_leaf_node_parsed_corrs: # Check if the list is not empty
            current_sample_success_rate = sum(c == 1.0 for c in all_samples_leaf_node_parsed_corrs) / len(all_samples_leaf_node_parsed_corrs)
   
        current_sample_overthinking_rate = 0.0
        if all_samples_leaf_node_parsed_corrs:
            try:
                first_one_index = all_samples_leaf_node_parsed_corrs.index(1.0)
                elements_after_first_one = len(all_samples_leaf_node_parsed_corrs) - 1 - first_one_index
                current_sample_overthinking_rate = elements_after_first_one / len(all_samples_leaf_node_parsed_corrs)
            except ValueError: # No 1.0 found in the list
                current_sample_overthinking_rate = 0.0
                
    else:
        current_sample_success_rate = 0.0
        current_sample_overthinking_rate = 0.0
        
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

    return {
        "graph": vis_path,
        "filtered_ajd": filtered_ajd,
        "average_solution_count": solution_count,
        "calculation_count": calculation_count,
        "verification_count": verification_count,
        "backtracking_count": backtracking_count,
        "total_node_count": total_node_count,
        "forgetting_rate": forgetting_rate,
        "average_verification_rate": average_verification_rate,
        "corr": json_data["corr"],
        "no_calculation_edge": no_calculation_edge_value,
        "missing_edges_info": ', '.join(missing_calculation_edges_list) if no_calculation_edge_value == 1 else "",
        "success_rate": current_sample_success_rate,
        "overthinking_rate": current_sample_overthinking_rate,
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
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.00])
    parser.add_argument("--corr_constraint", type=lambda x: None if x == "None" else int(x), default=None, choices=[None, 0, 1])
    parser.add_argument("--replicate_id", type=int, default=0)
    args = parser.parse_args()
    
    models = args.model_name
    if len(args.temperature) == 1:
        temperatures = [args.temperature[0] for _ in models]
    else:
        if len(args.temperature) != len(models):
            raise ValueError(f"Number of temperatures ({len(args.temperature)}) must match number of models ({len(models)})")
        temperatures = args.temperature
    
    all_metrics = {}
    for model_name, temperature in zip(models, temperatures):
        if args.wandb:
            wandb_config = {
                "dataset_name": args.dataset_name,
                "model_name": model_name,
                "num_samples": args.num_samples,
                "mode": args.mode,
                "temperature": temperature,
                "replicate_id": args.replicate_id,
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
            temperature = temperature,
            replicate_id = args.replicate_id,
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
        all_samples_success_rates = []
        all_samples_overthinking_rates = []
        
        for idx in tqdm(idxs):
            attempts, success, overwrite, skip = 0, False, args.overwrite, False
            while attempts < 5 and not success:
                try:
                    graph_metric = get_analysis(idx, results, results_dir, overwrite, args.corr_constraint)
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
            average_solution_counts.append(graph_metric["average_solution_count"]) # Append average solution count
            calculation_arrow_counts.append(graph_metric["calculation_count"])
            verification_arrow_counts.append(graph_metric["verification_count"])
            backtracking_arrow_counts.append(graph_metric["backtracking_count"])
            total_node_counts.append(graph_metric["total_node_count"]) # Append total node count
            forgetting_rates.append(graph_metric["forgetting_rate"]) # Append forgetting rate
            average_verification_rates_list.append(graph_metric["average_verification_rate"]) # Append average_verification_rate
            corrs.append(graph_metric["corr"])
            all_samples_success_rates.append(graph_metric["success_rate"])
            all_samples_overthinking_rates.append(graph_metric["overthinking_rate"])
            
            no_calculation_edge_values.append(graph_metric["no_calculation_edge"])
            if graph_metric["no_calculation_edge"] == 1:
                no_calculation_edge_one_indices.append(idx)
            
            if graph_metric["forgetting_rate"] == 1: # Check if forgetting_rate is 1
                forgetting_rate_one_indices.append(idx) # Add index to the list
            
            if graph_metric["filtered_ajd"] is None: # Check if filtered_ajd is None
                none_ajd_indices.append(idx) # Add index to the list
        
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
            "success_rates": all_samples_success_rates,
            "overthinking_rates": all_samples_overthinking_rates,
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
                
        avg_success_rate = np.mean(metric_df["success_rates"]) if "success_rates" in metric_df.columns and not metric_df["success_rates"].empty else np.nan
        print(f"Average Success Rate: {avg_success_rate:.4f}" if avg_success_rate is not None and not np.isnan(avg_success_rate) else "Average Success Rate: N/A")
        
        avg_overthinking_rate = np.mean(metric_df["overthinking_rates"]) if "overthinking_rates" in metric_df.columns and not metric_df["overthinking_rates"].empty else np.nan
        print(f"Average Overthinking Rate: {avg_overthinking_rate:.4f}" if avg_overthinking_rate is not None and not np.isnan(avg_overthinking_rate) else "Average Overthinking Rate: N/A")
        
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
                "average_success_rate": avg_success_rate,
                "average_overthinking_rate": avg_overthinking_rate,
            }
            # Filter out NaN values before logging to wandb
            wandb_log_data = {k: v for k, v in wandb_log_data.items() if v is not None and not np.isnan(v)}
            wandb.log(wandb_log_data)
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
        # "success_rates",
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


        # Compose the output dictionary
        summary = {
            "classifier_accuracy": float(accuracy),
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
