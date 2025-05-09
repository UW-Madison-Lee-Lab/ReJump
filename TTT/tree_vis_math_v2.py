import json
from google import genai 
from google.genai import types
import re
import graphviz
import argparse
import pandas as pd
import os
from tqdm import tqdm
from constants import get_result_dir, supported_datasets
from utils import save_json, load_json, wandb_init
import pdb
from collections import defaultdict, deque
from openai import OpenAI

from verl.utils.llm_api import LLMAPI
from constants import supported_llms
import wandb
from environment import WANDB_INFO

import numpy as np
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

def get_divide_prompt(input_str, output_str):
    return f"""
    **Prompt: Divide Reasoning into Sequential Steps**

    **Objective:** To process a given reasoning text and divide it into a sequence of distinct steps, capturing the flow of thought.

    **Your Task:**
    Read the `REASONING_TEXT` provided below, which explains the thinking process for solving the `PROBLEM_DESCRIPTION`. Your goal is to break this reasoning text down into sequential, logical steps. For each step you identify:
    1.  Assign a unique `step` number, starting from 1.
    2.  Write a brief `summary` describing the main point or action of that step.
    3.  Include the exact segment of the original `text` that constitutes this step.

    **Inputs You Will Receive:**
    * `PROBLEM_DESCRIPTION`: The text of the problem.
    * `REASONING_TEXT`: The text containing the step-by-step reasoning.

    **Required Output Format:**
    You must output ***only*** a single JSON object. Do not add any explanations before or after the JSON. The JSON object should contain a list called `steps`. Each item in the list represents one step and must strictly follow this structure:

    ```json
    {{
        "steps": [
            {{
                "step": 1,
                "summary": "Example summary for step 1",
                "text": "Example text segment for step 1"
            }},
            {{
                "step": 2,
                "summary": "Example summary for step 2",
                "text": "Example text segment for step 2"
            }}
            // ... more step objects covering the entire reasoning text
        ]
    }}
    ```
    Instructions Summary:

    Divide the entire REASONING_TEXT into consecutive, meaningful steps based on the flow of reasoning. For each step, provide the step number, a concise summary, and the corresponding text segment. Ensure the output is a single, valid JSON object matching the specified format exactly.

    --- START OF INPUT ---
    PROBLEM_DESCRIPTION:
    {input_str}

    REASONING_TEXT:
    {output_str}
    --- END OF INPUT ---
    """
    
def get_tree_prompt(parsed_steps):
    return f"""
    **Prompt: Analyze Reasoning, Cluster Steps, and Determine Logical Flow Tree (Two-Part Output)**

    **Objective:** To process a sequence of reasoning steps, identify the underlying core logical functions (including distinct approaches), group the steps accordingly, determine the tree structure representing the exploration of these approaches and the successful path, and output these results as distinct parts within a single JSON object.

    **Your Task:**
    You will receive a JSON object containing a list of sequential reasoning steps (`steps`), where each step includes a `step` number, `summary`, and `text`. Your comprehensive task is to perform the following analysis in sequence:

    1.  **Identify & Define Logical Functions/Approaches:**
        * Analyze all input steps to identify the distinct logical functions or phases performed. Pay attention to explicit mentions of different **Approaches**.
        * For each distinct function or phase within an approach (e.g., Stating Approach 1, Executing Approach 1, Evaluating Approach 1), create a concise descriptive label and assign a unique code (`A1`, `A2`, `B1`, `C1`, etc.). Use your judgment to decide the granularity needed to represent the reasoning accurately.

    2.  **Cluster Steps by Function:**
        * Classify each step from the input `steps` list according to the logical functions/codes you defined in Task 1.
        * Group the `step` numbers based on the function code they are assigned to, ensuring every input step is assigned to a cluster.
        * [MOST IMPORTANT] Group verification or re-calculation steps into the *same cluster* as the original steps they verify or calculate. For instance, if step 1 calculates a sum (part of cluster `A1`) and step 3 verifies that sum, step 3 must also be included in cluster `A1`'s step list (e.g., `"A1": {{ "description": "Calculate Sum", "steps": [1, 3] }}`). Similarly, if step 2 calculates a product (part of cluster `A2`) and step 4 verifies it, step 4 must be included in cluster `A2`'s step list (e.g., `"A2": {{ "description": "Calculate Product", "steps": [2, 4] }}`). Avoid creating separate nodes/clusters solely for these verification steps. Therefore, for all nodes, the name shouldn't even contains "Verify" or "Recalculate".
        * Identify which step numbers specifically perform validation, verification, or re-calculation. Populate a `validation_steps` list for each cluster containing these step numbers. If a cluster contains no such steps, use an empty list `[]`

    3.  **Determine Natural Logical Flow (Tree Structure):**
        * **Goal:** Represent the reasoning's flow as a tree (`logical_flow`) reflecting sequential dependencies and branching points naturally present in the text. **Crucially, the final structure MUST be a single connected graph originating from the `root`.** Every node code defined in `clustering_results` must be included in the `logical_flow` structure (either as a key or within a value list) and be reachable from the `root`.
        * **Root Node:** The `"root"` node's value is a list containing the code(s) for the *very first* distinct logical phase(s).
        * **Successors & Connectivity:** For every node code `X` you identified:
            * Determine which code(s) `Y` represent functions/phases that directly follow from, build upon, or are the **immediate next logical step(s) in the reasoning narrative** compared to phase `X`.
            * **The primary guide for linking `Y` after `X` must be the sequence, context, and dependencies presented in the reasoning text.** If `Y`'s steps occur chronologically after `X`'s steps and relate to them or the overall goal, without indicating a branch starting from an earlier point, `Y` should generally be linked as a successor to `X` or another closely preceding, relevant node.
            * **Crucially, every node code defined** (except the root) **must be placed as a successor to its most logical predecessor node based on the narrative flow.** No nodes should be left disconnected. Actively find the most plausible connection point for all identified phases (including formatting, reflection, justification, etc.) based on where they appear and what they relate to in the text.
            * List the identified direct successor codes `Y` as the value for key `X`. Example: `"X": ["Y1"]`.
        * **Branching:** If multiple distinct phases (`Y`, `Z`) start after the *same preceding phase* `X`, then `X` is a branching point: `"X": ["Y", "Z"]`.
        * **Abandoned Branches:** Handle correctly, ending the abandoned path with `[]` and branching the new approach from the relevant common *prior* node.
        * **Successful Path:** The final node code of the branch leading to the successful conclusion must have `["solution"]` as its value. (Note: The node pointing to `solution` will be the one representing the very last step in the successful reasoning sequence according to the text).
        
    **Input:**
    * `STEP_1_OUTPUT_JSON`: A JSON object containing a list called `steps`. Each element has `step`, `summary`, and `text`. (Example structure as before).

    **Required Output Format:**
    Produce ***only*** a single JSON object containing exactly two top-level keys: `clustering_results` and `logical_flow`. Adhere strictly to the following structure:

    ```json
    {{
        "clustering_results": {{
            // Part 1 Output: Clustering Information
            // Keys: Uppercase letter codes (A1, A2, B1...) identified from Task 1.
            // Values: Objects containing:
            //    "description": (string) Provide a concise, GENERALIZED, and ABSTRACT descriptive label for the overall function performed by the steps in this cluster. This description MUST capture the mathematical or logical role and MUST NOT contain specific variable names (like 'a', 'x') or minor phrasing copied directly from the text. Examples: 'Calculate Symmetric Distance', 'Simplify Expression`. Ensure the description accurately reflects the purpose of ALL steps grouped under this code.
            //    "steps": (list of integers) All step numbers performing this function (from Task 2, including merged validation steps).
            //    "validation_steps": (list of integers) Step numbers *from the "steps" list above* that specifically perform validation/verification/re-calculation (from Task 2). Empty list `[]` if none.
            // Example (content depends on analysis):
            // "A1": {{
            //    "description": "Approach 1: Calculate Addition",
            //    "steps": [1, 4],
            //    "validation_steps": [4] // Step 4 verifies the addition in step 1
            // }},
            // "A2": {{
            //    "description": "Approach 1: Simplify Expression",
            //    "steps": [2, 3, 5],
            //    "validation_steps": [5] // Step 5 re-calculates or verifies step 2 or 3
            // }},
            // "A3": {{
            //    "description": "Approach 1: Calculate Final Result",
            //    "steps": [6],
            //    "validation_steps": [] // No specific validation steps in this cluster
            // }},
            // ... etc. for B1, C1...
        }},
        "logical_flow": {{
            // Part 3 Output: Tree Flow Information
            // Structure determined naturally based on dependencies and branching.
            // Example 1 (Natural structure with Setup S1):
            // "root": ["S1"],
            // "S1": ["A1", "B1"], // Approaches A and B both follow Setup S1
            // "A1": ["A2"],
            // "A2": [], // Branch A abandoned
            // "B1": ["B2"], // Branch B continues
            // "B2": ["solution"]

            // Example 2 (Natural structure, two parallel initial approaches):
            // "root": ["A1", "B1"],
            // "A1": ["A2"],
            // "A2": [],
            // "B1": ["B2"],
            // "B2": ["solution"]

             // Example 3 (Natural structure, simple linear flow):
            // "root": ["P1"],
            // "P1": ["P2"],
            // "P2": ["P3"],
            // "P3": ["solution"]
        }}
    }}
    Instructions Summary:

    Perform the three tasks (Identify/Define Functions, Cluster Steps, Determine Flow Tree) based on the input steps.
    Populate the clustering_results section. Ensure all input steps are clustered.
    Populate the logical_flow section using the exact tree structure specified, including the "root" key and ["solution"] endpoint for the successful path. Failed paths should end with [].
    Ensure the codes (A1, B1, etc.) are consistent across both output sections.
    Output only the single JSON object matching the specified two-part structure exactly. No additional text.
    [MOST IMPORTANT] Group verification or re-calculation steps into the *same cluster* as the original steps they verify or calculate. For instance, if step 1 calculates a sum (part of cluster `A1`) and step 3 verifies that sum, step 3 must also be included in cluster `A1`'s step list (e.g., `"A1": {{ "description": "Calculate Sum", "steps": [1, 3] }}`). Similarly, if step 2 calculates a product (part of cluster `A2`) and step 4 verifies it, step 4 must be included in cluster `A2`'s step list (e.g., `"A2": {{ "description": "Calculate Product", "steps": [2, 4] }}`). Avoid creating separate nodes/clusters solely for these verification steps. Therefore, for all nodes, the name shouldn't even contains "Verify" or "Recalculate".
    
    --- START OF INPUT ---
    STEP_1_OUTPUT_JSON:
    {parsed_steps}
    --- END OF INPUT ---
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
    
    # Parse the JSON content
    data = json.loads(json_content)
    
    return data 


# Calculate depth for each node (depth from leaves, leaves=0)

def calculate_depth(logical_flow, target_node_id):
    """
    Calculates the depth of a given node within a logical flow tree.

    The depth is defined as the number of edges on the shortest path from
    the conceptual root to the node.
    - The conceptual 'root' node itself has a depth of 0.
    - Nodes listed in logical_flow['root'] have a depth of 1.
    - The 'solution' node's depth depends on the path leading to it.

    Args:
        logical_flow (dict): A dictionary representing the tree structure.
                             Expected format includes a 'root' key listing
                             starting nodes, and keys for each node mapping
                             to a list of its direct children.
        target_node_id (str): The ID of the node whose depth is required
                              (e.g., 'root', 'A1', 'B2', 'solution').

    Returns:
        int: The depth of the target_node_id. Returns 0 for 'root'.
             Returns -1 if the node is not found or unreachable from the root,
             or if the logical_flow structure is invalid/empty.
    """

    # Handle the base case for the conceptual root
    if target_node_id == 'root':
        return 0

    # Check if logical_flow is valid and has a root
    if not isinstance(logical_flow, dict) or 'root' not in logical_flow:
        print("Error: Invalid logical_flow dictionary provided.")
        return -1 # Indicate error or invalid input

    start_nodes = logical_flow.get('root', [])

    # If the target is one of the immediate children of the root
    if target_node_id in start_nodes:
        return 1

    # Initialize BFS queue with starting nodes and their depth (1)
    queue = deque()
    visited = set()

    for node in start_nodes:
        # Ensure start nodes exist as keys in the flow, unless they are terminal
        # No need to strictly check here, .get below handles missing keys
        if node not in visited:
             # Add node and its depth (1) to the queue
            queue.append((node, 1))
            visited.add(node)

    # Perform BFS
    while queue:
        current_node, current_depth = queue.popleft()

        # Get children of the current node
        # Use .get() to handle cases where a node might not have children listed
        # (e.g., terminal nodes of failed branches)
        children = logical_flow.get(current_node, [])

        for child in children:
            # Check if the child is the target node
            if child == target_node_id:
                return current_depth + 1

            # If the child hasn't been visited and is not 'solution'
            # (We don't need to explore further *from* 'solution')
            if child != 'solution' and child not in visited:
                # Ensure child node exists as a key if it's not terminal
                # Again, .get() handles this implicitly later, but good practice
                visited.add(child)
                queue.append((child, current_depth + 1))

    # If the loop finishes and the target node was not found
    print(f"Warning: Node '{target_node_id}' not found or unreachable.")
    return -1

def create_flowchart_from_dict(data):
    """
    Create a Graphviz graph from a dictionary containing clustering results and logical flow.
    
    Args:
        data (dict): Dictionary containing 'clustering_results' and 'logical_flow' keys
        trace_path (list): Optional list of node IDs representing the trace path through the graph
                          (not used in this implementation)
        
    Returns:
        graphviz.Digraph: A directed graph visualization
        int: Maximum depth of the tree
        int: Breadth (number of leaf nodes) of the tree
        float: Average depth of leaf nodes
    """
    # Extract data
    clustering_results = data['clustering_results']
    logical_flow = data['logical_flow']
    
    # Create a directed graph
    dot = graphviz.Digraph(comment='Logical Flow Diagram')
    dot.attr(rankdir='TB', size='12,12', fontname='Arial', ranksep='0.75')
    
    # Define basic node styling
    dot.attr('node', shape='ellipse', fontname='Arial', fontsize='12', 
             margin='0.2,0.1', width='1.2', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Create nodes
    for node_id, node_data in clustering_results.items():
        label = f"{node_id}: {node_data['description']}"
        dot.node(node_id, label=label)
    
    # Add root node if it exists in logical flow
    if 'root' in logical_flow:
        dot.node('root', 'root', shape='ellipse')
    
    # Add solution node if referenced in logical flow
    solution_referenced = any('solution' in targets for targets in logical_flow.values())
    if solution_referenced:
        dot.node('solution', 'Solution', shape='ellipse')
    
    # Add edges according to logical flow
    for source, targets in logical_flow.items():
        for target in targets:
            dot.edge(source, target)
    
    # Calculate max depth and breadth
    max_depth = 0
    leaf_nodes = set()
    
    # Find all nodes that have outgoing edges
    nodes_with_outgoing = set()
    for source, targets in logical_flow.items():
        nodes_with_outgoing.add(source)
        if not targets:  # Empty target list means leaf node
            leaf_nodes.add(source)
    
    # Add nodes that appear as targets but don't have outgoing edges
    for source, targets in logical_flow.items():
        for target in targets:
            if target not in nodes_with_outgoing:
                leaf_nodes.add(target)
    
    
    # Breadth is the number of leaf nodes
    breadth = len(leaf_nodes)
    
    # Calculate average depth of leaf nodes
    total_depth, max_depth = 0, 0
    if leaf_nodes and 'root' in logical_flow:
        for leaf in leaf_nodes:
            leaf_depth = calculate_depth(logical_flow, leaf)
            total_depth += leaf_depth
            max_depth = max(max_depth, leaf_depth)
        avg_depth = total_depth / len(leaf_nodes) if leaf_nodes else 0
    else:
        avg_depth = 0
    
    return dot, max_depth, breadth, avg_depth

def add_highlighted_path(graph, path_nodes, color='red', penwidth='2.0'):
    """
    Adds a highlighted path (sequence of edges) to a graphviz graph object.

    This version attempts to create a smoother appearance by removing arrowheads
    and relying on the graph's spline settings (e.g., setting graph.attr(splines='curved')).
    It also highlights the node boundaries in the path with the same color.
    The start node is filled with green and the end node is filled with red.

    Args:
      graph: The graphviz.Graph or graphviz.Digraph object to modify.
             This object is modified in place.
      path_nodes: A list or tuple of node names (strings) representing the
                  ordered sequence of nodes in the desired path.
      color: The color to use for the highlighted path edges and node boundaries (default: 'red').
      penwidth: The thickness of the highlighted path edges (default: '2.0').
    """
    # Check if the path has at least two nodes to form an edge
    if not path_nodes or len(path_nodes) < 2:
        print("Warning: Path sequence is too short. Needs at least two nodes to draw an edge.")
        return

    print(f"Adding highlighted path (no arrows): {' -> '.join([node for node, node_type in path_nodes])}")

    # Highlight all nodes in the path (just the boundary, not filled)
    for i, (node, node_type) in enumerate(path_nodes):
        if i == 0:  # Start node
            graph.node(node, color='#0077b6', penwidth=str(penwidth))
        elif i == len(path_nodes) - 1:  # End node
            graph.node(node, color='#023e8a', penwidth=str(penwidth))
        else:  # Middle nodes
            graph.node(node, color=color, penwidth=str(penwidth))

    # Iterate through the path sequence to add edges between consecutive nodes
    for i in range(len(path_nodes) - 1):
        u, u_type = path_nodes[i]
        v, v_type = path_nodes[i+1]
        
        if u == v and (v_type != "val"):
            continue
        # Add the edge with styling, constraint=false, and no arrowhead
        graph.edge(u, v,
                   color=color,
                   penwidth=str(penwidth),
                   constraint='false',
                   arrowhead='none') # *** Added to remove arrowheads ***
    return graph

def get_node_visit_order(data):
    clustering_results = data.get('clustering_results', {})
    if not clustering_results:
        return [] 
    

    # 1. Create a mapping from step number to list of function codes
    step_to_nodes = {}
    all_steps = []
    for node_code, details in clustering_results.items():
        steps = details.get('steps', [])
        if not isinstance(steps, list):
            # Handle potential malformed data if 'steps' is not a list
             continue
        for step in steps:
            if isinstance(step, int):
                step_to_nodes[step] = node_code
                all_steps.append(step)
            # else: handle non-integer steps if necessary, currently ignored

    if not all_steps:
        return [] # No valid steps found

    # 2. Find the minimum and maximum step number
    min_step = min(all_steps)
    max_step = max(all_steps)

    # 3. Generate the trace by iterating through steps chronologically
    trace = []
    for step_num in range(min_step, max_step + 1):
        validation_steps = data["clustering_results"][step_to_nodes[step_num]]["validation_steps"]
        nodes_for_step = step_to_nodes[step_num]
        if step_num in validation_steps:
            trace.append((nodes_for_step, "val"))
        else:
            trace.append((nodes_for_step, "calc"))
    return trace

def compute_validation_rate(visit_order):
    validation_steps = 0
    for i, (node, node_type) in enumerate(visit_order):
        if node_type == "val":
            validation_steps += 1
    return validation_steps / len(visit_order)

def get_graph(idx, results, results_dir, overwrite=False):
    result_path = f"{results_dir}/tree_vis/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        pdb.set_trace()
        corr = results.iloc[idx]["answers"][0] == results.iloc[idx]["label"]
        divide_prompt = get_divide_prompt(input_str, output_str)
        parsed_steps = llm.generate([{
            "role": "user",
            "content": divide_prompt
        }])[2]
        
        tree_prompt = get_tree_prompt(parsed_steps)
        output_json = llm.generate([{
            "role": "user",
            "content": tree_prompt
        }])[2]
        json_data = parse_json(output_json)
        save_json({"parsed_steps": parsed_steps, "visualization": json_data}, result_path)
    else:
        json_data = load_json(result_path)["visualization"]
    
    visit_order = get_node_visit_order(json_data)
    corr = results.iloc[idx]["answers"][0] == results.iloc[idx]["reward_model"]["ground_truth"]["label"][0] 
    graph, max_depth, breadth, avg_depth = create_flowchart_from_dict(json_data)
    graph = add_highlighted_path(graph, visit_order, color='#0077b6', penwidth='2.5')
    if graph is not None:
        graph.render(f"{results_dir}/tree_vis/{idx}", format="pdf")
    
    return {
        "graph": graph,
        "max_depth": max_depth,
        "breadth": breadth,
        "avg_depth": avg_depth,
        "validation_rate": compute_validation_rate(visit_order),
        "corr": corr,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs='+', default=[])
    parser.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k", "math500", "gpqa-diamond"])
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-reasoner")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    if args.wandb:
        wandb_config = {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "num_samples": args.num_samples,
        }
        project_name = f"{WANDB_INFO['project']}-tree-vis"
        
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
        temperature = 0.00,
    )
    results = pd.read_parquet(f"{results_dir}/test_default.parquet")
    
    if len(args.idx) == 0:
        idxs = range(len(results))
    else:
        idxs = args.idx
    
    max_depths, breadths, avg_depths, b2d_ratios, validation_rates, corrs = [], [], [], [], [], []
    for idx in tqdm(idxs):
        attempts, success, overwrite = 0, False, args.overwrite
        while attempts < 5 and not success:
            try:
                graph_metric = get_graph(idx, results, results_dir, overwrite)
                success = True
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except pdb.bdb.BdbQuit:
                raise pdb.bdb.BdbQuit
            except Exception as e:
                print(f"Error: {type(e)} {e}")
                print(f"Attempt {attempts} failed")
                # pdb.set_trace()
                attempts += 1
                overwrite = True
                continue
            
        max_depths.append(graph_metric["max_depth"])
        breadths.append(graph_metric["breadth"])
        avg_depths.append(graph_metric["avg_depth"])
        b2d_ratios.append(graph_metric["breadth"] / graph_metric["max_depth"])
        validation_rates.append(graph_metric["validation_rate"])
        corrs.append(graph_metric["corr"])
        
    max_depth = sum(max_depths) / len(max_depths)
    breadth = sum(breadths) / len(breadths)
    avg_depth = sum(avg_depths) / len(avg_depths)
    b2d_ratio = sum(b2d_ratios) / len(b2d_ratios)
    validation_rate = sum(validation_rates) / len(validation_rates)
    print(f"Max depth: {max_depth}, Breadth: {breadth}, Avg depth: {avg_depth}, B2D ratio: {b2d_ratio}, Validation rate: {validation_rate}")
        


    print("\n--- XGBoost Analysis ---")
    # Prepare data for XGBoost
    # Ensure all lists have the same length and are not empty
    if len(max_depths) > 0 and len(max_depths) == len(breadths) == len(avg_depths) == len(b2d_ratios) == len(validation_rates) == len(corrs):
        X = np.array([max_depths, breadths, avg_depths, b2d_ratios, validation_rates]).T
        y = np.array(corrs) # Assuming corrs contains binary correctness labels (e.g., 0 or 1)

        # Check if there are at least two classes in the target variable
        unique_classes = np.unique(y)
        if len(unique_classes) >= 2:
            # Check if there are enough samples relative to features
            if X.shape[0] > X.shape[1]:
                # Import XGBoost
                from xgboost import XGBClassifier
                
                # Instantiate the XGBoost model
                xgb_model = XGBClassifier(
                    random_state=42,
                    scale_pos_weight=len(y) / sum(y) - 1 if sum(y) > 0 else 1  # For imbalanced classes
                )

                # Train the model
                xgb_model.fit(X, y)

                y_pred = xgb_model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                print(f"\nModel Training Accuracy: {accuracy:.4f}")
                print(f"Baseline Accuracy (predicting majority class): {max(np.mean(y), 1 - np.mean(y)):.4f}")
                
                # Feature importance
                importance = xgb_model.feature_importances_
                features = ['max_depth', 'breadth', 'avg_depth', 'b2d_ratio', 'validation_rate']
                print("\nFeature Importance:")
                for i, feat in enumerate(features):
                    print(f"{feat}: {importance[i]:.4f}")

            else:
                print("Skipping XGBoost: Not enough samples relative to the number of features.")
        else:
            print(f"Skipping XGBoost: Only one class ({unique_classes[0]}) found in the target variable 'corrs'.")
    else:
        print("Skipping XGBoost: Data lists are empty or have inconsistent lengths.")
    print("--- End XGBoost Analysis ---\n")
    
    if args.wandb:
        wandb.log({
            "max_depth": max_depth,
            "breadth": breadth,
            "avg_depth": avg_depth,
            "b2d_ratio": b2d_ratio,
            "validation_rate": validation_rate,
        })
        wandb.finish()