import json
from google import genai 
from environment import GEMINI_API_KEY
import re
import graphviz
import argparse
import pandas as pd
import os
from tqdm import tqdm
from constants import get_result_dir, supported_datasets
from utils import save_json, load_json
import pdb
from collections import defaultdict

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

    3.  **Determine Logical Flow (Tree Structure with Branches):**
        * Based on the clustering results and the reasoning narrative, identify the overall tree structure, including alternative approaches.
        * Determine the "starting node code" (like `A1`, `B1`, `C1`...) for each distinct top-level approach attempted.
        * Trace the sequence of function codes *within* each distinct approach branch.
        * Identify which approach branch leads to the final successful conclusion.
        * Represent this structure as a tree within the `logical_flow` object:
            * Include a special key `"root"` whose value is a list of the starting node codes for **all** distinct top-level approaches identified.
            * For each node code defined (`A1`, `A2`, etc.), create a key. Its value should be a list containing the node code(s) that *directly follow it within its specific approach branch*.
            * Nodes that represent the end of a failed or abandoned approach branch should have an empty list `[]` as their value.
            * The **final node** of the **successful** approach branch should have a list containing only the string `"solution"` (i.e., `["solution"]`) as its value.

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
            //    "description": (string) The concise descriptive label for the function/phase (from Task 1).
            //    "steps": (list of integers) The step numbers performing this function (from Task 2).
            // Example (content depends on analysis):
            // "A1": {{ "description": "Approach 1: State Method...", "steps": [1] }},
            // "A2": {{ "description": "Approach 1: Execute Tests...", "steps": [2, 3, 4, 5, 6] }},
            // "A3": {{ "description": "Approach 1: Evaluate Method...", "steps": [7] }},
            // ... etc. for B1-B3, C1-C4, D1-D4 ...
        }},
        "logical_flow": {{
            // Part 2 Output: Tree Flow Information
            // Matches the structure requested by the user.
            // Includes a "root" key pointing to starting nodes of each approach.
            // Other keys are node codes, values are lists of direct successors within that approach branch.
            // Failed branches end with []. Successful branch ends pointing to "solution".
            // Example (Based on user's desired structure and A1..D4 nodes):
            "root": ["A1", "B1", "C1", "D1"],
            "A1": ["A2"],
            "A2": ["A3"],
            "A3": [], // End of failed Approach 1
            "B1": ["B2"],
            "B2": ["B3"],
            "B3": [], // End of failed Approach 2
            "C1": ["C2"],
            "C2": ["C3"],
            "C3": ["C4"],
            "C4": [], // End of failed Approach 3
            "D1": ["D2"],
            "D2": ["D3"],
            "D3": ["D4"],
            "D4": ["solution"] // End of successful Approach 4
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

client = genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(prompt, model_name = "gemini-2.5-pro-preview-03-25"):
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
    )
    return response.candidates[0].content.parts[0].text

# Calculate depth for each node
def calculate_depth(node, visited=None, current_depth=0, logical_flow=None):
    if visited is None:
        visited = set()
    
    if node in visited:
        return current_depth
    
    visited.add(node)
    
    if node not in logical_flow or not logical_flow[node]:
        return current_depth
    
    max_child_depth = current_depth
    for child in logical_flow.get(node, []):
        child_depth = calculate_depth(child, visited.copy(), current_depth + 1, logical_flow)
        max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth

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
    
    
    # Start depth calculation from root
    if 'root' in logical_flow:
        max_depth = calculate_depth('root', logical_flow=logical_flow)
    
    # Breadth is the number of leaf nodes
    breadth = len(leaf_nodes)
    
    return dot, max_depth, breadth

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

    print(f"Adding highlighted path (no arrows): {' -> '.join(path_nodes)}")

    # Highlight all nodes in the path (just the boundary, not filled)
    for i, node in enumerate(path_nodes):
        if i == 0:  # Start node
            graph.node(node, color='green', penwidth=str(penwidth))
        elif i == len(path_nodes) - 1:  # End node
            graph.node(node, color='red', penwidth=str(penwidth))
        else:  # Middle nodes
            graph.node(node, color=color, penwidth=str(penwidth))

    # Iterate through the path sequence to add edges between consecutive nodes
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]

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
    step_to_nodes = defaultdict(list)
    all_steps = []
    for node_code, details in clustering_results.items():
        steps = details.get('steps', [])
        if not isinstance(steps, list):
            # Handle potential malformed data if 'steps' is not a list
             continue
        for step in steps:
            if isinstance(step, int):
                step_to_nodes[step].append(node_code)
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
        # Get nodes associated with the current step, default to empty list
        nodes_for_step = step_to_nodes.get(step_num, [])
        # Sort node codes alphabetically for consistent output when a step is in multiple nodes
        nodes_for_step.sort()
        # Add all nodes for this step to the trace
        trace.extend(nodes_for_step)

    return trace

def get_graph(idx, results, results_dir, overwrite=False):
    result_path = f"{results_dir}/tree_vis/{idx}.json"
    if not os.path.exists(result_path) or overwrite:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        input_str = results.iloc[idx]["prompt"][0]["content"]
        output_str = results.iloc[idx]["responses"][0]
        divide_prompt = get_divide_prompt(input_str, output_str)
        parsed_steps = call_gemini(divide_prompt)
        
        tree_prompt = get_tree_prompt(parsed_steps)
        output_json = call_gemini(tree_prompt)
        json_data = parse_json(output_json)
        save_json({"parsed_steps": parsed_steps, "visualization": json_data}, result_path)
    else:
        json_data = load_json(result_path)["visualization"]
    
    visit_order = get_node_visit_order(json_data)
    graph, max_depth, breadth = create_flowchart_from_dict(json_data)
    graph = add_highlighted_path(graph, visit_order, color='blue', penwidth='2.5')
    if graph is not None:
        graph.render(f"{results_dir}/tree_vis/{idx}", format="pdf")
    
    return graph, max_depth, breadth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs='+', default=[])
    parser.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-reasoner")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    
    results_dir = get_result_dir(
        dataset_name = args.dataset_name,
        model_name = args.model_name,
        shot = 0,
        template_type = "reasoning_api",
        response_length = 404,
        num_samples = 500,
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
    
    max_depths = []
    breadths = []
    for idx in tqdm(idxs):
        graph, max_depth, breadth = get_graph(idx, results, results_dir, args.overwrite)
        max_depths.append(max_depth)
        breadths.append(breadth)
        
    max_depth = sum(max_depths) / len(max_depths)
    breadth = sum(breadths) / len(breadths)
    print(f"Max depth: {max_depth}, Breadth: {breadth}")
        
