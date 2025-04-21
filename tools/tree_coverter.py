import argparse
import json
import sys
from typing import Dict, List, Tuple, Any
import math
import re

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
    * `PROBLEM_DESCRIPTION`: The text of the math problem.
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
        * Treat routine verification or re-calculation steps as part of the function they verify, unless they trigger significant confusion or method changes that constitute their own phase (as per previous refinement).

    2.  **Cluster Steps by Function:**
        * Classify each step from the input `steps` list according to the logical functions/codes you defined in Task 1.
        * Group the `step` numbers based on the function code they are assigned to, ensuring every input step is assigned to a cluster.

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
    --- START OF INPUT ---
    STEP_1_OUTPUT_JSON:
    {parsed_steps}
    --- END OF INPUT ---
    """
    
def parse_json(json_data: str) -> Dict:
    """Parse JSON string into a dictionary."""
    return json.loads(json_data)

def generate_node_positions(logical_flow: Dict) -> Dict[str, Tuple[float, float]]:
    """Generate x, y coordinates for each node based on the logical flow structure."""
    positions = {}
    
    # Count approaches to distribute them
    root_children = logical_flow.get("root", [])
    num_approaches = len(root_children)
    
    # Set root position
    positions["root"] = (400, 50)
    
    # Special case handling: D approach needs more space
    approach_widths = {
        "A": 200,
        "B": 400,
        "C": 600,
        "D": 700
    }
    
    # Process each approach
    for approach_id in root_children:
        approach_letter = approach_id[0]
        x_pos = approach_widths.get(approach_letter, 0)
        
        # Position first node in approach
        positions[approach_id] = (x_pos, 120)
        
        # Track current y position as we go down the chain
        current_y = 120
        current_node = approach_id
        
        # Follow the chain in this approach
        while current_node in logical_flow and logical_flow[current_node]:
            next_node = logical_flow[current_node][0]  # Get first (should be only) child
            current_y += 80  # Increment y position for next level
            positions[next_node] = (x_pos, current_y)
            current_node = next_node
    
    # Add solution node if in the flow
    if "solution" in logical_flow.get("D4", []):
        positions["solution"] = (700, 440)
    
    return positions

def create_node_descriptions(clustering_results: Dict) -> Dict[str, str]:
    """Extract short descriptions for each node."""
    descriptions = {}
    for node_id, info in clustering_results.items():
        descriptions[node_id] = info.get("description", "")
    return descriptions

def generate_trajectory_path(logical_flow: Dict, positions: Dict[str, Tuple[float, float]]) -> str:
    """Generate SVG path for the trajectory through the nodes."""
    # Start with root
    path = f"M {positions['root'][0]} {positions['root'][1]} "
    
    # Follow each approach in sequence
    approaches = logical_flow.get("root", [])
    
    for approach_id in approaches:
        # Add path to the first node of this approach
        x, y = positions[approach_id]
        # Add a curve to make it look nice
        cx1, cy1 = positions['root'][0] - 150, positions['root'][1] + 20
        cx2, cy2 = x - 50, y - 20
        path += f"\n           C {cx1} {cy1}, {cx2} {cy2}, {x} {y}"
        
        # Follow the chain in this approach
        current_node = approach_id
        
        while current_node in logical_flow and logical_flow[current_node]:
            next_node = logical_flow[current_node][0]
            nx, ny = positions[next_node]
            
            # Add a nice curve
            cx1, cy1 = x + 20, y + 10
            cx2, cy2 = nx, ny - 40
            path += f"\n           C {cx1} {cy1}, {cx2} {cy2}, {nx} {ny}"
            
            current_node = next_node
            x, y = nx, ny
        
        # If not the last approach, add a curve back to the root of the next approach
        if approach_id != approaches[-1]:
            next_approach = approaches[approaches.index(approach_id) + 1]
            next_x, next_y = positions[next_approach]
            
            # Curve back up and then to the next approach
            cx1, cy1 = x, y + 30
            cx2, cy2 = next_x - 100, next_y - 30
            path += f"\n           C {cx1} {cy1}, {cx2} {cy2}, {next_x} {next_y}"
            
            x, y = next_x, next_y
    
    return path

def generate_svg(data: Dict) -> str:
    """Generate SVG visualization from the parsed data."""
    clustering_results = data.get("clustering_results", {})
    logical_flow = data.get("logical_flow", {})
    
    # Generate positions for each node
    positions = generate_node_positions(logical_flow)
    
    # Get descriptions for nodes
    descriptions = create_node_descriptions(clustering_results)
    
    # Define node colors for each approach
    node_colors = {
        'root': '#3498db',
        'A': '#2ecc71',
        'B': '#9b59b6',
        'C': '#f39c12',
        'D': '#e74c3c',
        'solution': '#27ae60'
    }
    
    # Start building SVG
    svg = f"""<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Styles -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
    <marker id="trajectory-arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#e74c3c"/>
    </marker>
  </defs>
  
  <!-- Tree Structure -->"""
    
    # Add connections between nodes
    for parent, children in logical_flow.items():
        # Skip if parent not in positions (like 'solution')
        if parent not in positions:
            continue
            
        px, py = positions[parent]
        
        for child in children:
            # Skip if child not in positions
            if child not in positions:
                continue
                
            cx, cy = positions[child]
            svg += f"""
  <line x1="{px}" y1="{py}" x2="{cx}" y2="{cy}" stroke="#555" stroke-width="2" marker-end="url(#arrowhead)"/>"""
    
    # Add trajectory path
    trajectory_path = generate_trajectory_path(logical_flow, positions)
    svg += f"""
  
  <!-- Trajectory curve with arrow -->
  <path d="{trajectory_path}"
        fill="none" stroke="#e74c3c" stroke-width="3" stroke-dasharray="5,3" marker-end="url(#trajectory-arrowhead)"/>
  
  <!-- Nodes -->"""
    
    # Add root node
    svg += f"""
  <!-- Root -->
  <circle cx="{positions['root'][0]}" cy="{positions['root'][1]}" r="20" fill="{node_colors['root']}"/>
  <text x="{positions['root'][0]}" cy="{positions['root'][1] + 5}" text-anchor="middle" fill="white" font-weight="bold">Root</text>
  """
    
    # Add all other nodes
    for node_id, (x, y) in positions.items():
        if node_id == 'root' or node_id == 'solution':
            continue
            
        approach_letter = node_id[0]
        color = node_colors.get(approach_letter, '#777')
        
        svg += f"""
  <circle cx="{x}" cy="{y}" r="20" fill="{color}"/>
  <text x="{x}" cy="{y + 5}" text-anchor="middle" fill="white" font-weight="bold">{node_id}</text>
  """
    
    # Add solution node if present
    if 'solution' in positions:
        x, y = positions['solution']
        svg += f"""
  <!-- Solution node -->
  <circle cx="{x}" cy="{y}" r="20" fill="{node_colors['solution']}"/>
  <text x="{x}" cy="{y + 5}" text-anchor="middle" fill="white" font-weight="bold">Sol</text>
  """
    
    # Add legend
    svg += """
  <!-- Legend -->
  <rect x="50" y="500" width="250" height="90" rx="10" ry="10" fill="#f8f9fa" stroke="#ddd"/>
  <text x="60" y="520" font-weight="bold">Legend:</text>
  <circle cx="70" cy="540" r="7" fill="#3498db"/>
  <text x="85" y="545">Root node</text>
  <line x1="170" y1="540" x2="210" y2="540" stroke="#555" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="220" y="545">Logical flow</text>
  <line x1="60" y1="565" x2="100" y2="565" stroke="#e74c3c" stroke-width="3" stroke-dasharray="5,3" marker-end="url(#trajectory-arrowhead)"/>
  <text x="110" y="570">Solution trajectory</text>
  """
    
    # Add descriptions
    for node_id, description in descriptions.items():
        if node_id not in positions:
            continue
            
        x, y = positions[node_id]
        # Add offset for text positioning
        svg += f"""
  <text x="{x + 50}" y="{y}" text-anchor="start" font-size="10">{description}</text>"""
    
    # Add title
    svg += """
  
  <text x="400" y="20" text-anchor="middle" font-size="14" font-weight="bold">Problem-Solving Approach Flow and Trajectory</text>
</svg>"""
    
    return svg

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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_str", type=str, required=True)
    parser.add_argument("--output_str", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    divide_prompt = get_divide_prompt(args.input_str, args.output_str)
    tree_prompt = get_tree_prompt(divide_prompt)

    # Parse JSON
    data = parse_json(tree_prompt)
    
    # Generate SVG
    svg = generate_svg(data)
    
    # Write SVG to file
    with open(args.output_file, 'w') as f:
        f.write(svg)
    
    print(f"SVG visualization saved to {args.output_file}")
    