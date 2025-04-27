import json
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import matplotlib.patches as mpatches
import argparse
import statistics
import logging

def analyze_hypotheses(data):
    """
    Analyzes the CoT graph data and returns metrics in dictionary form:
    1. All points_to relationships with node pairs and step differences
    2. Validation counts for each hypothesis
    3. All conclusion nodes with text and ID
    
    Args:
        data: The CoT data dictionary
    
    Returns:
        dict: A dictionary of metrics
    """
    nodes_by_id = {node['id']: node for node in data['nodes']}
    hypothesis_nodes = [n for n in data['nodes'] if n.get('type') == 'Hypothesis']
    conclusion_nodes = [n for n in data['nodes'] if n.get('type') == 'Conclusion']
    observation_nodes = [n for n in data['nodes'] if n.get('type') == 'Observation']
    validation_nodes = [n for n in data['nodes'] if n.get('type') == 'Validation']
    
    # Initialize metrics dictionary
    metrics = {
        "points_to_relationships": [],
        "hypothesis_validations": {},
        "conclusion_nodes": []
    }
    
    # 1. Collect points_to relationships
    for node in hypothesis_nodes:
        if node.get('points_to'):
            # Get the IDs
            start_id = node['id']
            pointed_id = node['points_to']
            
            # Calculate step difference
            if start_id < pointed_id:
                step_diff = pointed_id - start_id
            else:
                step_diff = -(start_id - pointed_id)
                
            # Add to metrics
            metrics["points_to_relationships"].append({
                "node_start": start_id,
                "node_pointed": pointed_id,
                "step_diff": step_diff
            })
    
    # 2. Count validations for each hypothesis
    for hypo in hypothesis_nodes:
        hid = hypo['id']
        validation_count = sum(
            1 for n in data['nodes']
            if n.get('type') == 'Validation' and 
               n.get('depends_on') and hid in n['depends_on']
        )
        
        metrics["hypothesis_validations"][hid] = {
            "id": hid,
            "text": hypo.get('text', ''),
            "validation_count": validation_count
        }
    
    # 3. Collect all conclusion nodes
    for concl in conclusion_nodes:
        metrics["conclusion_nodes"].append({
            "id": concl['id'],
            "text": concl.get('text', '')
        })
    
    # Calculate total validation count
    total_validation_count = sum(v['validation_count'] for v in metrics['hypothesis_validations'].values())
    hypothesis_count = len(hypothesis_nodes)
    average_validations = total_validation_count / hypothesis_count if hypothesis_count > 0 else 0
    
    # Calculate total step difference for points_to
    total_step_difference_magnitude = sum(abs(rel['step_diff']) for rel in metrics["points_to_relationships"])
    total_nodes = len(data['nodes'])
    normalized_step_difference = total_step_difference_magnitude / total_nodes if total_nodes > 0 else 0
    
    # Conclusion Stats
    conclusion_count = len(conclusion_nodes)
    first_conclusion_id = conclusion_nodes[0]['id'] if conclusion_count > 0 else None
    first_conclusion_text = conclusion_nodes[0]['text'] if conclusion_count > 0 else None
    first_conclusion_ratio = first_conclusion_id / total_nodes if conclusion_count > 0 and total_nodes > 0 and first_conclusion_id is not None else None
    last_conclusion_id = conclusion_nodes[-1]['id'] if conclusion_count > 0 else None
    last_conclusion_text = conclusion_nodes[-1]['text'] if conclusion_count > 0 else None
    
    # Observation Stats
    total_observations = len(observation_nodes)
    hallucinated_observations = len([n for n in observation_nodes if n.get('is_hallucinated')])
    
    # Validation Stats
    validation_counts = [v['validation_count'] for v in metrics['hypothesis_validations'].values()]
    max_validations = max(validation_counts) if validation_counts else 0
    min_validations = min(validation_counts) if validation_counts else 0
    median_validations = statistics.median(validation_counts) if validation_counts else 0
    
    # ---- START: Calculate Graph Depth ----
    G = nx.DiGraph()
    graph_depth = 0
    if data['nodes']:
        for node in data['nodes']:
            G.add_node(node['id'])

        for node in data['nodes']:
            tgt = node['id']
            deps = node.get('depends_on') or []
            for d in deps:
                if G.has_node(d): # Ensure dependency node exists
                    G.add_edge(d, tgt)
        
        try:
            # Calculate the longest path length (number of edges) + 1 for number of nodes
            graph_depth = nx.dag_longest_path_length(G) + 1
        except nx.NetworkXUnfeasible:
            # This occurs if the graph is not a DAG (has cycles)
            logging.warning("Graph contains cycles, cannot compute DAG longest path length. Setting depth to -1.")
            graph_depth = -1 # Indicate cycle or error
        except Exception as e:
            logging.error(f"Error calculating graph depth: {e}")
            graph_depth = -1 # Indicate error

    # ---- END: Calculate Graph Depth ----

    # Assemble Summary Dictionary
    metrics["summary"] = {
        "total_nodes": total_nodes,
        "hypothesis": {
            "hypothesis_count": hypothesis_count,
            "points_to_count": len(metrics["points_to_relationships"]),
            "total_step_difference_magnitude": total_step_difference_magnitude,
            "normalized_step_difference": normalized_step_difference
        },
        "conclusion":{
            "conclusion_count": conclusion_count,
            "first_conclusion_id": first_conclusion_id,
            "first_conclusion_text": first_conclusion_text,
            "first_conclusion_ratio": first_conclusion_ratio,
            "last_conclusion_id": last_conclusion_id,
            "last_conclusion_text": last_conclusion_text,
        },
        "validation":{
            "total_validation_count": total_validation_count,
            "average_validations_per_hypothesis": average_validations,
            "max_validations_per_hypothesis": max_validations,
            "min_validations_per_hypothesis": min_validations,
            "median_validations_per_hypothesis": median_validations,
        },
        "observation":{
            "total_observations": total_observations,
            "hallucinated_observations": hallucinated_observations,
        },
        "depth": graph_depth
    }
    metrics.pop("hypothesis_validations")
    
    return metrics

def preprocess_nodes(data, filter_input_obs=False, merge_input_obs=False):
    """
    Preprocess nodes based on options:
    - filter_input_obs: remove input observations
    - merge_input_obs: merge all input observations into one node
    
    Returns modified data
    """
    if not filter_input_obs and not merge_input_obs:
        return data
    
    # Create a deep copy of the data to avoid modifying the original
    import copy
    new_data = {"nodes": copy.deepcopy(data['nodes'])}
    
    # First identify all input observations
    input_obs_nodes = []
    non_input_nodes = []
    
    for node in new_data['nodes']:
        if node['type'] == 'Observation' and node.get('origin', {}).get('source') == 'input':
            input_obs_nodes.append(node)
        else:
            non_input_nodes.append(node)
    
    if filter_input_obs:
        # Simply remove all input observations
        new_data['nodes'] = non_input_nodes
        
        # Update dependencies to remove references to input observations
        input_obs_ids = {node['id'] for node in input_obs_nodes}
        for node in new_data['nodes']:
            if node.get('depends_on'):
                node['depends_on'] = [dep for dep in node['depends_on'] if dep not in input_obs_ids]
                # If all dependencies are removed, set to empty list instead of None
                if not node['depends_on']:
                    node['depends_on'] = []
    
    elif merge_input_obs and input_obs_nodes:
        # Create a merged input observation node
        merged_text = "Merged Input Observations:\n" + "\n".join([
            f"- {node.get('text', '')}" for node in input_obs_nodes
        ])
        
        # Use the ID of the first input observation for the merged node
        merged_id = input_obs_nodes[0]['id']
        merged_node = {
            "id": merged_id,
            "text": merged_text,
            "depends_on": None,
            "type": "Observation",
            "is_hallucinated": False,
            "origin": {"source": "input"},
            "is_merged": True  # Add a flag to mark this as a merged node
        }
        
        # Add the merged node and all non-input nodes
        new_data['nodes'] = [merged_node] + non_input_nodes
        
        # Update the dependencies to point to the merged node
        input_obs_ids = {node['id'] for node in input_obs_nodes}
        
        for node in new_data['nodes']:
            if node.get('depends_on'):
                # Replace dependencies on any input observation with the merged node ID
                deps = node['depends_on']
                updated_deps = []
                has_merged_dep = False
                
                for dep in deps:
                    if dep in input_obs_ids:
                        if not has_merged_dep:
                            updated_deps.append(merged_id)
                            has_merged_dep = True
                    else:
                        updated_deps.append(dep)
                
                node['depends_on'] = updated_deps
    
    return new_data

def create_visualization(data, output_dir, filter_input_obs=False, merge_input_obs=False, display=False):
    """
    Creates and saves a visualization of the chain of thought.
    
    Args:
        data: The CoT data
        output_dir: Directory to save output files
        filter_input_obs: Whether to filter out input observations
        merge_input_obs: Whether to merge all input observations into one node
        display: Whether to display the plot
    
    Returns:
        Path to the saved image file
    """
    # Apply preprocessing 
    processed_data = data
    
    if filter_input_obs or merge_input_obs:
        processed_data = preprocess_nodes(processed_data, filter_input_obs, merge_input_obs)

    # node coloring - using more aesthetic color scheme
    COLORS = {
        "Observation_input": "#8dd3c7",        # teal
        "Observation_input_merged": "#2ca25f", # darker teal for merged node
        "Observation_inferred": "#bebada",     # light purple
        "Hypothesis": "#fb8072",               # light red
        "Validation": "#fdb462",               # light orange
        "Conclusion": "#80b1d3",               # light blue
    }
    
    # Create readable labels
    LABELS = {
        "Observation_input": "Input Observation",
        "Observation_input_merged": "Merged Input Observations",
        "Observation_inferred": "Inferred Observation",
        "Hypothesis": "Hypothesis",
        "Validation": "Validation",
        "Conclusion": "Conclusion",
    }

    G = nx.DiGraph()
    P = nx.DiGraph()  # for points_to

    # Create mapping of node types for legend
    node_types = set()
    node_colors = {}  # Store colors separately to avoid KeyError
    
    for node in processed_data['nodes']:
        nid = node['id']
        ntype = node['type']
        if ntype == 'Observation':
            src = node.get('origin',{}).get('source')
            ntype_key = f"Observation_{src}"
            if src == 'input' and node.get('is_merged'):
                ntype_key = "Observation_input_merged"
        else:
            ntype_key = ntype
            
        node_types.add(ntype_key)
        color = COLORS.get(ntype_key, '#dddddd')
        if ntype == 'Observation' and node.get('is_hallucinated'):
            color = '#d53e4f'  # vibrant red for hallucinated
            node_types.add('Hallucinated')
            
        G.add_node(nid, type=ntype_key)
        node_colors[nid] = color  # Store color separately
        P.add_node(nid)

    # add depends_on edges
    for node in processed_data['nodes']:
        tgt = node['id']
        deps = node.get('depends_on') or []
        for d in deps:
            if d in G:  # Only add edge if both nodes exist
                G.add_edge(d, tgt)

    # add points_to edges
    for node in processed_data['nodes']:
        if node.get('type') == 'Hypothesis' and 'points_to' in node:
            if node['points_to'] in G:  # Only add edge if target node exists
                P.add_edge(node['id'], node['points_to'])

    # detect cycles
    cycles = list(nx.simple_cycles(G))
    if cycles:
        cycle_nodes = set(n for cyc in cycles for n in cyc)
    else:
        cycle_nodes = set()

    # Dynamic figure size based on number of nodes
    node_count = len(G.nodes())
    fig_size = max(18, min(28, node_count / 2.5))
    
    # Create figure with appropriate size
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use a more pleasing layout algorithm
    try:
        pos = graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
    # Draw nodes with enhanced styling
    node_sizes = []
    node_color_list = []
    
    for n in G.nodes():
        if G.nodes[n].get('type') == 'Observation_input_merged':
            node_sizes.append(3000)  # Larger size for merged input node
        else:
            node_sizes.append(1500)
        node_color_list.append(node_colors[n])
            
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color_list,
        edgecolors=['#d53e4f' if n in cycle_nodes else 'black' for n in G.nodes()],
        linewidths=2.5,
        node_size=node_sizes,
        alpha=0.9
    )
    
    # Draw normal edges with better styling and CLEAR ARROWS
    nx.draw_networkx_edges(
        G, pos, 
        arrows=True,
        width=2.0,
        alpha=0.7,
        arrowsize=25,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        min_source_margin=25,
        min_target_margin=25
    )
    
    # Draw points_to edges with CLEARER ARROWS
    if P.number_of_edges():
        nx.draw_networkx_edges(
            P, pos,
            arrows=True,
            style='dashed',
            edge_color='#d53e4f',
            width=2.5,
            alpha=0.9,
            arrowsize=30,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.2',
            min_source_margin=25,
            min_target_margin=25
        )
        
    # Create better labels
    labels = {}
    for node in processed_data['nodes']:
        nid = node['id']
        if nid in G:  # Only add labels for nodes in the graph
            if node.get('type') == 'Observation' and node.get('is_merged'):
                labels[nid] = "Input"
            else:
                labels[nid] = f"{nid}"
        
    # Add network labels with enhanced settings
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels,
        font_size=12,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # Add a title
    title = "Chain of Thought Visualization"
    if filter_input_obs:
        title += " (Input Observations Filtered)"
    elif merge_input_obs:
        title += " (Input Observations Merged)"
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = []
    
    # Node type legend
    for node_type in sorted(node_types):
        if node_type == 'Hallucinated':
            color = '#d53e4f'
            label = 'Hallucinated Observation'
        else:
            color = COLORS.get(node_type, '#dddddd')
            label = LABELS.get(node_type, node_type)
            
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=label)
        )
    
    # Edge type legend with arrow examples
    legend_elements.append(
        plt.Line2D([0], [0], color='black', lw=2, label='Depends On',
                  marker='>', markersize=10, markevery=[1], markeredgewidth=0)
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='#d53e4f', lw=2, label='Points To',
                  marker='>', markersize=10, markevery=[1], markeredgewidth=0, 
                  linestyle='dashed')
    )
    
    # Add legend with better styling
    plt.legend(
        handles=legend_elements, 
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Legend",
        title_fontsize=14
    )
    
    plt.tight_layout()
    plt.axis('off')
    
    # Create suffix for filenames based on options
    suffix = ""
    if filter_input_obs:
        suffix += "_filtered"
    elif merge_input_obs:
        suffix += "_merged"
    
    # Save the figure with high quality as PNG
    img_path = os.path.join(output_dir, f"cot_graph{suffix}.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved PNG visualization to: {img_path}")
    
    # Show the graph if requested
    if display:
        plt.show()
    else:
        plt.close()
    
    return img_path

def draw_chain_of_thought(data, output_dir, filter_input_obs=False, merge_input_obs=False, save_all=True):
    """
    Constructs and displays a directed graph for the chain-of-thought.
    
    Args:
        data: The CoT data
        output_dir: Directory to save output files
        filter_input_obs: Whether to filter out input observations
        merge_input_obs: Whether to merge all input observations into one node
        save_all: Whether to save all three versions (normal, filtered, merged)
    """
    if save_all:
        # Save all versions
        print("Generating normal visualization...")
        create_visualization(data, output_dir, display=False)
        
        print("Generating filtered visualization...")
        create_visualization(data, output_dir, filter_input_obs=True, display=False)
        
        print("Generating merged visualization...")
        create_visualization(data, output_dir, merge_input_obs=True, display=False)
        
        # Show only the requested visualization (or normal if none specified)
        if filter_input_obs:
            print("Displaying filtered visualization...")
            create_visualization(data, output_dir, filter_input_obs=True, display=True)
        elif merge_input_obs:
            print("Displaying merged visualization...")
            create_visualization(data, output_dir, merge_input_obs=True, display=True)
        else:
            print("Displaying normal visualization...")
            create_visualization(data, output_dir, display=True)
    else:
        # Only save and display the requested version
        create_visualization(data, output_dir, 
                           filter_input_obs=filter_input_obs, 
                           merge_input_obs=merge_input_obs,
                           display=True)
                           
    return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize Chain of Thought reasoning.')
    parser.add_argument('input_file', nargs='?', 
                        default="/home/szhang967/liftr/reasoning_analysis/sample_cot.json",
                        help='Path to the JSON file containing the CoT data')
    parser.add_argument('--filter-input', action='store_true',
                        help='Filter out input observations from the visualization')
    parser.add_argument('--merge-input', action='store_true',
                        help='Merge all input observations into a single node')
    parser.add_argument('--only-selected', action='store_true',
                        help='Only save the selected visualization (not all versions)')
    
    args = parser.parse_args()
    
    # Check for incompatible options
    if args.filter_input and args.merge_input:
        print("Error: Cannot both filter and merge input observations. Choose one option.")
        sys.exit(1)
    
    # Extract the directory from the input path
    output_dir = os.path.dirname(args.input_file)
    if not output_dir:
        output_dir = "."
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    draw_chain_of_thought(data, output_dir, 
                          filter_input_obs=args.filter_input, 
                          merge_input_obs=args.merge_input,
                          save_all=not args.only_selected) 