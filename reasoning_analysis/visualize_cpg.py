import json
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import matplotlib.patches as mpatches
import argparse
import statistics # Keep for potential future use, but not used in current analyze_cpg
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_cpg(data):
    """
    Analyzes the Cognitive Process Graph (CPG) data based on the 4 simplified types.
    Currently returns basic node counts and graph depth in the summary.
    (Placeholder for future, more detailed CPG metric calculations)

    Args:
        data: The CPG data dictionary (nodes assumed to have 'id', 'depends_on', 'type')

    Returns:
        dict: A dictionary containing a 'summary' of basic metrics.
    """
    if not data or 'nodes' not in data or not data['nodes']:
        logging.warning("Input data is empty or missing 'nodes'. Returning empty summary.")
        return {"summary": {
            "total_nodes": 0,
            "observation_given_count": 0,
            "calculation_process_count": 0,
            "conclusion_assertion_count": 0,
            "meta_cognition_plan_count": 0,
            "depth": 0
        }}

    nodes_by_id = {node['id']: node for node in data['nodes']}
    total_nodes = len(data['nodes'])

    # Count nodes per type
    obs_given_count = sum(1 for n in data['nodes'] if n.get('type') == 'Observation/Given')
    calc_proc_count = sum(1 for n in data['nodes'] if n.get('type') == 'Calculation/Process Step')
    conc_assert_count = sum(1 for n in data['nodes'] if n.get('type') == 'Conclusion/Assertion')
    meta_cog_count = sum(1 for n in data['nodes'] if n.get('type') == 'Meta-Cognition/Plan')

    # ---- START: Calculate Graph Depth ----
    G = nx.DiGraph()
    graph_depth = 0
    if total_nodes > 0:
        for node in data['nodes']:
            # Ensure node id is suitable for networkx (e.g., integer)
            if isinstance(node.get('id'), int):
                G.add_node(node['id'])
            else:
                logging.warning(f"Node with invalid ID found: {node.get('id')}. Skipping.")


        for node in data['nodes']:
            if not isinstance(node.get('id'), int) or not G.has_node(node['id']):
                continue # Skip nodes with invalid IDs or not added

            tgt = node['id']
            deps = node.get('depends_on') or []
            for d in deps:
                # Ensure dependency node ID is valid and exists in the graph
                if isinstance(d, int) and G.has_node(d):
                    G.add_edge(d, tgt)
                else:
                     logging.warning(f"Node {tgt} depends on invalid or missing node ID: {d}. Skipping edge.")


        # Ensure graph is not empty before calculating depth
        if G.number_of_nodes() > 0:
            try:
                # Calculate the longest path length (number of edges) + 1 for number of nodes
                # Check if the graph is a DAG
                if nx.is_directed_acyclic_graph(G):
                     graph_depth = nx.dag_longest_path_length(G) + 1
                else:
                     # Handle cyclic graph - depth is often considered infinite or undefined.
                     # Alternatively, find longest path ignoring cycles if meaningful, or report cycle.
                     logging.warning("Graph contains cycles, cannot compute DAG longest path length. Setting depth to -1.")
                     graph_depth = -1 # Indicate cycle

            except nx.NetworkXUnfeasible:
                # This might occur if the graph structure has issues despite node checks
                logging.warning("Graph is not a DAG (contains cycles). Setting depth to -1.")
                graph_depth = -1 # Indicate cycle or error
            except Exception as e:
                logging.error(f"Error calculating graph depth: {e}")
                graph_depth = -1 # Indicate general error
        else:
             graph_depth = 0 # No nodes, depth is 0

    # ---- END: Calculate Graph Depth ----

    # Assemble Summary Dictionary
    summary_metrics = {
        "total_nodes": total_nodes,
        "observation_given_count": obs_given_count,
        "calculation_process_count": calc_proc_count,
        "conclusion_assertion_count": conc_assert_count,
        "meta_cognition_plan_count": meta_cog_count,
        "depth": graph_depth
    }

    # Return structure includes only the summary key
    metrics = {"summary": summary_metrics}

    return metrics


def preprocess_nodes(data, filter_obs_given=False, merge_obs_given=False):
    """
    Preprocess nodes based on options for 'Observation/Given' type:
    - filter_obs_given: remove Observation/Given nodes
    - merge_obs_given: merge all Observation/Given nodes into one node

    Returns modified data (deep copy)
    """
    if not filter_obs_given and not merge_obs_given:
        return data # Return original data if no preprocessing needed

    # Create a deep copy of the data to avoid modifying the original
    import copy
    new_data = {"nodes": copy.deepcopy(data['nodes'])}

    # Identify Observation/Given nodes and others
    obs_given_nodes = []
    non_obs_given_nodes = []

    for node in new_data['nodes']:
        if node.get('type') == 'Observation/Given':
            obs_given_nodes.append(node)
        else:
            non_obs_given_nodes.append(node)

    if filter_obs_given:
        # Simply remove all Observation/Given nodes
        new_data['nodes'] = non_obs_given_nodes

        # Update dependencies to remove references to these nodes
        obs_given_ids = {node['id'] for node in obs_given_nodes}
        for node in new_data['nodes']:
            if node.get('depends_on'):
                # Keep only dependencies that are NOT in the removed set
                node['depends_on'] = [dep for dep in node['depends_on'] if dep not in obs_given_ids]
                # If all dependencies are removed, set to empty list
                if not node['depends_on']:
                    node['depends_on'] = [] # Use empty list for consistency

    elif merge_obs_given and obs_given_nodes:
        # Create a merged Observation/Given node
        # Use the ID of the first Observation/Given node for the merged node
        # Ensure node IDs are integers before proceeding
        valid_obs_given_nodes = [n for n in obs_given_nodes if isinstance(n.get('id'), int)]
        if not valid_obs_given_nodes:
             logging.warning("No valid Observation/Given nodes found to merge.")
             return new_data # Return unmodified data if no valid nodes to merge

        merged_id = valid_obs_given_nodes[0]['id']
        merged_node = {
            "id": merged_id,
            # "text": "Merged Observation/Given", # Text field removed from schema
            "depends_on": None, # Merged node has no dependencies
            "type": "Observation/Given",
            "is_merged": True  # Add a flag to mark this as a merged node
        }

        # Add the merged node and all non-Observation/Given nodes
        new_data['nodes'] = [merged_node] + non_obs_given_nodes

        # Update the dependencies in other nodes to point to the merged node
        obs_given_ids = {node['id'] for node in valid_obs_given_nodes}

        for node in new_data['nodes']:
            # Skip the merged node itself
            if node.get('is_merged'):
                continue

            if node.get('depends_on'):
                deps = node['depends_on']
                updated_deps = []
                # Flag to ensure merged_id is added only once per node
                has_merged_dep_added = False

                for dep in deps:
                    if dep in obs_given_ids:
                        # If dependency is one of the merged nodes,
                        # add the merged_id only if not already added
                        if not has_merged_dep_added:
                            updated_deps.append(merged_id)
                            has_merged_dep_added = True
                    else:
                        # Keep dependencies that are not Observation/Given nodes
                        updated_deps.append(dep)

                # Remove potential duplicates if multiple original obs were dependencies
                node['depends_on'] = list(set(updated_deps))
                # If list becomes empty, set to []
                if not node['depends_on']:
                     node['depends_on'] = []

    return new_data

def create_visualization(data, output_dir, filter_obs_given=False, merge_obs_given=False, display=False):
    """
    Creates and saves a visualization of the Cognitive Process Graph (CPG).

    Args:
        data: The CPG data (nodes assumed to have 'id', 'depends_on', 'type')
        output_dir: Directory to save output files
        filter_obs_given: Whether to filter out Observation/Given nodes
        merge_obs_given: Whether to merge Observation/Given nodes
        display: Whether to display the plot

    Returns:
        Path to the saved image file
    """
    # Apply preprocessing
    processed_data = data
    if filter_obs_given or merge_obs_given:
        # Pass the correct flags based on args
        processed_data = preprocess_nodes(processed_data,
                                          filter_obs_given=filter_obs_given,
                                          merge_obs_given=merge_obs_given)

    # Define colors and labels for the 4 new types + merged type
    COLORS = {
        "Observation/Given": "#8dd3c7",        # teal (default for this type)
        "Observation/Given_merged": "#2ca25f", # darker teal for merged node
        "Calculation/Process Step": "#fdb462", # light orange
        "Conclusion/Assertion": "#80b1d3",     # light blue
        "Meta-Cognition/Plan": "#bebada",      # light purple
    }

    LABELS = {
        "Observation/Given": "Observation/Given",
        "Observation/Given_merged": "Merged Observation/Given",
        "Calculation/Process Step": "Calculation/Process Step",
        "Conclusion/Assertion": "Conclusion/Assertion",
        "Meta-Cognition/Plan": "Meta-Cognition/Plan",
    }

    G = nx.DiGraph()

    # Keep track of node types present and assign colors
    node_types_present = set()
    node_colors_map = {} # Store assigned colors for nodes

    if not processed_data or 'nodes' not in processed_data:
         logging.error("No nodes found in processed data for visualization.")
         return None

    for node in processed_data['nodes']:
        nid = node.get('id')
        ntype = node.get('type')

        # Ensure node ID is valid
        if not isinstance(nid, int):
             logging.warning(f"Node with invalid ID found: {nid}. Skipping for visualization.")
             continue

        # Determine the type key for color/label lookup
        ntype_key = ntype
        if ntype == 'Observation/Given' and node.get('is_merged'):
            ntype_key = "Observation/Given_merged"

        node_types_present.add(ntype_key)
        color = COLORS.get(ntype_key, '#dddddd') # Default grey for unknown types

        G.add_node(nid, type=ntype_key) # Store the type key
        node_colors_map[nid] = color # Store the assigned color

    # Add depends_on edges
    for node in processed_data['nodes']:
        tgt_id = node.get('id')
        # Skip if target node wasn't added (e.g., invalid ID)
        if not isinstance(tgt_id, int) or tgt_id not in G:
            continue

        deps = node.get('depends_on') or []
        for dep_id in deps:
             # Ensure dependency node ID is valid and exists in the graph
            if isinstance(dep_id, int) and dep_id in G:
                G.add_edge(dep_id, tgt_id)
            else:
                 logging.warning(f"Visualization: Node {tgt_id} depends on invalid or missing node ID: {dep_id}. Skipping edge.")


    # Detect cycles (optional, but good for layout/analysis)
    cycles = []
    cycle_nodes = set()
    try:
        # Only run cycle detection if graph has nodes/edges
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
             cycles = list(nx.simple_cycles(G))
             if cycles:
                cycle_nodes = set(n for cyc in cycles for n in cyc)
                logging.warning(f"Cycles detected involving nodes: {cycle_nodes}")
    except Exception as e:
        logging.error(f"Error during cycle detection: {e}")


    # Dynamic figure size based on number of nodes
    node_count = G.number_of_nodes()
    if node_count == 0:
        logging.warning("Graph has no nodes to visualize.")
        return None
    fig_size = max(18, min(28, node_count / 2.5))

    plt.figure(figsize=(fig_size, fig_size))

    # Use graphviz layout if available and graph is acyclic, otherwise fall back
    pos = None
    try:
        # Use 'dot' for hierarchical layout, good for DAGs
        if not cycles: # Use dot only if no cycles detected
             pos = graphviz_layout(G, prog='dot')
        else:
             logging.info("Graph has cycles, using spring_layout.")
             pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42) # Seed for reproducibility
    except ImportError:
         logging.warning("PyGraphviz not found, using spring_layout.")
         pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)
    except Exception as e:
         logging.error(f"Layout calculation failed: {e}. Using default spring_layout.")
         pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)


    # Prepare node properties for drawing
    node_sizes_list = []
    node_color_list_viz = []
    node_edge_colors = []

    for n in G.nodes():
        node_attr = G.nodes[n]
        # Size based on type (e.g., merged nodes)
        if node_attr.get('type') == 'Observation/Given_merged':
            node_sizes_list.append(3000)
        else:
            node_sizes_list.append(1500)
        # Color from pre-calculated map
        node_color_list_viz.append(node_colors_map.get(n, '#dddddd'))
        # Edge color based on cycles
        node_edge_colors.append('#d53e4f' if n in cycle_nodes else 'black')

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color_list_viz,
        edgecolors=node_edge_colors,
        linewidths=2.5,
        node_size=node_sizes_list,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        width=2.0,
        alpha=0.7,
        arrowsize=25,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        min_source_margin=25, # Adjust based on node size
        min_target_margin=25
    )

    # Prepare labels
    labels_map = {n: str(n) for n in G.nodes()}
    # Special label for merged node
    for n, attr in G.nodes(data=True):
         if attr.get('type') == 'Observation/Given_merged':
              labels_map[n] = "Obs/Given\n(Merged)"


    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels=labels_map,
        font_size=10, # Slightly smaller font size for potentially larger graphs
        font_weight='bold',
        font_family='sans-serif'
    )

    # Add title
    title = "Cognitive Process Graph (CPG)"
    if filter_obs_given:
        title += " (Observation/Given Filtered)"
    elif merge_obs_given:
        title += " (Observation/Given Merged)"
    plt.title(title, fontsize=20, fontweight='bold', pad=20)

    # Add legend
    legend_elements = []
    # Add node types present in the graph to the legend
    for ntype_key in sorted(list(node_types_present)):
         color = COLORS.get(ntype_key, '#dddddd')
         label = LABELS.get(ntype_key, ntype_key.replace('_', ' ').title()) # Default label if not found
         legend_elements.append(
             mpatches.Patch(facecolor=color, edgecolor='black', label=label)
         )

    # Add edge legend
    legend_elements.append(
        plt.Line2D([0], [0], color='black', lw=2, label='Depends On',
                  marker='>', markersize=10, markevery=[1], linestyle='-', markeredgewidth=0) # Use linestyle for clarity
    )

    # Place legend outside the plot
    plt.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.01, 1), # Position outside plot area
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Legend",
        title_fontsize=14
    )

    # Adjust layout and turn off axis
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend
    plt.axis('off')

    # Determine filename suffix
    suffix = ""
    if filter_obs_given:
        suffix += "_filtered"
    elif merge_obs_given:
        suffix += "_merged"

    # Save the figure
    img_path = os.path.join(output_dir, f"cpg_graph{suffix}.png")
    try:
        plt.savefig(img_path, bbox_inches='tight', dpi=300)
        logging.info(f"Saved PNG visualization to: {img_path}")
    except Exception as e:
        logging.error(f"Failed to save visualization: {e}")
        img_path = None


    # Show the graph if requested
    if display:
        plt.show()
    else:
        plt.close() # Close the plot if not displayed

    return img_path


def draw_cognitive_process_graph(data, output_dir, filter_obs_given=False, merge_obs_given=False, save_all=True):
    """
    Manages the creation and display of CPG visualizations based on options.

    Args:
        data: The CPG data
        output_dir: Directory to save output files
        filter_obs_given: Whether to filter Observation/Given nodes
        merge_obs_given: Whether to merge Observation/Given nodes
        save_all: Whether to save all three versions (normal, filtered, merged)
    """
    if save_all:
        # Save all versions
        logging.info("Generating normal visualization...")
        create_visualization(data, output_dir, display=False)

        logging.info("Generating filtered visualization...")
        create_visualization(data, output_dir, filter_obs_given=True, display=False)

        logging.info("Generating merged visualization...")
        create_visualization(data, output_dir, merge_obs_given=True, display=False)

        # Determine which version to display based on args (default to normal)
        display_filter = filter_obs_given
        display_merge = merge_obs_given
        if not filter_obs_given and not merge_obs_given:
             display_filter = False
             display_merge = False # Display normal version

        logging.info(f"Displaying selected visualization (filter={display_filter}, merge={display_merge})...")
        create_visualization(data, output_dir,
                             filter_obs_given=display_filter,
                             merge_obs_given=display_merge,
                             display=True)
    else:
        # Only save and display the specifically requested version
        logging.info(f"Generating and displaying selected visualization (filter={filter_obs_given}, merge={merge_obs_given})...")
        create_visualization(data, output_dir,
                           filter_obs_given=filter_obs_given,
                           merge_obs_given=merge_obs_given,
                           display=True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize Cognitive Process Graph (CPG).')
    parser.add_argument('input_file', nargs='?',
                        # Provide a sensible default or remove if input is always required
                        default="sample_cpg_data.json", # Example default filename
                        help='Path to the JSON file containing the CPG data (nodes with types: Observation/Given, Calculation/Process Step, Conclusion/Assertion, Meta-Cognition/Plan)')
    parser.add_argument('--filter-obs', action='store_true',
                        help='Filter out Observation/Given nodes from the visualization')
    parser.add_argument('--merge-obs', action='store_true',
                        help='Merge all Observation/Given nodes into a single node')
    parser.add_argument('--only-selected', action='store_true',
                        help='Only save the selected visualization (normal, filtered, or merged) instead of all three versions')

    args = parser.parse_args()

    # Check for incompatible options
    if args.filter_obs and args.merge_obs:
        print("Error: Cannot both filter and merge Observation/Given nodes. Choose one option.")
        sys.exit(1)

    # Ensure input file exists
    if not os.path.exists(args.input_file):
         print(f"Error: Input file not found at {args.input_file}")
         sys.exit(1)


    # Determine output directory
    output_dir = os.path.dirname(args.input_file)
    if not output_dir: # Handle case where input file is in current directory
        output_dir = "."

    # Load data from JSON file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            cpg_data = json.load(f)
    except json.JSONDecodeError:
         print(f"Error: Invalid JSON format in {args.input_file}")
         sys.exit(1)
    except Exception as e:
         print(f"Error loading file {args.input_file}: {e}")
         sys.exit(1)


    # --- Optional: Call analysis function ---
    # You might want to calculate metrics even if not displaying them all
    # analysis_results = analyze_cpg(cpg_data)
    # print("Analysis Summary:", json.dumps(analysis_results.get("summary", {}), indent=2))
    # ---

    # Generate visualization(s)
    draw_cognitive_process_graph(cpg_data, output_dir,
                                filter_obs_given=args.filter_obs,
                                merge_obs_given=args.merge_obs,
                                save_all=not args.only_selected)