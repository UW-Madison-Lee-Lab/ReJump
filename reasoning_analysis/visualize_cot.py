import json
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import matplotlib.patches as mpatches

def analyze_hypotheses(data, output_dir):
    """
    Analyzes each hypothesis node and generates a table with statistics
    about its dependencies and validations, plus additional quantifiable metrics.
    Returns a pandas DataFrame.
    """
    results = []
    nodes_by_id = {node['id']: node for node in data['nodes']}
    hypothesis_nodes = [n for n in data['nodes'] if n.get('type') == 'Hypothesis']

    for hypo in hypothesis_nodes:
        hid = hypo['id']
        # count how many nodes depend on this hypo
        dependent_nodes_count = sum(
            1 for n in data['nodes']
            if n.get('depends_on') and hid in n['depends_on']
        )
        dependent_hypotheses_count = sum(
            1 for n in hypothesis_nodes
            if n.get('depends_on') and hid in n['depends_on']
        )
        points_to_count = sum(
            1 for n in hypothesis_nodes
            if n.get('points_to') == hid
        )
        validations_count = sum(
            1 for n in data['nodes']
            if n.get('type') == 'Validation' and n.get('depends_on') and hid in n['depends_on']
        )

        # count observations by origin
        deps = hypo.get('depends_on') or []
        input_obs = sum(
            1 for did in deps
            if nodes_by_id[did].get('type') == 'Observation'
               and nodes_by_id[did].get('origin', {}).get('source') == 'input'
        )
        inferred_obs = sum(
            1 for did in deps
            if nodes_by_id[did].get('type') == 'Observation'
               and nodes_by_id[did].get('origin', {}).get('source') == 'inferred'
        )
        results.append({
            'hypothesis_id': hid,
            'text': hypo.get('text',''),
            'input_observations': input_obs,
            'inferred_observations': inferred_obs,
            'validation_count': validations_count,
            'dependent_nodes_count': dependent_nodes_count,
            'dependent_hypotheses_count': dependent_hypotheses_count,
            'points_to_count': points_to_count
        })

    df = pd.DataFrame(results)
    # Save CSV to file
    csv_path = os.path.join(output_dir, "hypothesis_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved analysis to: {csv_path}")
    
    return df

def draw_chain_of_thought(data, output_dir):
    """
    Constructs and displays a directed graph for the chain-of-thought.
    """
    # analyze and print hypothesis table
    analyze_hypotheses(data, output_dir)

    # node coloring - using more aesthetic color scheme
    COLORS = {
        "Observation_input": "#8dd3c7",        # teal
        "Observation_inferred": "#bebada",     # light purple
        "Hypothesis": "#fb8072",               # light red
        "Validation": "#fdb462",               # light orange
        "Conclusion": "#80b1d3",               # light blue
    }
    
    # Create readable labels
    LABELS = {
        "Observation_input": "Input Observation",
        "Observation_inferred": "Inferred Observation",
        "Hypothesis": "Hypothesis",
        "Validation": "Validation",
        "Conclusion": "Conclusion",
    }

    G = nx.DiGraph()
    P = nx.DiGraph()  # for points_to

    # Create mapping of node types for legend
    node_types = set()
    
    for node in data['nodes']:
        nid = node['id']
        ntype = node['type']
        if ntype == 'Observation':
            src = node.get('origin',{}).get('source')
            ntype_key = f"Observation_{src}"
        else:
            ntype_key = ntype
            
        node_types.add(ntype_key)
        color = COLORS.get(ntype_key, '#dddddd')
        if ntype == 'Observation' and node.get('is_hallucinated'):
            color = '#d53e4f'  # vibrant red for hallucinated
            node_types.add('Hallucinated')
            
        G.add_node(nid, color=color, type=ntype_key)
        P.add_node(nid)

    # add depends_on edges
    for node in data['nodes']:
        tgt = node['id']
        deps = node.get('depends_on') or []
        for d in deps:
            G.add_edge(d, tgt)

    # add points_to edges
    for node in data['nodes']:
        if node.get('type') == 'Hypothesis' and 'points_to' in node:
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
    nx.draw_networkx_nodes(
        G, pos,
        node_color=[G.nodes[n]['color'] for n in G.nodes()],
        edgecolors=['#d53e4f' if n in cycle_nodes else 'black' for n in G.nodes()],
        linewidths=2.5,
        node_size=1500,
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
    for node in data['nodes']:
        nid = node['id']
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
    plt.title("Chain of Thought Visualization", fontsize=20, fontweight='bold', pad=20)
    
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
    
    # Save the figure with high quality as PNG
    img_path = os.path.join(output_dir, "cot_graph.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved PNG visualization to: {img_path}")
    
    # Also save as SVG (vector format)
    svg_path = os.path.join(output_dir, "cot_graph.svg")
    plt.savefig(svg_path, bbox_inches='tight', format='svg')
    print(f"Saved SVG visualization to: {svg_path}")
    
    # Show the graph
    plt.show()

if __name__ == "__main__":
    # Default input path
    default_path = "/home/szhang967/liftr/reasoning_analysis/sample_cot.json"
    
    # Use command line argument if provided, otherwise use default
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # Extract the directory from the input path
    output_dir = os.path.dirname(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    draw_chain_of_thought(data, output_dir)
