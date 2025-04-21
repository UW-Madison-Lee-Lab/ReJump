import graphviz

class MLModelNode:
    def __init__(self, id, label, node_type=None):
        self.id = id
        self.label = label
        self.node_type = node_type  # Can be 'root', 'category', 'subcategory', 'basis', 'activation', etc.
        self.children = []
    
    def add_child(self, child_node):
        """Add a child node to this node"""
        self.children.append(child_node)
        return child_node

class MLModelGraph:
    def __init__(self):
        self.nodes = {}  # Dictionary of all nodes: {id: node_object}
        self.root = None
        self.custom_path = []  # Store the custom path sequence
    
    def add_node(self, id, label, node_type=None, parent_id=None):
        """Add a node to the graph, optionally as a child of parent_id"""
        node = MLModelNode(id, label, node_type)
        self.nodes[id] = node
        
        if parent_id is None and self.root is None:
            self.root = node
        elif parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node)
        else:
            raise ValueError(f"Parent node with ID '{parent_id}' not found")
        
        return node
    
    def add_custom_path(self, node_sequence):
        """
        Add a custom path that connects nodes in the specified sequence.
        
        Parameters:
        - node_sequence: List of node IDs to connect in order
        
        Returns:
        - Self for method chaining
        """
        # Validate that all nodes in the sequence exist
        for node_id in node_sequence:
            if node_id not in self.nodes:
                raise ValueError(f"Node '{node_id}' not found in the graph")
        
        self.custom_path = node_sequence
        return self
    
    def visualize(self):
        """Visualize the graph using Graphviz"""
        dot = graphviz.Digraph(comment='ML Model Hierarchy')
        dot.attr('node', shape='ellipse')
        
        # Define node styles based on type
        node_styles = {
            'root': {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightblue'},
            'category': {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'},
            'subcategory': {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow'},
            'model': {'shape': 'ellipse'},
        }
        
        # Add all nodes to the visualization
        for node_id, node in self.nodes.items():
            node_attr = node_styles.get(node.node_type, {})
            dot.node(node_id, node.label, **node_attr)
        
        # Add all regular edges (parent-child relationships)
        for node_id, node in self.nodes.items():
            for child in node.children:
                dot.edge(node_id, child.id)
        
        # Add custom path edges if defined
        if self.custom_path and len(self.custom_path) > 1:
            for i in range(len(self.custom_path) - 1):
                # Add edge with custom styling (red and bold)
                dot.edge(
                    self.custom_path[i], 
                    self.custom_path[i+1], 
                    color='red', 
                    penwidth='2.0',
                    constraint='false'  # This allows edges to be drawn more freely
                )
        
        return dot

# Example usage
def create_example_graph():
    graph = MLModelGraph()
    
    # Create root
    graph.add_node("root", "Root", "root")
    
    # Create main categories
    graph.add_node("A", "Category A", "category", "root")
    graph.add_node("B", "Category B", "category", "root")
    graph.add_node("C", "Category C", "category", "root")
    
    # Create subcategories
    graph.add_node("A1", "Subcategory A1", "subcategory", "A")
    graph.add_node("A2", "Subcategory A2", "subcategory", "A")
    graph.add_node("B1", "Subcategory B1", "subcategory", "B")
    graph.add_node("B2", "Subcategory B2", "subcategory", "B")
    graph.add_node("C1", "Subcategory C1", "subcategory", "C")
    
    # Create leaf nodes
    graph.add_node("A1a", "Model A1a", "model", "A1")
    graph.add_node("A1b", "Model A1b", "model", "A1")
    graph.add_node("A2a", "Model A2a", "model", "A2")
    graph.add_node("B1a", "Model B1a", "model", "B1")
    graph.add_node("B2a", "Model B2a", "model", "B2")
    graph.add_node("C1a", "Model C1a", "model", "C1")
    
    return graph

def main():
    # Create graph
    graph = create_example_graph()
    
    # Define a custom path (the red curve)
    # This path traverses: Root -> A -> A1 -> A1a -> A1b -> A2 -> A2a -> B -> B1 -> B1a
    custom_path = ["root", "A", "A1", "A1a", "A1b", "A2", "A2a", "B", "B1", "B1a"]
    
    # Add the custom path
    graph.add_custom_path(custom_path)
    
    # Visualize and render
    dot = graph.visualize()
    dot.render('example_graph', format='png', cleanup=True)
    print("Graph visualization created as 'example_graph.png'")

if __name__ == "__main__":
    main() 