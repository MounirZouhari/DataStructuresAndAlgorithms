

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def num_spanning_trees(adj_matrix):
    # Create Graph from adjacency matrix
    G = nx.Graph()
    num_nodes = len(adj_matrix)
    
    # Add nodes and edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    
    # Compute the Laplacian matrix
    laplacian = nx.laplacian_matrix(G).toarray()
    
    # Compute the determinant of any (n-1)x(n-1) minor of the Laplacian matrix
    laplacian_minor = laplacian[1:, 1:]  # Remove first row and first column
    
    # Calculate number of spanning trees using Kirchhoff's Matrix Tree Theorem
    num_trees = round(np.linalg.det(laplacian_minor))
    return int(num_trees)

# Example adjacency matrix
adj_matrix = [
     [0,1,0,0,0,1,1,0,0,0,0, 0],
     [1,0,1,1,0,0,0,1,0,0,0, 0],
     [0,1,0,1,0,0,0,0,1,0,0, 0],
     [0,1,1,0,1,0,0,0,0,1,0, 0],
     [0,0,0,1,0,1,0,0,0,0,1, 0],
     [1,0,0,0,1,0,0,0,0,0,0, 1],
     [1,0,0,0,0,0,0,0,0,0,0, 0],
     [0,1,0,0,0,0,0,0,0,0,0, 0],
     [0,0,1,0,0,0,0,0,0,0,0, 0],
     [0,0,0,1,0,0,0,0,0,0,0, 0],
     [0,0,0,0,1,0,0,0,0,0,0, 0],
     [0,0,0,0,0,1,0,0,0,0,0, 0]
]

# Calculate and print number of spanning trees
num_trees = num_spanning_trees(adj_matrix)
print("Number of spanning trees:", num_trees)

# Create graph from adjacency matrix
G = nx.Graph()
num_nodes = len(adj_matrix)

# Add nodes
G.add_nodes_from(range(num_nodes))

# Add edges
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if adj_matrix[i][j] == 1:
            G.add_edge(i, j)

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title("Graph from Adjacency Matrix")
plt.show()