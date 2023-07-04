# import networkx as nx
# import matplotlib.pyplot as plt

# def construct_star_graph(n):
#     G = nx.Graph()
    
#     # Add center node
#     center_node = str(n)
#     G.add_node(center_node)
    
#     # Add outer nodes
#     for i in range(1, n):
#         node = str(i)
#         G.add_node(node)
#         G.add_edge(center_node, node)
    
#     return G

# # Example usage
# n = 4  # Number of nodes
# star_graph = construct_star_graph(n)

# # Visualize the graph
# nx.draw(star_graph, with_labels=True)
# plt.show()

import numpy as np
import networkx as nx
from itertools import tee, product, chain, islice
from collections import deque
import matplotlib.pyplot as plt


def unit_disk_grid_graph(grid, radius=np.sqrt(2) + 1e-5, periodic=False, visualize=True):
    x = grid.shape[1]
    y = grid.shape[0]

    def neighbors_from_geometry(n):
        """Identify the neighbors within a unit distance of the atom at index (i, j) (zero-indexed).
        Returns a numpy array listing both the geometric graph of the neighbors, and the indices of the
        neighbors of the form [[first indices], [second indices]]"""
        # Assert that we actually have an atom at this location
        assert grid[n[0], n[1]] != 0
        grid_x, grid_y = np.meshgrid(np.arange(x), np.arange(y))
        # a is 1 if the location is within a unit distance of (i, j), and zero otherwise
        a = np.sqrt((grid_x - n[1]) ** 2 + (grid_y - n[0]) ** 2) <= radius
        if periodic:
            b = np.sqrt((np.abs(grid_x - n[1]) - x) ** 2 + (grid_y - n[0]) ** 2) <= radius
            a = a + b
            b = np.sqrt((grid_x - n[1]) ** 2 + (np.abs(grid_y - n[0]) - y) ** 2) <= radius
            a = a + b
            b = np.sqrt((np.abs(grid_x - n[1]) - x) ** 2 + (np.abs(grid_y - n[0]) - y) ** 2) <= radius
            a = a + b
        # Remove the node itself
        a[n[0], n[1]] = 0
        # a is 1 if  within a unit distance of (i, j) and a node is at that location, and zero otherwise
        a = a * grid
        return np.argwhere(a != 0)

    nodes_geometric = np.argwhere(grid != 0)
    nodes = list(range(len(nodes_geometric)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    j = 0
    for node in nodes_geometric:
        neighbors = neighbors_from_geometry(node)
        neighbors = [np.argwhere(np.all(nodes_geometric == i, axis=1))[0, 0] for i in neighbors]
        for neighbor in neighbors:
            graph.add_edge(j, neighbor)
        j += 1

    if visualize:
        pos = {nodes[i]: nodes_geometric[i] for i in range(len(nodes))}
        nx.draw_networkx_nodes(graph, pos=pos, node_color='white', node_size=120,
                               edgecolors='black')  
        nx.draw_networkx_edges(graph, pos=pos, edge_color='black')
        nx.draw_networkx_labels(graph, pos=pos)  # Added this line to draw labels
        plt.axis('off')
        plt.show()
    # TODO: assess whether it's an issue to add random nx.Graph attributes
    

    graph.positions = grid
    graph.radius = radius
    graph.periodic = periodic
    return graph


