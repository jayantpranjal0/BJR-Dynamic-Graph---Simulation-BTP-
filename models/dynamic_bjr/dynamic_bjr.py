import numpy as np
import random
from typing import List, Dict, Tuple, Optional


class Node:
    def __init__(self, node_id: int, vector: np.ndarray):
        self.node_id = node_id
        self.vector = vector

    def __repr__(self):
        return f"Node({self.node_id}, vector={self.vector})"


class Edge:
    def __init__(self, node_u: Node, node_v: Node):
        self.node_u = node_u
        self.node_v = node_v

    def __repr__(self):
        return f"Edge({self.node_u.node_id} - {self.node_v.node_id}, prob={self.probability})"


class DynamicBJR:
    def __init__(self, k: int, isolated_vector: np.ndarray):
        self.k = k
        self.nodes: Dict[int, Node] = {}  # Store all nodes, including isolated nodes
        self.edges: List[Edge] = []       # Store edges
        self.isolated_vector = isolated_vector  # Vector for isolated nodes

        # Initialize with two isolated nodes
        self._initialize_isolated_nodes()

    # @classmethod
    # def from_dataset(cls, k: int, isolated_vector: np.ndarray, dataset: Dict):
    #     """Creates a DynamicBJR graph from a dataset."""
    #     instance = cls(k, isolated_vector)

    #     # Add nodes from dataset
    #     for node_id, vector in dataset.get("nodes", {}).items():
    #         instance.add_node_with_vector(node_id, np.array(vector))

    #     # Add edges from dataset
    #     for edge_data in dataset.get("edges", []):
    #         node_u_id, node_v_id, probability = edge_data
    #         instance.add_edge_by_id(node_u_id, node_v_id, probability)

    #     # Ensure isolated nodes are replenished
    #     instance._replenish_isolated_nodes()

    #     return instance

    def _initialize_isolated_nodes(self):
        """Ensures that there are at least two isolated nodes with the given vector in the graph."""
        for _ in range(2):
            self._add_isolated_node()

    def _replenish_isolated_nodes(self):
        """Ensures there are always two isolated nodes with the specified vector."""
        isolated_count = sum(1 for node in self.nodes.values() if np.array_equal(node.vector, self.isolated_vector))
        while isolated_count < 2:
            self._add_isolated_node()
            isolated_count += 1

    def add_node(self, node_vector) -> Node:
        """Adds a new node with a given vector"""
        node_id = len(self.nodes)
        new_node = Node(node_id, node_vector)
        self.nodes[node_id] = new_node
        return new_node

    
    def delete_node(self, id: int):
        """Deletes a node from the graph."""
        if id in self.nodes:
            del self.nodes[id]
        else:
            raise ValueError(f"Node ID {id} does not exist.")

    def _add_isolated_node(self):
        """Adds an isolated node with the given vector to the nodes dictionary."""
        self.add_node(self.isolated_vector)

    def add_node_with_vector(self, node_id: int, vector: np.ndarray):
        """Adds a node with a specific vector to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists.")
        self.nodes[node_id] = Node(node_id, vector)

    def add_random_node(self) -> Node:
        """Adds a new node with a random vector to the graph."""
        node_id = len(self.nodes)
        new_vector = np.random.rand(self.k)
        self.add_node(new_vector)
        
    def add_edge(self, source: Node, destination: Node) -> Edge:
        """Adds an edge between two nodes."""
        edge = Edge(source, destination)
        self.edges.append(edge)
        if source == self.isolated_vector:
            self._replenish_isolated_nodes
        if destination == self.isolated_vector:
            self._replenish_isolated_nodes
        return edge
    
    def delete_edge(self, edge: Edge):
        """Deletes an edge from the graph."""
        if edge in self.edges:
            self.edges.remove(edge)
        else:
            raise ValueError("Edge does not exist in the graph.")


    def edge_probability(self, node_u: Node, node_v: Node) -> float:
        """Calculates edge probability based on distance between node vectors.
        Returns a value between -1 and 1, where the absolute value represents the action
        and the sign represents whether the edge should be added or deleted."""
        distance = np.linalg.norm(node_u.vector - node_v.vector)
        max_distance = np.sqrt(len(node_u.vector))  # Maximum possible distance in the vector space
        normalized_distance = distance / max_distance
        probability = 2 * (normalized_distance - 0.5)  # Map to range [-1, 1]
        return probability
    
    def simulate(self, num_steps: int):
        """Simulates the dynamic graph over a specified number of steps."""
        for step in range(num_steps):
           for source in self.nodes.values():
               for destination in self.nodes.values():
                   if source.node_id == destination.node_id:
                       continue
                   prob = self.edge_probability(source, destination)
                   if random.random() < prob:
                       self.add_edge(source, destination, prob)
    def __repr__(self):
        return f"DynamicBJR(nodes={len(self.nodes)}, edges={len(self.edges)})"