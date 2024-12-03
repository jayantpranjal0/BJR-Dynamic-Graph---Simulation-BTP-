import os
import json
import random
from datetime import datetime

class BJRGraphSimulation:
    def __init__(self, p, steps, new_nodes_param, initial_nodes=10):
        self.p = p  # Probability of adding an edge
        self.steps = steps  # Number of time steps
        self.new_nodes_param = new_nodes_param  # Factor for determining new nodes
        self.graph = {}  # Graph represented as an adjacency list
        self.node_degrees = {}  # Dictionary to store node degrees
        self.total_nodes = initial_nodes  # Start with `max_nodes` as the initial total nodes

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.node_degrees[node] = [0, 0]  # [in-degree, out-degree]

    def add_edge(self, from_node, to_node):
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)
            self.node_degrees[from_node][1] += 1  # Increment out-degree
            self.node_degrees[to_node][0] += 1  # Increment in-degree

    def simulate(self):
        for step in range(self.steps):
            # Add a random number of new nodes
            new_nodes = random.randint(0, int(self.total_nodes * self.new_nodes_param))
            for _ in range(new_nodes):
                new_node = self.total_nodes
                self.add_node(new_node)
                self.total_nodes += 1

            # Iterate over all pairs of nodes and add edges
            for i in range(self.total_nodes):
                for j in range(self.total_nodes):
                    if i != j:
                        self.add_node(i)
                        self.add_node(j)

                        # Add edge with probability p if it doesn't exist
                        if j not in self.graph[i] and random.random() < self.p:
                            self.add_edge(i, j)

    def save_results(self, output_dir):
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = os.path.join(output_dir, timestamp)
        latest_dir = os.path.join(output_dir, "latest")

        os.makedirs(timestamped_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # Save graph and node degrees
        graph_file = os.path.join(timestamped_dir, "graph.json")
        degrees_file = os.path.join(timestamped_dir, "node_degrees.json")

        with open(graph_file, "w") as gf:
            json.dump(self.graph, gf, indent=4)

        with open(degrees_file, "w") as df:
            json.dump(self.node_degrees, df, indent=4)

        # Update latest directory
        for file_name in ["graph.json", "node_degrees.json"]:
            src = os.path.join(timestamped_dir, file_name)
            dst = os.path.join(latest_dir, file_name)
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)

# Parameters
p = 0.05  # Probability of edge addition
steps = 1000  # Number of simulation steps
new_nodes_param = 0.1  # Factor for determining new nodes
output_folder = "output"
initial_nodes = 10  # Initial number of nodes

# Run simulation
simulator = BJRGraphSimulation(p, steps, new_nodes_param, initial_nodes)
simulator.simulate()
simulator.save_results(output_folder)
