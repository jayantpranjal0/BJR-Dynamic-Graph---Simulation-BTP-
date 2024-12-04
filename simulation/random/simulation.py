import os
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import math
from collections import Counter

def tuple_key_to_str(key):
    return f"({key[0]},{key[1]})"

class BJRGraphSimulation:
    def __init__(self, p, steps, new_nodes_param, initial_nodes=10):
        self.p = p  # Probability of adding an edge
        self.steps = steps  # Number of time steps
        self.new_nodes_param = new_nodes_param  # Factor for determining new nodes
        self.graph = {}  # Graph represented as an adjacency list
        self.node_degrees = {}  # Dictionary to store node degrees
        self.total_nodes = initial_nodes  # Start with `initial_nodes` as the initial total nodes
        self.degree_pair_counts_history = []  # History of (in-degree, out-degree) counts per time step
        self.metrics = {  # Store metrics for each timestep
            "total_nodes": [],
            "average_in_degree": [],
            "average_out_degree": [],
            "total_edges": []
        }

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.node_degrees[node] = [0, 0]  # [in-degree, out-degree]

    def add_edge(self, from_node, to_node):
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)
            self.node_degrees[from_node][1] += 1  # Increment out-degree
            self.node_degrees[to_node][0] += 1  # Increment in-degree

    def compute_metrics(self):
        """Compute and store metrics at the current timestep."""
        self.metrics["total_nodes"].append(self.total_nodes)
        avg_in_degree = sum(deg[0] for deg in self.node_degrees.values()) / len(self.node_degrees)
        self.metrics["average_in_degree"].append(avg_in_degree)
        avg_out_degree = sum(deg[1] for deg in self.node_degrees.values()) / len(self.node_degrees)
        self.metrics["average_out_degree"].append(avg_out_degree)
        total_edges = sum(len(edges) for edges in self.graph.values())
        self.metrics["total_edges"].append(total_edges)

    def compute_degree_pair_counts(self):
        """Compute and store counts of (in-degree, out-degree) pairs at the current timestep."""
        degree_pair_counts = Counter()
        for in_degree, out_degree in self.node_degrees.values():
            degree_pair_counts[(in_degree, out_degree)] += 1
        self.degree_pair_counts_history.append(degree_pair_counts)

    def simulate(self):
        for step in range(self.steps):
            # Add a random number of new nodes
            new_nodes = random.random() * self.total_nodes * self.new_nodes_param
            for _ in range(math.floor(new_nodes)):
                new_node = self.total_nodes
                self.add_node(new_node)
                self.total_nodes += 1
            if random.random() < new_nodes - math.floor(new_nodes):
                new_node = self.total_nodes
                self.add_node(new_node)
                self.total_nodes += 1

            # Iterate over all pairs of nodes and add edges
            for i in range(self.total_nodes):
                for j in range(self.total_nodes):
                    if i != j:
                        self.add_node(i)
                        self.add_node(j)
                        if j not in self.graph[i] and random.random() < self.p:
                            self.add_edge(i, j)

            # Compute metrics and degree pair counts for this timestep
            self.compute_metrics()
            self.compute_degree_pair_counts()

    def save_results(self, output_dir):
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = os.path.join(output_dir, timestamp)
        latest_dir = os.path.join(output_dir, "latest")

        os.makedirs(timestamped_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # Save graph, node degrees, metrics, and degree pair counts
        graph_file = os.path.join(timestamped_dir, "graph.json")
        degrees_file = os.path.join(timestamped_dir, "node_degrees.json")
        metrics_file = os.path.join(timestamped_dir, "metrics.json")
        degree_pairs_file = os.path.join(timestamped_dir, "degree_pair_counts_history.json")

        with open(graph_file, "w") as gf:
            json.dump(self.graph, gf, indent=4)
        with open(degrees_file, "w") as df:
            json.dump(self.node_degrees, df, indent=4)
        with open(metrics_file, "w") as mf:
            json.dump(self.metrics, mf, indent=4)
        with open(degree_pairs_file, "w") as dpf:
            json.dump(
                [
                    {tuple_key_to_str(k): v for k, v in counter.items()} 
                    for counter in self.degree_pair_counts_history
                ],
                dpf,
                indent=4,
            )

        # Update latest directory
        for file_name in ["graph.json", "node_degrees.json", "metrics.json", "degree_pair_counts_history.json"]:
            src = os.path.join(timestamped_dir, file_name)
            dst = os.path.join(latest_dir, file_name)
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)

    def plot_parameters(self, output_dir):
        metrics_file = os.path.join(output_dir, "latest", "metrics.json")
        with open(metrics_file, "r") as mf:
            metrics = json.load(mf)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir_timestamped = os.path.join(output_dir, timestamp, "plots")
        plots_dir_latest = os.path.join(output_dir, "latest", "plots")

        os.makedirs(plots_dir_timestamped, exist_ok=True)
        os.makedirs(plots_dir_latest, exist_ok=True)

        for parameter_name, parameter_values in metrics.items():
            plt.figure()
            plt.plot(range(len(parameter_values)), parameter_values, label=parameter_name)
            plt.xlabel("Time Step")
            plt.ylabel(parameter_name)
            plt.title(f"{parameter_name} Over Time")
            plt.legend()
            plt.grid(True)

            plot_file_timestamped = os.path.join(plots_dir_timestamped, f"{parameter_name}.png")
            plot_file_latest = os.path.join(plots_dir_latest, f"{parameter_name}.png")

            plt.savefig(plot_file_timestamped)
            plt.savefig(plot_file_latest)
            plt.close()

# Parameters
p = 0.05  # Probability of edge addition
steps = 100  # Number of simulation steps
new_nodes_param = 0.05  # Factor for determining new nodes
output_folder = "output"
initial_nodes = 10  # Initial number of nodes

# Run simulation
simulator = BJRGraphSimulation(p, steps, new_nodes_param, initial_nodes)
simulator.simulate()
simulator.save_results(output_folder)
simulator.plot_parameters(output_folder)
