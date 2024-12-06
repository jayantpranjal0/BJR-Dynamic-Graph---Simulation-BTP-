import os
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import math
from collections import Counter
import tqdm
def tuple_key_to_str(key):
    return f"({key[0]},{key[1]})"

class DirectedBAGraphSimulation:
    def __init__(self, m, steps, initial_nodes=10):
        self.m = m  # Number of edges each new node adds
        self.steps = steps  # Number of time steps
        self.graph = {}  # Directed graph as adjacency list {node: [outgoing_neighbors]}
        self.in_degrees = {}  # In-degree of each node
        self.out_degrees = {}  # Out-degree of each node
        self.total_nodes = initial_nodes  # Start with `initial_nodes`
        self.degree_pair_counts = Counter()  # Count (in-degree, out-degree) pairs
        self.metrics = {  # Store metrics for each timestep
            "total_nodes": [],
            "average_in_degree": [],
            "average_out_degree": [],
            "total_edges": []
        }

        # Initialize the graph with initial nodes (fully connected)
        for i in range(initial_nodes):
            self.add_node(i)
        for i in range(initial_nodes):
            for j in range(initial_nodes):
                if i != j:
                    self.add_edge(i, j)

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.in_degrees[node] = 0
            self.out_degrees[node] = 0

    def add_edge(self, from_node, to_node):
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)
            
            # Update degrees
            self.out_degrees[from_node] += 1
            self.in_degrees[to_node] += 1

            # Update degree pairs
            self.degree_pair_counts[(self.in_degrees[to_node] - 1, self.out_degrees[to_node])] -= 1
            self.degree_pair_counts[(self.in_degrees[to_node], self.out_degrees[to_node])] += 1

            self.degree_pair_counts[(self.in_degrees[from_node], self.out_degrees[from_node] - 1)] -= 1
            self.degree_pair_counts[(self.in_degrees[from_node], self.out_degrees[from_node])] += 1

    def compute_metrics(self):
        """Compute and store metrics for the current timestep."""
        # Total nodes
        self.metrics["total_nodes"].append(len(self.graph))

        # Average in-degree
        avg_in_degree = sum(self.in_degrees.values()) / len(self.in_degrees) if self.in_degrees else 0
        self.metrics["average_in_degree"].append(avg_in_degree)

        # Average out-degree
        avg_out_degree = sum(self.out_degrees.values()) / len(self.out_degrees) if self.out_degrees else 0
        self.metrics["average_out_degree"].append(avg_out_degree)

        # Total edges
        total_edges = sum(len(edges) for edges in self.graph.values())
        self.metrics["total_edges"].append(total_edges)

    def simulate(self):
        for step in tqdm.tqdm(range(self.steps)):
            new_node = self.total_nodes
            self.add_node(new_node)
            self.total_nodes += 1

            # Add `m` incoming edges (preferential attachment based on in-degree)
            existing_nodes = list(self.in_degrees.keys())
            total_in_degree = sum(self.in_degrees.values())
            in_connection_probabilities = [
                self.in_degrees[node] / total_in_degree if total_in_degree > 0 else 1 / len(existing_nodes)
                for node in existing_nodes
            ]

            incoming_connected_nodes = set()
            while len(incoming_connected_nodes) < self.m:
                chosen_node = random.choices(existing_nodes, weights=in_connection_probabilities, k=1)[0]
                if chosen_node not in incoming_connected_nodes:
                    incoming_connected_nodes.add(chosen_node)
                    self.add_edge(chosen_node, new_node)

            # Add `m` outgoing edges (preferential attachment based on out-degree)
            total_out_degree = sum(self.out_degrees.values())
            out_connection_probabilities = [
                self.out_degrees[node] / total_out_degree if total_out_degree > 0 else 1 / len(existing_nodes)
                for node in existing_nodes
            ]

            outgoing_connected_nodes = set()
            while len(outgoing_connected_nodes) < self.m:
                chosen_node = random.choices(existing_nodes, weights=out_connection_probabilities, k=1)[0]
                if chosen_node not in outgoing_connected_nodes:
                    outgoing_connected_nodes.add(chosen_node)
                    self.add_edge(new_node, chosen_node)

            # Compute metrics for this timestep
            self.compute_metrics()

    def save_results(self, output_dir):
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = os.path.join(output_dir, timestamp)
        latest_dir = os.path.join(output_dir, "latest")

        os.makedirs(timestamped_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # Save graph, degrees, metrics, and degree pair counts
        graph_file = os.path.join(timestamped_dir, "graph.json")
        in_degrees_file = os.path.join(timestamped_dir, "in_degrees.json")
        out_degrees_file = os.path.join(timestamped_dir, "out_degrees.json")
        metrics_file = os.path.join(timestamped_dir, "metrics.json")
        degree_pair_counts_file = os.path.join(timestamped_dir, "degree_pair_counts.json")

        with open(graph_file, "w") as gf:
            json.dump(self.graph, gf, indent=4)
        with open(in_degrees_file, "w") as idf:
            json.dump(self.in_degrees, idf, indent=4)
        with open(out_degrees_file, "w") as odf:
            json.dump(self.out_degrees, odf, indent=4)
        with open(metrics_file, "w") as mf:
            json.dump(self.metrics, mf, indent=4)
        with open(degree_pair_counts_file, "w") as dpf:
            # Convert tuple keys to strings
            degree_pair_counts_str_keys = {tuple_key_to_str(k): v for k, v in self.degree_pair_counts.items()}
            json.dump(degree_pair_counts_str_keys, dpf, indent=4)

        # Copy all files to the "latest" folder
        for file in [graph_file, in_degrees_file, out_degrees_file, metrics_file, degree_pair_counts_file]:
            os.system(f"cp {file} {latest_dir}")

    def plot_results(self, output_dir):
        """Generate plots for the simulation metrics."""
        # Create a timestamped directory for plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir_timestamped = os.path.join(output_dir, timestamp, "plots")
        plots_dir_latest = os.path.join(output_dir, "latest", "plots")
        os.makedirs(plots_dir_timestamped, exist_ok=True)
        os.makedirs(plots_dir_latest, exist_ok=True)

        # Plot each metric
        for metric, values in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(values)), values, marker='o', linestyle='-', label=metric)
            plt.title(f"{metric.capitalize()} Over Time")
            plt.xlabel("Time Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.grid()
            plt.legend()

            # Save the plot in both directories
            plot_file_timestamped = os.path.join(plots_dir_timestamped, f"{metric}.png")
            plot_file_latest = os.path.join(plots_dir_latest, f"{metric}.png")
            plt.savefig(plot_file_timestamped)
            plt.savefig(plot_file_latest)
            plt.close()

        print(f"Plots saved in:\n  - {plots_dir_timestamped}\n  - {plots_dir_latest}")


# Parameters
m = 3  # Number of edges each new node adds
steps = 1000  # Number of simulation steps
output_folder = "output"
initial_nodes = 10  # Initial number of nodes

# Run simulation
simulator = DirectedBAGraphSimulation(m, steps, initial_nodes)
simulator.simulate()
simulator.save_results(output_folder)
simulator.plot_results(output_folder)
