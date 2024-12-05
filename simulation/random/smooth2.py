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

class BJRGraphSimulation:
    def __init__(self, p, q, steps, new_nodes_param, initial_nodes=10):
        self.initial_nodes = initial_nodes
        self.p = p  # Probability of adding an edge
        self.q = q  # Probability of removing an edge
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
        # List of metric computation functions
        self.metric_functions = [
            self.compute_total_nodes,
            self.compute_average_in_degree,
            self.compute_average_out_degree,
            self.compute_total_edges
        ]
        
        # # Initialize the graph with the initial nodes
        # for node in range(self.initial_nodes):
        #     self.add_node(node)
        

    def create_graph_with_equal_deg(self):
        self.__init__(self.p, self.q, self.steps, self.new_nodes_param, self.initial_nodes)

        node_count = 0
        dummy_node = 0
        node_count += 1
        self.add_node(dummy_node)

        
        for indegree in range(10):
            for outdegree in range(10):
                for nindex in range(10):
                    self.add_node(node_count)
                    node_count += 1
                    for i in range(indegree):
                        self.add_edge(node_count-1, dummy_node)
                    for i in range(outdegree):
                        self.add_edge(dummy_node, node_count-1)

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.node_degrees[node] = [0, 0]  # [in-degree, out-degree]

    def add_edge(self, from_node, to_node):
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)
            self.node_degrees[from_node][1] += 1  # Increment out-degree
            self.node_degrees[to_node][0] += 1  # Increment in-degree

    def remove_edge(self, from_node, to_node):
        if to_node in self.graph[from_node]:
            self.graph[from_node].remove(to_node)
            self.node_degrees[from_node][1] -= 1  # Decrement out-degree
            self.node_degrees[to_node][0] -= 1  # Decrement in-degree

    def remove_orphan_nodes(self):
        """Remove nodes with no edges."""
        to_remove = [node for node, (in_degree, out_degree) in self.node_degrees.items() if in_degree == 0 and out_degree == 0]
        for node in to_remove:
            del self.graph[node]
            del self.node_degrees[node]

    # Metric computation functions
    def compute_total_nodes(self):
        self.metrics["total_nodes"].append(len(self.node_degrees))

    def compute_average_in_degree(self):
        if len(self.node_degrees) > 0:
            avg_in_degree = sum(deg[0] for deg in self.node_degrees.values()) / len(self.node_degrees)
            self.metrics["average_in_degree"].append(avg_in_degree)
        else:
            self.metrics["average_in_degree"].append(0)

    def compute_average_out_degree(self):
        if len(self.node_degrees) > 0:
            avg_out_degree = sum(deg[1] for deg in self.node_degrees.values()) / len(self.node_degrees)
            self.metrics["average_out_degree"].append(avg_out_degree)
        else:
            self.metrics["average_out_degree"].append(0)

    def compute_total_edges(self):
        total_edges = sum(len(edges) for edges in self.graph.values())
        self.metrics["total_edges"].append(total_edges)

    def compute_degree_pair_counts(self):
        """Compute and store counts of (in-degree, out-degree) pairs at the current timestep."""
        degree_pair_counts = Counter()
        for in_degree, out_degree in self.node_degrees.values():
            degree_pair_counts[(in_degree, out_degree)] += 1
        self.degree_pair_counts_history.append(degree_pair_counts)

    def compute_metrics(self):
        """Compute and store all metrics for the current timestep."""
        for metric_function in self.metric_functions:
            metric_function()

    def compute_degree_pair_counts(self):
        """Compute and store counts of (in-degree, out-degree) pairs at the current timestep."""
        degree_pair_counts = Counter()
        for in_degree, out_degree in self.node_degrees.values():
            degree_pair_counts[(in_degree, out_degree)] += 1
        self.degree_pair_counts_history.append(degree_pair_counts)

    def simulate(self):
        print("Started")
        self.create_graph_with_equal_deg()
        print("Reached")
        print(self.steps)
        for step in tqdm.tqdm(range(self.steps)):
            # Add a random number of new nodes
            # new_nodes = random.random() * self.total_nodes * self.new_nodes_param
            # for _ in range(math.floor(new_nodes)):
            #     new_node = self.total_nodes
            #     self.add_node(new_node)
            #     self.total_nodes += 1
            # if random.random() < new_nodes - math.floor(new_nodes):
            #     new_node = self.total_nodes
            #     self.add_node(new_node)
            #     self.total_nodes += 1

            # print(f"Step {step + 1}/{self.steps}: {len(self.graph)} nodes, {sum(len(edges) for edges in self.graph.values())} edges")
            # Iterate over all pairs of nodes
            for i in list(self.graph.keys()):
                for j in list(self.graph.keys()):
                    if i != j:
                        # Add edges with probability `p`
                        if j not in self.graph[i] and random.random() < self.p:
                            self.add_edge(i, j)
                            
                        # Remove edges with probability `q`
                        if j in self.graph[i] and random.random() < self.q:
                            self.remove_edge(i, j)

            # Remove orphan nodes
            self.remove_orphan_nodes()

            # Compute metrics and degree pair counts for this timestep
            self.compute_metrics()
            self.compute_degree_pair_counts()

    def save_parameters(self, output_dir, timestamped_dir):
        """Save simulation parameters."""
        parameters = {
            "p": self.p,
            "q": self.q,
            "steps": self.steps,
            "new_nodes_param": self.new_nodes_param,
            "Final number of nodes": self.total_nodes,
            "Initial number of nodes": self.initial_nodes
        }

        params_file_timestamped = os.path.join(timestamped_dir, "parameters.json")
        params_file_latest = os.path.join(output_dir, "latest", "parameters.json")

        # Save to timestamped and latest directories
        with open(params_file_timestamped, "w") as pf:
            json.dump(parameters, pf, indent=4)
        with open(params_file_latest, "w") as pf:
            json.dump(parameters, pf, indent=4)

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

        # Save parameters
        self.save_parameters(output_dir, timestamped_dir)

        # Update latest directory
        for file_name in ["graph.json", "node_degrees.json", "metrics.json", "degree_pair_counts_history.json"]:
            src = os.path.join(timestamped_dir, file_name)
            dst = os.path.join(latest_dir, file_name)
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
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


# # Parameters
p = 0.005  # Probability of edge addition
q = 0.05  # Probability of edge removal
steps = 1000  # Number of simulation steps
new_nodes_param = 0  # Factor for determining new nodes
output_folder = "output"
initial_nodes = 100  # Initial number of nodes

# Take parameters as input
# p = float(input("Enter the probability of edge addition (p): "))
# q = float(input("Enter the probability of edge removal (q): "))
# steps = int(input("Enter the number of simulation steps: "))
# new_nodes_param = float(input("Enter the factor for determining new nodes: "))
# output_folder = input("Enter the output folder: ")
# initial_nodes = int(input("Enter the initial number of nodes: "))

# Run simulation
simulator = BJRGraphSimulation(p, q, steps, new_nodes_param, initial_nodes)
simulator.simulate()
simulator.save_results(output_folder)
simulator.plot_results(output_folder)