import os
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import math
from collections import Counter
from tqdm import tqdm


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

    def indegree_ka_bhukha(self, node):
        if self.node_degrees[node][0] >= node/100+1:
            return True

    def create_graph_with_equal_deg(self):
        self.__init__(self.p, self.q, self.steps, self.new_nodes_param, self.initial_nodes)

        node_count = 0
        # dummy_node = 0
        # node_count += 1
        # self.add_node(dummy_node)
        for _ in range(1000):
            self.add_node(node_count)
            node_count+=1

        indegree_ka_bhukha = 0
        node_count2 = 0
        for indegree in range(10):
            for outdegree in range(10):
                for nindex in range(10):
                    indegree_ka_bhukha1 = indegree_ka_bhukha
                    for _ in range(outdegree+1):
                        self.add_edge(node_count2, indegree_ka_bhukha1)
                        indegree_ka_bhukha1 += 1
                        print(node_count2, indegree_ka_bhukha)
                        while self.indegree_ka_bhukha(indegree_ka_bhukha):
                            indegree_ka_bhukha += 1
                    node_count2 += 1
        
        
        # for indegree in range(10):
        #     for outdegree in range(10):
        #         for nindex in range(10):
        #             self.add_node(node_count)
        #             node_count += 1
        #             for _ in range(outdegree):
        #                 print(node_count,"outdegree")
        #                 self.add_edge(node_count-1, dummy_node)
        #             for _ in range(indegree):
        #                 self.add_edge(dummy_node, node_count-1)
        #                 print(node_count,"indegree")
        self.compute_degree_pair_counts()
        print(self.degree_pair_counts_history)
        # print(self.graph)


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

    def simulate(self):
        print("Started")
        self.create_graph_with_equal_deg()
        print("Reached")
        self.compute_total_nodes()
        self.compute_average_in_degree()
        self.compute_average_out_degree()
        self.compute_total_edges()
        self.compute_degree_pair_counts()
        for step in tqdm(range(self.steps), desc="Simulating"):
            new_node = self.total_nodes
            self.add_node(new_node)
            self.total_nodes += 1
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
            self.compute_total_nodes()
            self.compute_average_in_degree()
            self.compute_average_out_degree()
            self.compute_total_edges()
            self.compute_degree_pair_counts()

    def save_parameters(self, output_dir, timestamped_dir):
        parameters = {
            "p": self.p,
            "q": self.q,
            "steps": self.steps,
            "new_nodes_param": self.new_nodes_param,
            "Final number of nodes": self.total_nodes,
            "Initial number of nodes": self.initial_nodes
        }
        # Save to both timestamped and latest directories
        for folder in [timestamped_dir, os.path.join(output_dir, "latest")]:
            with open(os.path.join(folder, "parameters.json"), "w") as pf:
                json.dump(parameters, pf, indent=4)

    def save_results(self, output_dir):
        # Prepare directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = os.path.join(output_dir, timestamp)
        latest_dir = os.path.join(output_dir, "latest")
        os.makedirs(timestamped_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # Save data files
        file_mapping = {
            "graph.json": self.graph,
            "node_degrees.json": self.node_degrees,
            "metrics.json": self.metrics,
            "degree_pair_counts_history.json": [
                {tuple_key_to_str(k): v for k, v in counter.items()}
                for counter in self.degree_pair_counts_history
            ]
        }
        for file_name, data in file_mapping.items():
            for folder in [timestamped_dir, latest_dir]:
                with open(os.path.join(folder, file_name), "w") as f:
                    json.dump(data, f, indent=4)

        # Save parameters
        self.save_parameters(output_dir, timestamped_dir)

    def plot_results(self, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir_timestamped = os.path.join(output_dir, timestamp, "plots")
        plots_dir_latest = os.path.join(output_dir, "latest", "plots")
        os.makedirs(plots_dir_timestamped, exist_ok=True)
        os.makedirs(plots_dir_latest, exist_ok=True)

        for metric, values in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(values)), values, marker="o", linestyle="-", label=metric)
            plt.title(f"{metric.capitalize()} Over Time")
            plt.xlabel("Time Steps")
            plt.ylabel(metric.replace("_", " ").capitalize())
            plt.grid()
            plt.legend()

            # Save plots
            for folder in [plots_dir_timestamped, plots_dir_latest]:
                plt.savefig(os.path.join(folder, f"{metric}.png"))
            plt.close()


# Parameters
p = 0.05
q = 0.09
steps = 10
new_nodes_param = 0.0001
output_folder = "output"
initial_nodes = 1000

# Run simulation
simulator = BJRGraphSimulation(p, q, steps, new_nodes_param, initial_nodes)
simulator.simulate()
simulator.save_results(output_folder)
simulator.plot_results(output_folder)
