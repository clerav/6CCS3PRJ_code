"""Contains class that imlements TSP instance and functions for generating random instances."""

import math
import random
import copy
from enum import Enum

def generate_random_EUC_2D_TSP_instance(n, region_size, name=None, seed=None):
    if seed:
        random.seed(seed)
    nodes = [[random.uniform(0, region_size), random.uniform(0, region_size)] for _ in range(n)]
    return EuclideanTSPInstance(nodes=nodes, name=name)

def get_tour_length(tsp, tour):
    length = 0
    for i in range(len(tour) - 1):
        length += tsp.get_edge_weight(tour[i], tour[i+1])
    return length


class TSPInstance:
    def __init__(self, edge_weights, name=None, optimal_tour_length=None):
        self.set_edge_weights(edge_weights)
        self.name = name
        self.optimal_tour_length = optimal_tour_length
    
    def get_edge_weight(self, i, j):
        if i == j:
            return 0
        elif i < j:
            return self.edge_weights[i][j-i-1]
        else:
            return self.edge_weights[j][i-j-1]
    
    def get_node_edges(self, i, exclude_nodes=None):
        result = [[self.edge_weights[j][i - j - 1], j] for j in range(i) \
                   if (not exclude_nodes or j not in exclude_nodes)]

        if i < self.dimension - 1:
            result = result + [[self.edge_weights[i][k], k + i + 1] for k in range(len(self.edge_weights[i])) \
                               if (not exclude_nodes or k + i + 1 not in exclude_nodes)]

        return result

    def get_all_edges(self):
        result = []
        for i in range(self.dimension - 1):
            result = result + [[(i, j + i + 1), self.edge_weights[i][j]] for j in range(len(self.edge_weights[i]))]
        return result
    
    def set_edge_weights(self, edge_weights):
        self.dimension = len(edge_weights) + 1
        self.edge_weights = edge_weights
    
    def set_edge_weight(self, i, j, weight):
        if i == j:
            return
        elif i < j:
            self.edge_weights[i][j-i-1] = weight
        else:
            self.edge_weights[j][i-j-1] = weight

    def get_edge_weights_matrix_copy(self):
        return copy.deepcopy(self.edge_weights)

    def print_weight_matrix_v1(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                print(self.get_edge_weight(i, j), end=' ')
            print()

    def print_weight_matrix_v2(self):
        for i in range(self.dimension - 1):
            print(self.edge_weights[i])


class EuclideanTSPInstance(TSPInstance):
    def __init__(self, nodes, name=None, optimal_tour_length=None):
        self.nodes = nodes
        self.dimension = len(nodes)
        weights = self.calculate_euclidean_edge_weights()

        name_start = f'euc_{self.dimension}'
        name = name_start if not name else name_start + '_' + name
        
        super().__init__(weights, name=name, optimal_tour_length=optimal_tour_length)

    def calculate_euclidean_edge_weight(self, i, j):
        return round(math.sqrt((self.nodes[j][0] - self.nodes[i][0]) ** 2 + (self.nodes[j][1] - self.nodes[i][1]) ** 2))
    
    def calculate_euclidean_edge_weights(self):
        weights = [ [] for _ in range(self.dimension - 1)]
        for i in range(self.dimension - 1):
            for j in range(i+1, self.dimension):
                weights[i].append(self.calculate_euclidean_edge_weight(i, j))
        
        return weights
    
class IntervalExpectationType(Enum):
    MIN = 'MIN'
    MAX = 'MAX'
    AVG = 'AVG'

class DistributionType(Enum):
    UNIFORM = 'UNIFORM'
    NORMAL = 'NORMAL'
    
class IntervalTSPInstance(TSPInstance):
    DISTRIBUTION_SAMPLING_FUNCTIONS = {
        DistributionType.UNIFORM.value: lambda a, b: random.randint(a, b),
        DistributionType.NORMAL.value: lambda a, b: round(random.normalvariate((a + b) / 2, (b - a) / 6))
    }

    def __init__(self, interval_edge_weights, name=None):
        super().__init__(interval_edge_weights, name=name)
        self.EXPECTATION_FUNCTIONS = {
            IntervalExpectationType.MIN.value: self.get_min_edge_weights_matrix,
            IntervalExpectationType.MAX.value: self.get_max_edge_weights_matrix,
            IntervalExpectationType.AVG.value: self.get_avg_edge_weights_matrix
        }

    @staticmethod
    def generate_edge_weights_intervals(dimension, interval_function):
        weights = [[] for _ in range(dimension - 1)]
        for i in range(dimension - 1):
            for j in range(i + 1, dimension):
                weights[i].append(interval_function(i, j))
        return weights

    def get_weight_expectation_function(self, expectation_type):
        return self.EXPECTATION_FUNCTIONS[expectation_type]

    def get_min_edge_weights_matrix(self):
        return [[min(edge) for edge in row] for row in self.edge_weights]
    
    def get_max_edge_weights_matrix(self):
        return [[max(edge) for edge in row] for row in self.edge_weights]
    
    def get_avg_edge_weights_matrix(self):
        return [[round(sum(edge) / 2) for edge in row] for row in self.edge_weights]
    
    def get_expected_TSP(self, expectation_type=IntervalExpectationType.AVG):
        new_name = str(self.name) + '_' + expectation_type
        expectation_function = self.get_weight_expectation_function(expectation_type)
        new_weight_matrix = expectation_function()
        return TSPInstance(new_weight_matrix, name=new_name)
    
    def get_sample_TSP(self, distribution_type=DistributionType.UNIFORM, name=None):
        new_name = str(self.name) + '_' + distribution_type
        new_name = new_name + '_' + name if name else new_name
        sampling_function = self.DISTRIBUTION_SAMPLING_FUNCTIONS[distribution_type]
        new_weight_matrix = [[sampling_function(edge[0], edge[1]) for edge in row] for row in self.edge_weights]
        return TSPInstance(new_weight_matrix, name=new_name)

class DeterministicBasedIntervalTSPInstance(IntervalTSPInstance):
    def __init__(self, base_instance, interval_bounds_multiplier):
        self.base_instance = base_instance
        self.interval_bounds_multiplier = interval_bounds_multiplier
        weights = IntervalTSPInstance.generate_edge_weights_intervals(
            self.base_instance.dimension,
            lambda i, j: [
                random.randint(round((1 - self.interval_bounds_multiplier) * base_instance.get_edge_weight(i, j)), round(base_instance.get_edge_weight(i, j))),
                random.randint(round(base_instance.get_edge_weight(i, j)), round((1 + self.interval_bounds_multiplier) * base_instance.get_edge_weight(i, j)))
            ]    
        )
        super().__init__(weights)


class RandomIntervalTSPInstance(IntervalTSPInstance):
    def __init__(self, dimension, base_range, name=None):
        weights = IntervalTSPInstance.generate_edge_weights_intervals(
            dimension,
            lambda i, j: [
                random.randint(0, base_range),
                random.randint(0, base_range) + base_range
            ]
        )
        super().__init__(weights, name=name)
