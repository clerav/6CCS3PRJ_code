"""Tour constructors for the TSP instances. pyomo is used to solve the TSP instance as an LP (relaxation with subtour elimination) and IP problem (MTZ formulation)
    networkx is used to detect subtours in the LP solution.
"""

import random
import pyomo.environ as pyo
import networkx as nx
from queue import PriorityQueue
import tsp_instance as TSP

# ------------------- Tour constructors -------------------

class TourConstructor:
    def __init__(self, name):
        self.name = name
        self.problem = None
        self.tour = None

    def initialise(self, tsp_instance, options=None):
        self.problem = tsp_instance
        self.tour = None

    def construct_tour(self):
        raise NotImplementedError()

    def get_tour(self):
        raise NotImplementedError()
    
    def get_tour_cost(self):
        if self.tour:
            return self.problem.get_tour_cost(self.build_tour())
        return -1
    
    def get_tc_name(self):
        return self.name


class RandomTourConstructor(TourConstructor):
    def __init__(self):
        super().__init__('Random')

    def construct_tour(self):
        nodes = range(self.problem.dimension)
        self.tour = list(nodes)
        random.shuffle(self.tour)
        self.tour.append(self.tour[0])
        return True
        
    def get_tour(self):
        return self.tour


class NearestNeighborTourConstructor(TourConstructor):
    def __init__(self):
        super().__init__('NearestNeighbor')

    def construct_tour(self): # start_node?
        nodes = range(self.problem.dimension)
        start_node = random.choice(nodes)
        
        self.tour = [start_node]
        nodes = set(nodes)
        nodes.remove(start_node)
        unvisited = nodes

        while unvisited:
            edges = self.problem.get_node_edges(self.tour[-1], exclude_nodes=self.tour)
            edges.sort(key=lambda x: x[0])
            nearest = edges[0][1]
            self.tour.append(nearest)
            unvisited.remove(nearest)

        self.tour.append(self.tour[0])
        return True
        
    def get_tour(self):
        return self.tour


class GreedyTourConstructor(TourConstructor):
    def __init__(self):
        super().__init__('Greedy')
    
    def construct_tour(self):
        nodes = range(self.problem.dimension)
        tour_degrees = {i: [] for i in nodes}
        tour_edges = []
        edges = self.problem.get_all_edges()
        edges.sort(key=lambda x: x[1])

        while len(tour_edges) < self.problem.dimension:
            edge = edges.pop(0)
            if len(tour_degrees[edge[0][0]]) < 2 and len(tour_degrees[edge[0][1]]) < 2:
                introduces_cycle = self.introduces_cycle(edge, tour_degrees)
                completes = introduces_cycle and len(tour_edges) == self.problem.dimension - 1
                if not introduces_cycle or completes:
                    tour_edges.append(edge)
                    tour_degrees[edge[0][0]].append(edge[0][1])
                    tour_degrees[edge[0][1]].append(edge[0][0])
        
        self.tour = self.build_tour(tour_degrees)
        return True
    
    def introduces_cycle(self, edge, tour_degrees):
        start = edge[0][0]
        current = edge[0][1]
        previous = -1
        end_of_path = False
        while current != start and not end_of_path:
            current_connections = tour_degrees[current]
            if len(current_connections) == 0:
                end_of_path = True
            elif len(current_connections) == 1:
                if current_connections[0] != previous:
                    previous = current
                    current = current_connections[0]
                else:
                    end_of_path = True
            elif current_connections[0] == previous:
                previous = current
                current = current_connections[1]
            else:
                previous = current
                current = current_connections[0]
        return current == start
    
    def build_tour(self, tour_degrees):
        tour = [tour_degrees[0][0], 0, tour_degrees[0][1]]
        previous = 0
        current = tour_degrees[0][1]
        while len(tour) < self.problem.dimension:
            if tour_degrees[current][0] == previous:
                previous = current
                current = tour_degrees[current][1]
            else:
                previous = current
                current = tour_degrees[current][0]
            tour.append(current)
        tour.append(tour[0])
        return tour
        

    def get_tour(self):
        return self.tour


class FarthestInsertionTourConstructor(TourConstructor):
    def __init__(self):
        super().__init__('FarthestInsertion')

    def construct_tour(self, save_history=False):
        edges = self.problem.get_all_edges() # format: [ [(i, j), weight] ]
        edges.sort(key=lambda x: x[1])

        current_tour_edges = [edges[-1], [(edges[-1][0][1], edges[-1][0][0]), edges[-1][1]]]
        edges.pop(-1)

        current_tour_cities = [
            current_tour_edges[0][0][0], 
            current_tour_edges[0][0][1], 
            current_tour_edges[0][0][0]
        ]
        available_cities = [[i, min(self.problem.get_edge_weight(i, current_tour_cities[0]), self.problem.get_edge_weight(i, current_tour_cities[1]))] for i in range(self.problem.dimension) if i not in current_tour_cities]

        while len(current_tour_cities) < self.problem.dimension + 1:
            if save_history:
                self.history.append((current_tour_cities.copy(), current_tour_edges.copy()))

            available_cities.sort(key=lambda x: x[1])
            best_city = available_cities.pop(-1)
            best_replacement_edges = None
            best_edge_replacement_cost = float('inf')
            best_edge_index = None

            for i in range(len(current_tour_edges)):
                current_edge = current_tour_edges[i]
                current_replacement_edges = [[(current_edge[0][0], best_city[0]), self.problem.get_edge_weight(current_edge[0][0], best_city[0])], [(best_city[0], current_edge[0][1]), self.problem.get_edge_weight(current_edge[0][1], best_city[0])]]
                replacement_edges_cost = current_replacement_edges[0][1] + current_replacement_edges[1][1] - current_edge[1]
                
                if replacement_edges_cost < best_edge_replacement_cost:
                    best_edge_index = i
                    best_edge_replacement_cost = replacement_edges_cost
                    best_replacement_edges = current_replacement_edges

            current_tour_edges.pop(best_edge_index)
            current_tour_edges.insert(best_edge_index, best_replacement_edges[1])
            current_tour_edges.insert(best_edge_index, best_replacement_edges[0])
            current_tour_cities.insert(best_edge_index + 1, best_city[0])

            for city in available_cities:
                city[1] = min(city[1], self.problem.get_edge_weight(city[0], best_city[0]))
        
        if save_history:
                self.history.append((current_tour_cities, current_tour_edges))
        self.tour = current_tour_cities
        return True

    def get_tour(self):
        return self.tour
    
    def get_history(self):
        return self.history


class ChristofidesAlgorithmTourConstructor(TourConstructor):
    def __init__(self):
        super().__init__('Christofides')
    
    def construct_tour(self):
        mst = self.build_mst()
        odd_degree_vertices = self.get_odd_degree_vertices(mst)
        perfect_matching = self.get_perfect_matching(odd_degree_vertices)
        eulerian_walk = self.get_eulerian_tour(mst, perfect_matching)
        self.tour = self.get_hamiltonian_tour(eulerian_walk)
        return True
    
    def build_mst(self):
        edges = self.problem.get_all_edges()
        edges.sort(key=lambda x: x[1])
        vertex_sets = {vertex: vertex for vertex in range(self.problem.dimension)}

        mst_edges = []
        mst_node_connections = {vertex: [] for vertex in range(self.problem.dimension)}
        
        while len(edges) > 0:
            current_edge = edges.pop(0)
            a, b = current_edge[0]
            current_set_a = vertex_sets[a]
            current_set_b = vertex_sets[b]

            if current_set_a != current_set_b:
                for vertex in vertex_sets:
                    if vertex_sets[vertex] == current_set_b:
                        vertex_sets[vertex] = current_set_a
                mst_edges.append(current_edge)
                mst_node_connections[a].append(b)
                mst_node_connections[b].append(a)
        
        return mst_node_connections, mst_edges
            # check if the edge connects two different sets (i.e. does not create a cycle)
            # if it does, add it to the mst
        
    def get_odd_degree_vertices(self, mst):
        # change to list comprehension
        odd_degree_vertices = []
        for vertex in mst[0]:
            if len(mst[0][vertex]) % 2 == 1:
                odd_degree_vertices.append(vertex)
        return odd_degree_vertices

    def get_perfect_matching(self, odd_degree_vertices):
        # naive greedy implementation
        perfect_matching_edges = []
        edges = []
        for i in range(len(odd_degree_vertices) - 1):
            for j in range(i + 1, len(odd_degree_vertices)):
                edges.append([(odd_degree_vertices[i], odd_degree_vertices[j]), self.problem.get_edge_weight(odd_degree_vertices[i], odd_degree_vertices[j])])
        
        edges.sort(key=lambda x: x[1])
        while len(odd_degree_vertices) > 0:
            current_edge = edges.pop(0)
            a, b = current_edge[0]
            if a in odd_degree_vertices and b in odd_degree_vertices:
                perfect_matching_edges.append(current_edge)
                odd_degree_vertices.remove(a)
                odd_degree_vertices.remove(b)
        
        return perfect_matching_edges

    def get_eulerian_tour(self, mst, perfect_matching):
        for edge in perfect_matching:
            a, b = edge[0]
            mst[0][a].append(b)
            mst[0][b].append(a)
    
        eulerian_tour = []
        stack = [0]
        while len(stack) > 0:
            if len(mst[0][stack[-1]]) == 0:
                eulerian_tour.append(stack.pop())
            else:
                other_end = mst[0][stack[-1]].pop()
                mst[0][other_end].remove(stack[-1])
                stack.append(other_end)
        
        return eulerian_tour
    
    def get_hamiltonian_tour(self, eulerian_tour):
        hamiltonian_tour = []
        travel_set = set()
        for i in range(len(eulerian_tour) - 1):
            vertex = eulerian_tour[i]
            before = len(travel_set)
            travel_set.add(vertex)
            if len(travel_set) > before:
                hamiltonian_tour.append(vertex)
        hamiltonian_tour.append(eulerian_tour[-1])
        return hamiltonian_tour
    
    def get_tour(self):
        return self.tour

# ----------------------- LP solvers -----------------------


# ---------------------------------
# model building functions
def total_cost(model):
    return pyo.summation(model.distances, model.x)

def indegree_constraint(model, i):
    return sum(model.x[j, i] for j in model.nodes if i != j) == 1

def outdegree_constraint(model, i):
    return sum(model.x[i, j] for j in model.nodes if i != j) == 1

def total_degree_constraint(model, i):
    l = [model.x[j, i] for j in range(i)]
    l += [model.x[i, j] for j in range(i+1, len(model.nodes))]
    return sum(l) == 2

def tour_city_order_constraint(model, i, j):
    if i != 0 and j != 0 and i != j:
        return model.u[i] - model.u[j] + 1 <= (len(model.nodes) - 1) * (1 - model.x[i, j])
    return pyo.Constraint.Skip
    
def generate_lean_symmetric_edge_list(n):
    edge_list = []
    for i in range(n - 1):
        for j in range(i+1, n):
            edge_list.append((i, j))
    return edge_list

def generate_complete_edge_list_without_self_loops(n):
    edge_list = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_list.append((i, j))
    return edge_list

# ---------------------------------


class LPTSP_MTZ_Solver(TourConstructor):
    def __init__(self, solver):
        super().__init__('IP_MTZ')
        self.solver = solver
        self.model = None

    def initialise(self, tsp_instance):
        super().initialise(tsp_instance)
        self.model = self.build_LP()

    def build_LP(self):
        # build concrete model
        model = pyo.ConcreteModel()

        # define index sets
        model.nodes = pyo.RangeSet(0, self.problem.dimension - 1)
        model.node_order = pyo.RangeSet(1, self.problem.dimension - 1)
        model.edges = pyo.Set(initialize=generate_complete_edge_list_without_self_loops(self.problem.dimension))

        # define cost parameters
        model.distances = pyo.Param(model.edges, initialize=lambda model, i, j: self.problem.get_edge_weight(i, j))

        # define decision variables accoding to MTZ formulation
        model.x = pyo.Var(model.edges, domain=pyo.Binary)
        model.u = pyo.Var(model.node_order, domain=pyo.NonNegativeIntegers, bounds=(1, self.problem.dimension - 1))

        # define objective function
        model.objective = pyo.Objective(rule=total_cost, sense=pyo.minimize)

        # define constraints on node degrees
        model.outdegree_constraints = pyo.Constraint(model.nodes, rule=outdegree_constraint)
        model.indegree_constraints = pyo.Constraint(model.nodes, rule=indegree_constraint)

        # define order constraints for 'u' variables accoding to MTZ formulation
        model.order_constraint = pyo.Constraint(model.node_order, model.node_order, rule=tour_city_order_constraint)

        self.model = model
        return model
    
    def solve(self):
        results = self.solver.solve(self.model)
        self.solution = results
    
    def build_tour(self):
        tour = [-1 for _ in range(self.problem.dimension + 1)]
        if self.solution.solver.status == pyo.SolverStatus.ok:
            tour[0] = 0
            tour[-1] = 0
            
            for i in range(1, self.problem.dimension):
                order = round(self.model.u[i].value)
                tour[order] = i
        return tour

    def construct_tour(self):
        self.solve()
        self.tour = self.build_tour()
        return True

    def get_tour(self):
        return self.tour
    

class LPTSP_SubtourEliminationBranchAndCut_Solver(TourConstructor):
    def __init__(self, solver, upper_bound_constructor=None):
        name = 'BranchAndCut_SE'
        name = name + f'_{upper_bound_constructor.name}' if upper_bound_constructor is not None else name
        super().__init__(name)
        self.solver = solver
        self.base_model = None
        self.valid_model = None
        self.solution = None
        self.upper_bound_constructor = upper_bound_constructor      
    
    def initialise(self, tsp_instance):
        super().initialise(tsp_instance)
        self.build_LP()
        self.compute_upper_bound()
        self.valid_model = None
        self.solution = None
    
    def compute_upper_bound(self):
        if self.upper_bound_constructor is not None:
            self.upper_bound_constructor.initialise(self.problem)
            solved = self.upper_bound_constructor.construct_tour()
            self.upper_bound = TSP.get_tour_length(self.problem, self.upper_bound_constructor.get_tour()) if solved else None
        else:
            self.upper_bound = None

    def build_LP(self):
        # build concrete model
        model = pyo.ConcreteModel()

        # define index sets
        model.nodes = pyo.RangeSet(0, self.problem.dimension - 1)
        model.edges = pyo.Set(initialize=generate_lean_symmetric_edge_list(self.problem.dimension))

        # define cost parameters
        model.distances = pyo.Param(model.edges, initialize=lambda model, i, j: self.problem.get_edge_weight(i, j))

        # define edge decision variables
        model.x = pyo.Var(model.edges, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # define constraints on node degrees with LP relaxation
        model.degree_constraints = pyo.Constraint(model.nodes, rule=total_degree_constraint)
        model.subtour_constraints = pyo.ConstraintList()

        # define objective function
        model.objective = pyo.Objective(rule=total_cost, sense=pyo.minimize)
    
        self.base_model = model
    
    def solve(self):
        # solve root problem
        base_results = self.solver.solve(self.base_model)
        base_objective = self.base_model.objective()
        base_entry = (base_objective, id(self.base_model), (self.base_model, base_results))
        
        # initialize priority queue with root problem to perform branch and cut
        problems = PriorityQueue()
        problems.put(base_entry)
        
        while not problems.empty():
            # select problem with best objective value
            current_entry = problems.get()

            (current_objective, current_id, (current_problem, current_results)) = current_entry

            # apply cutting planes by cutting any emerged subtours
            current_entry = self.cut_subtours(current_entry)
            (current_objective, current_id, (current_problem, current_results)) = current_entry

            # check if problem is feasible after cutting planes
            if current_results.solver.status == pyo.SolverStatus.ok:
                upper_bound_check = self.upper_bound is None or current_objective <= self.upper_bound
                if upper_bound_check:
                    solution_edges = self.get_solution_edges(current_problem)
                    
                    # check if problem is integer feasible
                    branch_edge = self.get_fractional_edge_to_branch_on(solution_edges)

                    # if problem is integer feasible, we have found an optimal solution, otherwise we branch
                    if branch_edge:
                        self.branch(current_entry, branch_edge, problems)
                    else:
                        self.valid_model = current_problem
                        self.solution = current_results
                        return current_results
        return None

    def cut_subtours(self, entry):
        (current_objective, current_id, (current_problem, current_results)) = entry
        solution_edges = self.get_solution_edges(current_problem)

        # find a minimum cut corresponding to a subtour in the current solution
        partition = self.find_subtour_minimum_cut(solution_edges)

        # while a subtour is found, add a subtour elimination constraint and solve the problem again
        while partition:
            self.add_subtour_elimination_constraint(current_problem, partition)
            current_results = self.solver.solve(current_problem)
            
            # check if problem is feasible
            if current_results.solver.status == pyo.SolverStatus.ok:
                solution_edges = self.get_solution_edges(current_problem)
                
                # find a minimum cut corresponding to a new subtour in the solution to the updated problem
                partition = self.find_subtour_minimum_cut(solution_edges)
            else:
                partition = None
        
        return (current_problem.objective(), current_id, (current_problem, current_results))

    def get_solution_edges(self, model):
        solution_edges = []
        for i,j in model.x:
            if model.x[i,j].value > 0.0:
                solution_edges.append(((i, j), model.x[i,j].value))
        return solution_edges

    def find_subtour_minimum_cut(self, solution_edges):
        tolerance = 1e-6
        graph = nx.Graph()
        for edge in solution_edges:
            graph.add_edge(edge[0][0], edge[0][1], capacity=edge[1])
            graph.add_edge(edge[0][1], edge[0][0], capacity=edge[1])

        for i in range(1, self.problem.dimension):
            cut_value, partition = nx.minimum_cut(graph, 0, i)
            if cut_value < 2.0 - tolerance:
                return partition
        
        return None
    
    def add_subtour_elimination_constraint(self, model, partition):
        separation_list = []
        for i in partition[0]:
            for j in partition[1]:
                edge_key = (i, j) if i < j else (j, i)
                separation_list.append(model.x[edge_key])
        model.subtour_constraints.add(sum(separation_list) >= 2.0)

    def get_fractional_edge_to_branch_on(self, solution_edges):
        fractionals = PriorityQueue()
        for edge in solution_edges:
            if edge[1] < 1.0:
                # entry structure (edge weight, (edge nodes, edge value))
                fractionals.put((-self.problem.get_edge_weight(edge[0][0], edge[0][1]), (edge[0], edge[1])))
        
        if not fractionals.empty():
            (weight, (key, value)) = fractionals.get()
            return [weight, key, value]
        else:
            return None
    
    def branch(self, current_entry, branch_edge, problems):
        (current_objective, current_id, (current_problem, current_results)) = current_entry

        lower_problem = current_problem.clone()
        lower_problem.x[branch_edge[1]].fix(0.0)
        lower_results = self.solver.solve(lower_problem)
        lower_objective = lower_problem.objective()
        
        upper_bound_check = self.upper_bound is None or lower_objective <= self.upper_bound
        if upper_bound_check:
            lower_entry = (lower_objective, id(lower_problem), (lower_problem, lower_results))
            problems.put(lower_entry)


        upper_problem = current_problem.clone()
        upper_problem.x[branch_edge[1]].fix(1.0)
        upper_results = self.solver.solve(upper_problem)
        upper_objective = upper_problem.objective()
        
        upper_bound_check = self.upper_bound is None or upper_objective <= self.upper_bound
        if upper_bound_check:
            upper_entry = (upper_objective, id(upper_problem), (upper_problem, upper_results))
            problems.put(upper_entry)

    def construct_tour(self):
        result = self.solve()
        if result:
            self.tour = self.build_tour()
            return True
        return False

    def build_tour(self):
        tour = []
        if self.solution.solver.status == pyo.SolverStatus.ok:
            node_connections = {i: [] for i in range(self.problem.dimension)}
            solution_edges = self.get_solution_edges(self.valid_model)
            for ((i, j), value) in solution_edges:
                node_connections[i].append(j)
                node_connections[j].append(i)
            
            current_node = 0
            previous_node = -1
            tour.append(current_node)
            while len(tour) < self.problem.dimension:
                next_node = node_connections[current_node][0] if node_connections[current_node][0] != previous_node else node_connections[current_node][1]
                tour.append(next_node)
                previous_node = current_node
                current_node = next_node
            tour.append(0)
        return tour

    def compute_lower_bound(self):
        model = self.base_model.clone()
        model_id = id(model)
        results = self.solver.solve(model)
        if results.solver.status == pyo.SolverStatus.ok:
            entry = (model.objective(), model_id, (model, results))
            (lower_bound, model_id, (model, results)) = self.cut_subtours(entry)
            if results.solver.status == pyo.SolverStatus.ok:
                return lower_bound
        return -1
    
    def get_tour(self):
        return self.tour