"""Tour improvement algorithms for TSP."""

class TourImprovement():
    def __init__(self, name):
        self.name = name
        self.problem = None
        self.tour = None
        self.improved_tour = None
        self.improved = False
        self.improvement_count = 0

    def initialise(self, tsp, tour):
        self.problem = tsp
        self.tour = tour
        self.improved_tour = tour.copy()
        self.improved = False
        self.improvement_count = 0

    def get_improved_tour(self):
        raise NotImplementedError()
    
    def improve_tour(self):
        raise NotImplementedError()

class TwoOpt(TourImprovement): # speed-up is available through nearest neighbour heuristic
    def __init__(self):
        super().__init__('2-opt')

    def improve_tour(self):
        finished = False
        while not finished:
            finished = True
            for i in range(len(self.improved_tour) - 3):
                end = len(self.improved_tour) - 2 if i == 0 else len(self.improved_tour) - 1
                for j in range(i + 2, end):
                    a_1, a_2 = self.improved_tour[i], self.improved_tour[i + 1]
                    b_1, b_2 = self.improved_tour[j], self.improved_tour[j + 1]
                    current_weight = self.problem.get_edge_weight(a_1, a_2) + self.problem.get_edge_weight(b_1, b_2)
                    new_weight = self.problem.get_edge_weight(a_1, b_1) + self.problem.get_edge_weight(a_2, b_2)
                    if new_weight < current_weight:
                        self.perform_2opt(i, j)
                        self.improvement_count += 1
                        self.improved = True
                        finished = False
        return self.improved

    def perform_2opt(self, i, j):
        self.improved_tour[i+1:j+1] = self.improved_tour[j:i:-1]
    
    def get_improved_tour(self):
        return self.improved_tour
    
    def get_improvement_count(self):
        return self.improvement_count