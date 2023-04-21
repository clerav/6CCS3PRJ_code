""" Experiment module for running experiments on TSP instances with algorithms."""

import json

import time 
import tsp_instance as TSP

class InstanceRun:
    """ InstanceRun class for running a list of tour constructors with a list of tour improvements on a single TSP instance.

    Computes the following results:
    - Lower bound for the instance if the lower bound calculator is provided.
    - Tour constructor metrics for each tour constructor:
        - Tour
        - Tour length
        - Tour construction time
        - Margin of the tour length over the lower bound
        - Improvement metrics for the performance of each tour improvement on the tour:
            - Whether the tour was improved
            - Improvement time
            - The reduction in the margin of the tour length over the lower bound
            - Number of alterations to the tour

    
    Args:
        tsp_instance (TSP.TSPInstance): TSP instance to run the tour constructors and improvements on.
        tour_constructors (list): List of tour constructors to run.
        tour_improvements (list): List of tour improvements to run.
        lower_bound_calculator (TSP.LPTSP_BranchAndCutSubtourEliminationConstrains_Solver ): Lower bound calculator to use.

    """
    def __init__(self, tsp_instance, tour_constructors, tour_improvements, lower_bound_calculator=None):
        self.tsp_instance = tsp_instance
        self.tsp_optimal_tour_length = tsp_instance.optimal_tour_length

        self.tour_constructors = tour_constructors
        self.tour_improvements = tour_improvements

        self.lower_bound_calculator = lower_bound_calculator

        self.tour_constructors_results = {tc.name: None for tc in tour_constructors}
        self.tour_improvements_results = {ti.name: {} for ti in tour_improvements}

    def initialise_constructors(self):
        for tc in self.tour_constructors:
            tc.initialise(self.tsp_instance)
    
    def compute_lower_bound(self):
        if self.lower_bound_calculator:
            self.lower_bound_calculator.initialise(self.tsp_instance)
            self.lower_bound = self.lower_bound_calculator.compute_lower_bound()
        else:
            self.lower_bound = 1

    def initialise(self):
        self.compute_lower_bound()
        self.initialise_constructors()
    
    def run(self):
        self.initialise()
        self.run_constructors()
        self.run_improvements()
    
    def run_constructors(self):
        for tc in self.tour_constructors:
            start_time = time.time()
            result = tc.construct_tour()
            end_time = time.time()

            tour = tc.get_tour() if result else None
            tour_length = TSP.get_tour_length(self.tsp_instance, tour) if result else None
            bound_margin = tour_length / self.lower_bound if result else None
            optimal_tour_margin = tour_length / self.tsp_optimal_tour_length if result and self.tsp_optimal_tour_length else None
            
            result_record = {
                'time': end_time - start_time,
                'result': result,
                'tour': tour,
                'tour_length': tour_length,
                'bound_margin': bound_margin,
                'optimal_tour_margin': optimal_tour_margin,
            }

            self.tour_constructors_results[tc.name] = result_record
        
        
    
    def run_improvements(self):
        for ti in self.tour_improvements:
            for tc_result in self.tour_constructors_results:
                result_record = {
                    'time': None,
                    'result': None,
                    'improved_tour': None,
                    'improvement_count': 0,
                    'improved_tour_length': None,
                    'bound_margin': None,
                    'bound_margin_improvement': 0,
                }

                if self.tour_constructors_results[tc_result]['result']:
                    tour = self.tour_constructors_results[tc_result]['tour']
                    ti.initialise(self.tsp_instance, tour)
                    
                    start_time = time.time()
                    result = ti.improve_tour()
                    end_time = time.time()
                    
                    result_record['time'] = end_time - start_time
                    result_record['result'] = result
                    if result:
                        improved_tour = ti.get_improved_tour()
                        improvement_count = ti.get_improvement_count()
                        improved_tour_length = TSP.get_tour_length(self.tsp_instance, improved_tour)
                        bound_margin = improved_tour_length / self.lower_bound
                        bound_margin_improvement = self.tour_constructors_results[tc_result]['bound_margin'] - bound_margin
                    
                        result_record['improved_tour'] = improved_tour
                        result_record['improvement_count'] = improvement_count
                        result_record['improved_tour_length'] = improved_tour_length
                        result_record['bound_margin'] = bound_margin
                        result_record['bound_margin_improvement'] = bound_margin_improvement

                self.tour_improvements_results[ti.name][tc_result] = result_record
    
    def get_results(self):
        return {
            'tsp_name': str(self.tsp_instance.name),
            'lower_bound': self.lower_bound,
            'optimal_tour_length': self.tsp_optimal_tour_length,
            'tour_constructors': self.tour_constructors_results,
            'tour_improvements': self.tour_improvements_results
        }

class Experiment:
    """Experiment class for running experiments on TSP instances.

    Creates InstanceRun objects for each TSP instance and runs the experiment on them.
    Once the experiment is run, the results can be accessed through the experiment_results attribute.
    Computes the following results:
        - Constructor cross results:
            - The average running time on the experiment instances for each constructor.
            - The average margin from the lower bound on the experiment instances for each constructor.

        - Improvement cross results:
            - Separated by constructor:
                - The average running time on the tour generated by the constructor
                - The average reduction in the margin from the lower bound on the tour generated by the constructor (separately all improvements and only successful improvements)
                - Count and ratio of successful improvement runs on the tour generated by the constructor

    Args:
        tsp_instances (list): List of TSP instances to run the experiment on.
        tour_constructors (list): List of tour constructors to run the experiment with.
        tour_improvements (list): List of tour improvements to run the experiment with.
        lower_bound_calculator (TSP.LPTSP_BranchAndCutSubtourEliminationConstrains_Solver): TSP LP solver to use for lower bound calculations in the experiment.
    """
    def __init__(self, tsp_instances, tour_constructors, tour_improvements, lower_bound_calculator=None):
        self.tsp_instances = tsp_instances
        self.tour_constructors = tour_constructors
        self.tour_improvements = tour_improvements

        self.lower_bound_calculator = lower_bound_calculator

        self.instance_runs = [InstanceRun(tsp_instance, tour_constructors, tour_improvements, lower_bound_calculator=lower_bound_calculator) for tsp_instance in tsp_instances]
        self.instance_results = []


    
    def run(self):
        for instance_run in self.instance_runs:
            instance_run.run()
            self.instance_results.append(instance_run.get_results())
        
        self.evaluate()
    
    def evaluate(self):
        constructor_cross_results = self.evaluate_constructor_cross_results()
        improvement_cross_results = self.evaluate_improvement_cross_results()

        self.experiment_results = {
            'constructor_cross_results': constructor_cross_results,
            'improvement_cross_results': improvement_cross_results,
            'instance_results': self.instance_results
        }

    def evaluate_constructor_cross_results(self):
        constructor_cross_results = {tc.name: {'times': [], 'bound_margins': []} for tc in self.tour_constructors}
        for instance in self.instance_results:
            for tc in instance['tour_constructors']:
                constructor_cross_results[tc]['times'].append(instance['tour_constructors'][tc]['time'])
                constructor_cross_results[tc]['bound_margins'].append(instance['tour_constructors'][tc]['bound_margin'])
        
        for tc in constructor_cross_results:
            constructor_cross_results[tc]['mean_time'] = sum(constructor_cross_results[tc]['times']) / len(constructor_cross_results[tc]['times'])
            constructor_cross_results[tc]['mean_bound_margin'] = sum(constructor_cross_results[tc]['bound_margins']) / len(constructor_cross_results[tc]['bound_margins'])
        
        return constructor_cross_results
    
    def evaluate_improvement_cross_results(self):
        improvement_cross_results = {ti.name: {tc.name: {'times': [], 'success_count': 0, 'bound_margin_improvements': []} for tc in self.tour_constructors} for ti in self.tour_improvements}
        for instance in self.instance_results:
            for ti in instance['tour_improvements']:
                for tc in instance['tour_improvements'][ti]:
                    if instance['tour_improvements'][ti][tc]['result']:
                         improvement_cross_results[ti][tc]['success_count'] += 1

                    improvement_cross_results[ti][tc]['times'].append(instance['tour_improvements'][ti][tc]['time'])
                    improvement_cross_results[ti][tc]['bound_margin_improvements'].append(instance['tour_improvements'][ti][tc]['bound_margin_improvement'])
        
        for ti in improvement_cross_results:
            for tc in improvement_cross_results[ti]:
                improvement_cross_results[ti][tc]['mean_time'] = sum(improvement_cross_results[ti][tc]['times']) / len(improvement_cross_results[ti][tc]['times'])
                improvement_cross_results[ti][tc]['mean_total_bound_margin_improvement'] = sum(improvement_cross_results[ti][tc]['bound_margin_improvements']) / len(improvement_cross_results[ti][tc]['bound_margin_improvements'])
                improvement_cross_results[ti][tc]['mean_nonzero_bound_margin_improvement'] = sum(improvement_cross_results[ti][tc]['bound_margin_improvements']) / improvement_cross_results[ti][tc]['success_count'] if improvement_cross_results[ti][tc]['success_count'] > 0 else 0
                improvement_cross_results[ti][tc]['success_rate'] = improvement_cross_results[ti][tc]['success_count'] / len(self.instance_results)
        
        return improvement_cross_results
    
    def get_results(self):
        return self.experiment_results
    
    def save_results(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_results(), file)

class IntervalInstanceRun():
    def __init__(self, interval_tsp_intance, expectation_types, distribution_types, sample_size, tour_constructors, lower_bound_calculator=None):
        self.interval_tsp_instance = interval_tsp_intance
        self.expectation_types = expectation_types
        self.distribution_types = distribution_types
        self.tour_constructors = tour_constructors
        self.lower_bound_calculator = lower_bound_calculator
        self.sample_size = sample_size
        
    
    def generate_expected_tsp_tours(self):
        expected_instances = {}
        for exp_type in self.expectation_types:
            expected_instance_results = {}
            expected_tsp = self.interval_tsp_instance.get_expected_TSP(exp_type)

            for tc in self.tour_constructors:
                tc_result = {
                    'result': None,
                    'tour': None,
                }

                tc.initialise(expected_tsp)
                result = tc.construct_tour()
                tc_result['result'] = result
                if result:
                    tc_result['tour'] = tc.get_tour()
                
                expected_instance_results[tc.name] = tc_result
            
            expected_instances[exp_type] = expected_instance_results
        self.expected_instances = expected_instances
    
    # ------------------------------------
    
    def generate_scenarios(self):
        self.scenarios = {}
        for dist_type in self.distribution_types:
            dist_scenarios = []
            for i in range(self.sample_size):
                scenario = self.interval_tsp_instance.get_sample_TSP(dist_type, name=f'i{i}')
                self.lower_bound_calculator.initialise(scenario)
                lower_bound = self.lower_bound_calculator.compute_lower_bound()
                dist_scenarios.append((scenario, lower_bound))
            self.scenarios[dist_type] = dist_scenarios

    def evaluate_scenarios(self):
        self.raw_scenario_results = {}
        self.scenario_results = {}
        for exp_type in self.expected_instances:
            tc_results = {}
            raw_tc_results = {}
            for tc_name in self.expected_instances[exp_type]:
                tc_tour = self.expected_instances[exp_type][tc_name]['tour']
                dist_margins = {}
                raw_dist_margins = {}
                for dist_type in self.scenarios:
                    bound_margins = []
                    for (scenario, lower_bound) in self.scenarios[dist_type]:
                        tour_length = TSP.get_tour_length(scenario, tc_tour)
                        bound_margin = tour_length / lower_bound
                        bound_margins.append(bound_margin)
                    
                    mean_bound_margin = sum(bound_margins) / len(bound_margins)
                    dist_margins[dist_type] = mean_bound_margin
                    raw_dist_margins[dist_type] = bound_margins
                
                tc_results[tc_name] = dist_margins
                raw_tc_results[tc_name] = raw_dist_margins
        
            self.scenario_results[exp_type] = tc_results
            self.raw_scenario_results[exp_type] = raw_tc_results
    
    def run(self):
        self.generate_expected_tsp_tours()
        self.generate_scenarios()
        self.evaluate_scenarios()


    def get_results(self):
        return {
            'expected_instances': self.expected_instances,
            'scenario_results': self.scenario_results,
            'raw_scenario_results': self.raw_scenario_results,
        }

    def save_results(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_results(), file)

class IntervalExperiment:
    def __init__(self, interval_tsp_intances, expectation_types, distribution_types, sample_size, tour_constructors, lower_bound_calculator=None):
        self.interval_tsp_instances = interval_tsp_intances
        self.expectation_types = expectation_types
        self.distribution_types = distribution_types
        self.tour_constructors = tour_constructors
        self.lower_bound_calculator = lower_bound_calculator
        self.sample_size = sample_size
    
    def run_instances(self):
        self.instance_results = []
        for instance in self.interval_tsp_instances:
            instance_run = IntervalInstanceRun(instance, self.expectation_types, self.distribution_types, self.sample_size, self.tour_constructors, self.lower_bound_calculator)
            instance_run.run()
            self.instance_results.append(instance_run.get_results())
    
    def evaluate(self):
        self.experiment_results = {}
        for exp_type in self.expectation_types:
            exp_results = {}
            for tc in self.tour_constructors:
                bounds = {}
                for dist_type in self.distribution_types:
                    dist_type_bounds = []
                    for instance in self.instance_results:
                        dist_type_bounds.append(instance['scenario_results'][exp_type][tc.name][dist_type])
                    dist_bound_margin = sum(dist_type_bounds) / len(dist_type_bounds)
                    bounds[dist_type] = dist_bound_margin
                exp_results[tc.name] = bounds
            self.experiment_results[exp_type] = exp_results

    def run(self):
        self.run_instances()
        self.evaluate()
    
    def get_results(self):
        return {
            'experiment_results': self.experiment_results,
            'instance_results': self.instance_results,
        }
    
    def save_results(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_results(), file)