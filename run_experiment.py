"""Module for running experiments. By default uses CPLEX as the LP solver."""

import tsp_instance as TSP
import tsp_tour_constructors as TC
import tsp_tour_improvements as TI
import experiment
import pyomo.environ as pyo
from enum import Enum

import argparse

parser = argparse.ArgumentParser(description='Run experiment.')
parser.add_argument('-t', '--type', help='Experiment type', required=True)
parser.add_argument('-o', '--output', help='Output directory', default='./experiment_results')
parser.add_argument('-s', '--solver', help='LP solver', default='cplex')

args = parser.parse_args()
experiment_type = args.type
output_dir = args.output
output_dir = output_dir[:-1] if output_dir.endswith('/') else output_dir
solver = args.solver


lp_solver = pyo.SolverFactory(solver)

tour_constructors = [ 
    TC.RandomTourConstructor(),
    TC.NearestNeighborTourConstructor(),
    TC.GreedyTourConstructor(),
    TC.FarthestInsertionTourConstructor(),
    TC.ChristofidesAlgorithmTourConstructor(),

    TC.LPTSP_MTZ_Solver(lp_solver),
    TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver),
]

lower_bound_calculator = TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver)


class ExperimentTypes(Enum):
    RANDOM_EUC = 'random_euc'
    RANDOM_INTERVALS = 'intervals'


if experiment_type == ExperimentTypes.RANDOM_EUC.value:
    extra_constructors = [
        TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver, upper_bound_constructor=TC.RandomTourConstructor()),
        TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver, upper_bound_constructor=TC.NearestNeighborTourConstructor()),
        TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver, upper_bound_constructor=TC.GreedyTourConstructor()),
        TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver, upper_bound_constructor=TC.FarthestInsertionTourConstructor()),
        TC.LPTSP_SubtourEliminationBranchAndCut_Solver(lp_solver, upper_bound_constructor=TC.ChristofidesAlgorithmTourConstructor()),
    ]

    tour_constructors += extra_constructors

    tour_improvements = [
        TI.TwoOpt(),
    ]

    random_instance_dimensions = [5, 6, 7, 8, 9, 10, 20, 30, 40]
    random_instance_coordinate_range = 200
    random_instance_iterations = 10

    for dimension in random_instance_dimensions:
        random_deterministic_EUC_2D_instances = [\
            TSP.generate_random_EUC_2D_TSP_instance(dimension, random_instance_coordinate_range, name=f'random-i{i}' ) \
                    for i in range(random_instance_iterations) \
        ]

        random_EUC_2D_experiment = experiment.Experiment(random_deterministic_EUC_2D_instances, tour_constructors, tour_improvements, lower_bound_calculator=lower_bound_calculator)
        output_file = f'{output_dir}/random_EUC_2D_results_{dimension}.json'
        
        print(f'Running experiment for {random_instance_iterations} {dimension}-city random EUC deterministic instances...', end=' ')
        random_EUC_2D_experiment.run()
        print(f'Done, saving results as "{output_file}" ...', end=' ')
        random_EUC_2D_experiment.save_results(output_file)
        print('Done')

elif experiment_type == ExperimentTypes.RANDOM_INTERVALS.value:
    tour_imrpvoements = []
    random_instance_dimensions = [5, 10, 20]
    random_instance_iterations = 10
    random_instance_coordinate_range = 200
    interval_sample_size = 100

    expectation_types = [
        TSP.IntervalExpectationType.AVG.value,
        TSP.IntervalExpectationType.MIN.value,
        TSP.IntervalExpectationType.MAX.value,
    ]

    distribution_types = [
        TSP.DistributionType.UNIFORM.value,
        TSP.DistributionType.NORMAL.value,
    ]

    for dimension in random_instance_dimensions:
        random_interval_instances = [TSP.RandomIntervalTSPInstance(dimension, random_instance_coordinate_range, name=f'random-interval-i{i}') for i in range(random_instance_iterations)]
        exp = experiment.IntervalExperiment(random_interval_instances, expectation_types, distribution_types, interval_sample_size, tour_constructors, lower_bound_calculator=lower_bound_calculator)

        output_file = f'{output_dir}/random_interval_results_{dimension}.json'
        print(f'Running experiment for {random_instance_iterations} {dimension}-city random instances with interval distances...', end=' ')
        exp.run()
        print(f'Done, saving results as "{output_file}" ...', end=' ')
        exp.save_results(output_file)
        print('Done')



