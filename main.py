import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from datetime import datetime
import tensorflow as tf
import glob

from nsma.algorithms.algorithm_utils.graphical_plot import GraphicalPlot

from algorithms.algorithm_factory import AlgorithmFactory
from problems.problem_factory import ProblemFactory
from general_utils.args_utils import print_parameters, args_preprocessing, args_file_creation
from general_utils.management_utils import folder_initialization, execution_time_file_initialization, write_in_execution_time_file, write_results_in_csv_file, save_plots
from general_utils.pareto_utils import points_initialization, points_postprocessing, points_initialization_gurobi, points_initialization_scalarization
from general_utils.progress_bar import ProgressBarWrapper
from parser_management import get_args


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    args = get_args()

    print_parameters(args)
    single_point_methods_names, refiners, prob_settings, general_settings, sparsity_settings, algorithms_settings, DDS_settings, ALS_settings = args_preprocessing(args)

    if prob_settings['prob_type'] == 'Portfolio':
        prob_paths = prob_settings['prob_path']

    else:
        raise NotImplementedError

    print('N° Single Point Methods: ', len(single_point_methods_names))
    print('N° Refiners: ', len(refiners))
    print('N° Problems: ', len(prob_paths))
    print('N° Upper Bounds for Cardinality Constraint: ', len(sparsity_settings['s']))
    print('N° Seeds: ', len(general_settings['seeds']))
    print()

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if general_settings['verbose']:
        progress_bar = ProgressBarWrapper(len(single_point_methods_names) * len(prob_paths) * len(sparsity_settings['s']) * len(general_settings['seeds']))
        progress_bar.show_bar()

    for seed in general_settings['seeds']:
        
        print()
        print('Seed: ', str(seed))

        if general_settings['general_export']:
            folder_initialization(date, seed, single_point_methods_names, ['None'] + [r+('_Null' if general_settings['max_t_p2'] == 0 else '') for r in refiners], prob_settings['scalarization'], algorithms_settings['NSMA-based']['local_opt'])
            args_file_creation(date, seed, args)
            execution_time_file_initialization(date, seed, single_point_methods_names, ['None'] + [r+('_Null' if general_settings['max_t_p2'] == 0 else '') for r in refiners], prob_settings['scalarization'], algorithms_settings['NSMA-based']['local_opt'])

        for prob_path in prob_paths:
            for s in sparsity_settings['s']:
                for idx_single_point_method, single_point_method_name in enumerate(single_point_methods_names):
                    for refiner in refiners:

                        session = tf.compat.v1.Session()
                        with session.as_default():

                            problem_instance = ProblemFactory.get_problem(prob_settings['prob_type'],
                                                                          prob_path=prob_path,
                                                                          scaling_factors=prob_settings['scaling_factors'],
                                                                          s=s,
                                                                          sparsity_tol=sparsity_settings['sparsity_tol'],
                                                                          objectives=prob_settings['objectives'],
                                                                          financial_index=prob_settings['financial_index'],
                                                                          scalarization=prob_settings['scalarization'],
                                                                          beta_bounds=prob_settings['beta_bounds'])
                            
                            if not idx_single_point_method:
                                print()
                                print('Problem Type: ', problem_instance.family_name())
                                print('Problem Path: ', problem_instance.name())
                                print('Problem Dimensionality: ', problem_instance.n)
                                print('Upper Bound for Cardinality Constraint: ', s)

                            algorithm = AlgorithmFactory.get_algorithm(single_point_method_name,
                                                                    general_settings=general_settings,
                                                                    algorithms_settings=algorithms_settings,
                                                                    refiner=refiner,
                                                                    DDS_settings=DDS_settings,
                                                                    ALS_settings=ALS_settings,
                                                                    sparsity_settings=sparsity_settings)

                            print()
                            print('Single Point Method: ', single_point_method_name)
                            print('Refiner: ', refiner)
                            if single_point_method_name == 'NS':
                                print('Local Optimizer: ', algorithms_settings['NSMA-based']['local_opt'])

                            np.random.seed(seed=seed)
                            if single_point_method_name == 'Gurobi':
                                initial_p_list, lambdas, initial_f_list, n_initial_points = points_initialization_gurobi(problem_instance, seed)
                            elif problem_instance.scalarization:
                                initial_p_list, lambdas, initial_f_list, n_initial_points = points_initialization_scalarization(problem_instance, seed)
                            else:
                                initial_p_list, initial_f_list, n_initial_points = points_initialization(problem_instance, seed)
                                
                            problem_instance.evaluate_functions(initial_p_list[0, :])
                            problem_instance.evaluate_functions_jacobian(initial_p_list[0, :])       

                            if single_point_method_name == 'Gurobi':
                                res_dict = algorithm.search(initial_p_list, initial_f_list, problem_instance, lambdas)
                            else:
                                if problem_instance.scalarization:
                                    res_dict = algorithm.search(initial_p_list, initial_f_list, problem_instance, lambdas)
                                else:
                                    res_dict = algorithm.search(initial_p_list, initial_f_list, problem_instance)

                            assert type(res_dict) == dict

                            res_names = {'p1': 'None', 'p2': refiner + ('_Null' if general_settings['max_t_p2'] == 0 else '')}

                            for phase in res_dict.keys():
                                p_list = np.copy(res_dict[phase]['p_list'])
                                f_list = np.copy(res_dict[phase]['f_list'])
                                elapsed_time = np.copy(res_dict[phase]['elapsed_time'])
                            
                                final_p_list, final_f_list = points_postprocessing(p_list, f_list, problem_instance)

                                if general_settings['plot_pareto_front']:
                                    graphical_plot = GraphicalPlot(general_settings['plot_pareto_solutions'], general_settings['plot_dpi'])
                                    graphical_plot.show_figure(final_p_list, final_f_list, hold_still=True)
                                    graphical_plot.close_figure()

                                if general_settings['general_export']:
                                    if problem_instance.scalarization:
                                        postfix = '-Scal'
                                    elif single_point_method_name == 'NS':
                                        postfix = '-' + str(algorithms_settings['NSMA-based']['local_opt'])
                                    else:
                                        postfix = ''
                                    write_in_execution_time_file(date, seed, single_point_method_name + postfix, res_names[phase], problem_instance, elapsed_time)
                                    write_results_in_csv_file(date, seed, single_point_method_name + postfix, res_names[phase], problem_instance, final_p_list, final_f_list, export_pareto_solutions=general_settings['export_pareto_solutions'])
                                    if 1 < final_f_list.shape[1] <= 3:
                                        save_plots(date, seed, single_point_method_name + postfix, res_names[phase], problem_instance, final_p_list, final_f_list, general_settings['export_pareto_solutions'], general_settings['plot_dpi'])

                            if general_settings['verbose']:
                                progress_bar.increment_current_value()
                                progress_bar.show_bar()

                            tf.compat.v1.reset_default_graph()
                            session.close()
