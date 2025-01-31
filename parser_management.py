import argparse
import sys


def get_args():

    parser = argparse.ArgumentParser(description='algorithms for Multi-Objective Optimization')

    parser.add_argument('--single_point_methods', type=str, help='Single Point Method', nargs='+', choices=['MOSPD', 'MOHyb', 'Gurobi', 'NG2', 'NS'])

    parser.add_argument('--refiner', type=str, help='Refiner to use with the Single Methods', nargs='+', default=['SFSD'])

    parser.add_argument('--prob_type', type=str, help='problems Type', default='Portfolio')

    parser.add_argument('--prob_scaling_factor_var_mean', type=float, help='Scaling factor to be used in the Porfolio Problem', default=1e3)

    parser.add_argument('--prob_scaling_factor_esg', type=float, help='Scaling factor to be used in the Porfolio Problem', default=0.01)

    parser.add_argument('--prob_scaling_factor_skew', type=float, help='Scaling factor to be used in the Porfolio Problem', default=0.1)

    parser.add_argument('--prob_path', help='problems path (folder or file)', type=str, nargs='+')

    parser.add_argument('--financial_index', help='financial index used for parameters estimation in Portfolio problem', type=str, choices=['EuroStoxx', 'S&P'])
    
    parser.add_argument('--seeds', help='Execution Seeds', nargs='+', type=int, default=[16007])

    parser.add_argument('--max_t_p1', help='Maximum number of elapsed minutes per problem in the first phase', default=None, type=float)

    parser.add_argument('--max_t_p2', help='Maximum number of elapsed minutes per problem in the second phase', default=None, type=float)

    parser.add_argument('--max_f_evals', help='Maximum number of function evaluations for both the single-point method and the refiner', default=None, type=int)

    parser.add_argument('--verbose', help='Verbose during the iterations', action='store_true', default=False)

    parser.add_argument('--verbose_interspace', help='Used interspace in the verbose (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('--plot_pareto_front', help='Plot Pareto front', action='store_true', default=False)

    parser.add_argument('--plot_pareto_solutions', help='Plot Pareto solutions (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('--general_export', help='Export fronts (including plots), execution times and arguments files', action='store_true', default=False)

    parser.add_argument('--export_pareto_solutions', help='Export pareto solutions, including the plots if n in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('--plot_dpi', help='DPI of the saved plots (Requirements: general_export activated)', default=100, type=int)

    parser.add_argument('--objectives', type=str, nargs='+', help='Objectives', choices=['Variance', 'Mean', 'SR', 'ESG', 'Skew'])

    parser.add_argument('--beta_bounds', type=str, nargs='+', help='Bounds for beta', default=['None', 'None'])

    parser.add_argument('--scalarization', help='Scalarization Mode for MOSPD', action='store_true', default=False)

    ####################################################
    ### ONLY FOR Sparsity ###
    ####################################################

    parser.add_argument('--s', help='Sparsity parameter -- Upper Bound for the Solution Cardinality', type=int, nargs='+')

    parser.add_argument('--sparsity_tol', help='Sparsity parameter -- Tolerance on sparsity', type=float, default=1e-7)

    ####################################################
    ### ONLY FOR MOIHT ###
    ####################################################

    parser.add_argument('--MOIHT_L', help='MOIHT parameter -- L-stationarity parameter (if not indicated, it is calculated depending on the problem)', type=float, default=None)

    parser.add_argument('--MOIHT_L_inc_factor', help='MOIHT parameter -- L incremental factor', type=float, default=1.1)

    parser.add_argument('--MOIHT_theta_tol', help='MOIHT parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)

    ####################################################
    ### ONLY FOR MOSPD ###
    ####################################################

    parser.add_argument('--MOSPD_xy_diff', help='MOSPD parameter -- Difference between current x and y', default=1e-3, type=float)

    parser.add_argument('--MOSPD_max_inner_iter_count', help='MOSPD parameter -- Maximum number of iterations in the inner loop', default=10, type=int)

    parser.add_argument('--MOSPD_max_MOSD_iters', help='MOSPD parameter -- Maximum number of MOSD iterations', default=10, type=int)

    parser.add_argument('--MOSPD_tau_0', help='MOSPD parameter -- Initial value for tau', default=0.01, type=float)

    parser.add_argument('--MOSPD_max_tau_0_inc_factor', help='MOSPD parameter -- Maximum Increment factor for initial value of tau', default=1e8, type=float)

    parser.add_argument('--MOSPD_tau_inc_factor', help='MOSPD parameter -- Increment factor for tau', default=2, type=float)

    parser.add_argument('--MOSPD_epsilon_0', help='MOSPD parameter -- Initial value for epsilon', default=-1e-3, type=float)

    parser.add_argument('--MOSPD_min_epsilon_0_dec_factor', help='MOSPD parameter -- Minimum Decrement factor for initial value of epsilon', default=1e-6, type=float)

    parser.add_argument('--MOSPD_epsilon_dec_factor', help='MOSPD parameter -- Decrement factor for epsilon', default=0.9, type=float)

    parser.add_argument('--MOSPD_epsilon_simplex_toll', help='MOSPD parameter -- Epsilon value for simplex constraint', default=1e-3, type=float)

    ####################################################
    ### ONLY FOR NSGA-II-based ###
    ####################################################

    parser.add_argument('-NG2_ps', '--NG2_pop_size', help='NSGA-II-based parameter -- Population size', default=100, type=int)

    parser.add_argument('-NG2_cp', '--NG2_crossover_probability', help='NSGA-II-based parameter -- Crossover probability', default=0.9, type=float)

    parser.add_argument('-NG2_ce', '--NG2_crossover_eta', help='NSGA-II-based parameter -- Crossover eta', default=20, type=float)

    parser.add_argument('-NG2_me', '--NG2_mutation_eta', help='NSGA-II-based parameter -- Mutation eta', default=20, type=float)

    ####################################################
    ### ONLY FOR NSMA-based ###
    ####################################################

    parser.add_argument('--NS_local_opt', help='NSMA parameter -- Local optimizer name', type=str, choices=['MOSD', 'MOIHT'])

    parser.add_argument('--NS_shift', help='NSMA parameter -- Shift parameter', default=10, type=float)

    parser.add_argument('--NS_crowding_quantile', help='NSMA parameter -- Crowding distance quantile', default=0.95, type=float)

    parser.add_argument('--NS_n_opt', help='NSMA parameter -- Number of iterations before doing optimization', default=10, type=int)

    parser.add_argument('--NS_FMOPG_max_iter', help='NSMA parameter -- Number of maximum iterations for FMOPG', default=5, type=int)

    parser.add_argument('--NS_theta_for_stationarity', help='NSMA parameter -- Theta for Pareto stationarity', default=-1.0e-10, type=float)

    parser.add_argument('--NS_theta_tol', help='NSMA parameter -- Theta tolerance', default=-1.0e-1, type=float)

    parser.add_argument('--NS_theta_dec_factor', help='NSMA parameter -- Theta decreasing factor', default=10 ** (-1 / 2), type=float)

    ####################################################
    ### ONLY FOR MOSD/IFSD ###
    ####################################################

    parser.add_argument('--MOSD_IFSD_theta_tol', help='MOSD/IFSD parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)
    
    parser.add_argument('--IFSD_qth_quantile', help='IFSD parameter -- q-th quantile', default=0.95, type=float)

    ####################################################
    ### ONLY FOR Gurobi ###
    ####################################################

    parser.add_argument('--gurobi_method', help='Gurobi parameter -- Used method', default=1, type=int)

    parser.add_argument('--gurobi_verbose', help='Gurobi parameter -- Verbose during the Gurobi iterations', action='store_true', default=False)

    parser.add_argument('--gurobi_feasibility_tol', help='Gurobi parameter -- Feasibility tolerance', default=1e-7, type=float)

    ####################################################
    ### ONLY FOR ArmijoTypeLineSearch ###
    ####################################################

    parser.add_argument('--ALS_alpha_0', help='ALS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('--ALS_delta', help='ALS parameter -- Coefficient for step size contraction', default=0.5, type=float)

    parser.add_argument('--ALS_beta', help='ALS parameter -- Beta', default=1.0e-4, type=float)

    parser.add_argument('--ALS_min_alpha', help='ALS parameter -- Min alpha', default=1.0e-14, type=float)

    return parser.parse_args(sys.argv[1:])

