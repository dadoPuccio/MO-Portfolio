import os


def print_parameters(args):
    if args.verbose:
        print()
        print('Parameters')
        print()

        for key in args.__dict__.keys():
            print(key.ljust(args.verbose_interspace), args.__dict__[key])
        print()


def check_args(args):

    assert len(args.single_point_methods) > 0
    for seed in args.seeds:
        assert seed > 0
    if args.max_t_p1 is not None:
        assert args.max_t_p1 > 0
    if args.max_t_p2 is not None:
        assert args.max_t_p2 >= 0
    if args.max_f_evals is not None:
        assert args.max_f_evals > 0
    assert args.verbose_interspace >= 1
    assert args.plot_dpi >= 1

    for s in args.s:
        assert s > 0
    assert args.sparsity_tol > 0

    if args.MOIHT_L is not None:
        assert args.MOIHT_L > 0
    assert args.MOIHT_L_inc_factor > 1
    assert args.MOIHT_theta_tol <= 0

    assert args.MOSPD_xy_diff >= 0
    assert args.MOSPD_max_inner_iter_count > 0
    assert args.MOSPD_max_MOSD_iters > 0
    assert args.MOSPD_tau_0 >= 0
    assert args.MOSPD_max_tau_0_inc_factor > args.MOSPD_tau_inc_factor
    assert args.MOSPD_tau_inc_factor > 1
    assert args.MOSPD_epsilon_0 <= 0
    assert 0 <= args.MOSPD_min_epsilon_0_dec_factor < args.MOSPD_epsilon_dec_factor
    assert 0 < args.MOSPD_epsilon_dec_factor < 1

    assert args.MOSD_IFSD_theta_tol <= 0
    assert 0 <= args.IFSD_qth_quantile <= 1

    assert -1 <= args.gurobi_method <= 5
    assert args.gurobi_feasibility_tol > 0

    assert args.ALS_alpha_0 > 0
    assert 0 < args.ALS_delta < 1
    assert 0 < args.ALS_beta < 1
    assert args.ALS_min_alpha > 0
    
    assert len(args.beta_bounds) == 2


def args_preprocessing(args):
    check_args(args)

    single_point_methods_names = []
    if 'MOIHT' in args.single_point_methods:
        single_point_methods_names.append('MOIHT')
    if 'MOSPD' in args.single_point_methods:
        single_point_methods_names.append('MOSPD')
    if 'MOHyb' in args.single_point_methods:
        single_point_methods_names.append('MOHyb')
    if 'Gurobi' in args.single_point_methods:
        single_point_methods_names.append('Gurobi')
    if 'NG2' in args.single_point_methods:
        single_point_methods_names.append('NG2')
    if 'NS' in args.single_point_methods:
        single_point_methods_names.append('NS')

    refiner = args.refiner

    try:
        beta_lb = float(args.beta_bounds[0])
    except:
        beta_lb = None

    try:
        beta_ub = float(args.beta_bounds[1])
    except:
        beta_ub = None    

    prob_settings = {'prob_type': args.prob_type,
                     'scaling_factors': {
                        'var_mean': args.prob_scaling_factor_var_mean,
                        'esg': args.prob_scaling_factor_esg,
                        'skew': args.prob_scaling_factor_skew
                     },
                     'prob_path': args.prob_path,
                     'objectives': args.objectives,
                     'financial_index': args.financial_index,
                     'scalarization': args.scalarization,
                     'beta_bounds': [beta_lb, beta_ub]}

    general_settings = {'seeds': args.seeds,
                        'max_t_p1': args.max_t_p1,
                        'max_t_p2': args.max_t_p2,
                        'max_f_evals': args.max_f_evals,
                        'verbose': args.verbose,
                        'verbose_interspace': args.verbose_interspace,
                        'plot_pareto_front': args.plot_pareto_front,
                        'plot_pareto_solutions': args.plot_pareto_solutions,
                        'general_export': args.general_export,
                        'export_pareto_solutions': args.export_pareto_solutions,
                        'plot_dpi': args.plot_dpi}

    sparsity_settings = {'s': args.s,
                         'sparsity_tol': args.sparsity_tol}

    MOIHT_settings = {'L': args.MOIHT_L,
                      'L_inc_factor': args.MOIHT_L_inc_factor,
                      'theta_tol': args.MOIHT_theta_tol}
    
    MOSPD_settings = {'xy_diff': args.MOSPD_xy_diff,
                      'max_inner_iter_count': args.MOSPD_max_inner_iter_count,
                      'max_MOSD_iters': args.MOSPD_max_MOSD_iters,
                      'tau_0': args.MOSPD_tau_0,
                      'max_tau_0_inc_factor': args.MOSPD_max_tau_0_inc_factor,
                      'tau_inc_factor': args.MOSPD_tau_inc_factor,
                      'epsilon_0': args.MOSPD_epsilon_0,
                      'min_epsilon_0_dec_factor': args.MOSPD_min_epsilon_0_dec_factor,
                      'epsilon_dec_factor': args.MOSPD_epsilon_dec_factor,
                      'epsilon_simplex_toll': args.MOSPD_epsilon_simplex_toll}
    
    NG2_settings = {'pop_size': args.NG2_pop_size,
                    'crossover_probability': args.NG2_crossover_probability,
                    'crossover_eta': args.NG2_crossover_eta,
                    'mutation_eta': args.NG2_mutation_eta}
    
    NS_settings = {'local_opt': args.NS_local_opt,
                    'shift': args.NS_shift,
                    'crowding_quantile': args.NS_crowding_quantile,
                    'n_opt': args.NS_n_opt,
                    'FMOPG_max_iter': args.NS_FMOPG_max_iter,
                    'theta_for_stationarity': args.NS_theta_for_stationarity,
                    'theta_tol': args.NS_theta_tol,
                    'theta_dec_factor': args.NS_theta_dec_factor}

    MOSD_IFSD_settings = {'theta_tol': args.MOSD_IFSD_theta_tol,
                          'qth_quantile': args.IFSD_qth_quantile}

    algorithms_settings = {'MOIHT': MOIHT_settings,
                           'MOSPD': MOSPD_settings,
                           'NSGA-II-based': NG2_settings,
                           'NSMA-based': NS_settings,
                           'MOSD_IFSD': MOSD_IFSD_settings}

    DDS_settings = {'method': args.gurobi_method,
                    'verbose': args.gurobi_verbose,
                    'feas_tol': args.gurobi_feasibility_tol}

    ALS_settings = {'alpha_0': args.ALS_alpha_0,
                    'delta': args.ALS_delta,
                    'beta': args.ALS_beta,
                    'min_alpha': args.ALS_min_alpha}

    return single_point_methods_names, refiner, prob_settings, general_settings, sparsity_settings, algorithms_settings, DDS_settings, ALS_settings


def args_file_creation(date: str, seed: int, args):
    if args.general_export:
        args_file = open(os.path.join('Execution_Outputs', date, str(seed), 'params.csv'), 'w')
        for key in args.__dict__.keys():
            if type(args.__dict__[key]) == float:
                args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
            else:
                args_file.write('{};{}\n'.format(key, args.__dict__[key]))
        args_file.close()
