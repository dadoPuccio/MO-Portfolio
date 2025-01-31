from algorithms.gradient_based.moiht import MOIHT
from algorithms.gradient_based.mospd import MOSPD
from algorithms.gradient_based.mohyb import MOHyb
from algorithms.genetic.ng2_port import NSGA2Portfolio
from algorithms.memetic.nm_port import NSMAPortfolio
from algorithms.gradient_based.gurobi import Gurobi


class AlgorithmFactory:

    @staticmethod
    def get_algorithm(algorithm_name: str, **kwargs):

        general_settings = kwargs['general_settings']

        algorithms_settings = kwargs['algorithms_settings']
        refiner = kwargs['refiner']
        MOSD_IFSD_settings = algorithms_settings['MOSD_IFSD']

        DDS_settings = kwargs['DDS_settings']
        ALS_settings = kwargs['ALS_settings']

        if algorithm_name == 'MOIHT':
            
            MOIHT_settings = algorithms_settings[algorithm_name]

            algorithm = MOIHT(general_settings['max_t_p1'], general_settings['max_t_p2'], general_settings['max_f_evals'],
                              general_settings['verbose'], general_settings['verbose_interspace'],
                              general_settings['plot_pareto_front'], general_settings['plot_pareto_solutions'], general_settings['plot_dpi'], 
                              MOIHT_settings['L'], MOIHT_settings['L_inc_factor'], MOIHT_settings['theta_tol'],
                              DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                              refiner, MOSD_IFSD_settings,
                              ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])

        elif algorithm_name == 'MOSPD':

            MOSPD_settings = algorithms_settings[algorithm_name]

            algorithm = MOSPD(general_settings['max_t_p1'], general_settings['max_t_p2'], general_settings['max_f_evals'],
                              general_settings['verbose'], general_settings['verbose_interspace'],
                              general_settings['plot_pareto_front'], general_settings['plot_pareto_solutions'], general_settings['plot_dpi'], 
                              MOSPD_settings['xy_diff'], MOSPD_settings['max_inner_iter_count'], MOSPD_settings['max_MOSD_iters'], MOSPD_settings['tau_0'], MOSPD_settings['max_tau_0_inc_factor'], MOSPD_settings['tau_inc_factor'], MOSPD_settings['epsilon_0'], MOSPD_settings['min_epsilon_0_dec_factor'], MOSPD_settings['epsilon_dec_factor'], MOSPD_settings['epsilon_simplex_toll'],
                              DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                              refiner, MOSD_IFSD_settings,
                              ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])

        elif algorithm_name == 'MOHyb':

            MOSPD_settings = algorithms_settings['MOSPD']
            MOIHT_settings = algorithms_settings['MOIHT']

            algorithm = MOHyb(general_settings['max_t_p1'], general_settings['max_t_p2'], general_settings['max_f_evals'],
                              general_settings['verbose'], general_settings['verbose_interspace'],
                              general_settings['plot_pareto_front'], general_settings['plot_pareto_solutions'], general_settings['plot_dpi'], 
                              MOSPD_settings['xy_diff'], MOSPD_settings['max_inner_iter_count'], MOSPD_settings['max_MOSD_iters'], MOSPD_settings['tau_0'], MOSPD_settings['max_tau_0_inc_factor'], MOSPD_settings['tau_inc_factor'], MOSPD_settings['epsilon_0'], MOSPD_settings['min_epsilon_0_dec_factor'], MOSPD_settings['epsilon_dec_factor'], MOSPD_settings['epsilon_simplex_toll'],
                              MOIHT_settings['L'], MOIHT_settings['L_inc_factor'], MOIHT_settings['theta_tol'],
                              DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                              refiner, MOSD_IFSD_settings,
                              ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])
            
        elif algorithm_name == 'NG2':
            NG2_settings = algorithms_settings['NSGA-II-based']

            algorithm = NSGA2Portfolio(general_settings['max_t_p1'], general_settings['max_t_p2'],
                        general_settings['max_f_evals'],
                        general_settings['verbose'],
                        general_settings['verbose_interspace'],
                        general_settings['plot_pareto_front'],
                        general_settings['plot_pareto_solutions'],
                        general_settings['plot_dpi'],
                        NG2_settings['pop_size'],
                        NG2_settings['crossover_probability'],
                        NG2_settings['crossover_eta'],
                        NG2_settings['mutation_eta'],
                        DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                        refiner, MOSD_IFSD_settings,
                        ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])
            
        elif algorithm_name == 'NS':
            NG2_settings = algorithms_settings['NSGA-II-based']
            NS_settings = algorithms_settings['NSMA-based']
            MOIHT_settings = algorithms_settings['MOIHT']

            algorithm = NSMAPortfolio(general_settings['max_t_p1'], general_settings['max_t_p2'],
                        general_settings['max_f_evals'],
                        general_settings['verbose'],
                        general_settings['verbose_interspace'],
                        general_settings['plot_pareto_front'],
                        general_settings['plot_pareto_solutions'],
                        general_settings['plot_dpi'],
                        NG2_settings['pop_size'],
                        NG2_settings['crossover_probability'],
                        NG2_settings['crossover_eta'],
                        NG2_settings['mutation_eta'],
                        NS_settings['local_opt'],
                        NS_settings['shift'],
                        NS_settings['crowding_quantile'],
                        NS_settings['n_opt'],
                        NS_settings['FMOPG_max_iter'],
                        NS_settings['theta_for_stationarity'],
                        NS_settings['theta_tol'],
                        NS_settings['theta_dec_factor'],
                        DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                        MOIHT_settings['L'], MOIHT_settings['L_inc_factor'], 
                        refiner, MOSD_IFSD_settings,
                        ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])

        elif algorithm_name == 'Gurobi':
            DDS_settings = kwargs['DDS_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return Gurobi(general_settings['max_t_p1'], general_settings['max_t_p2'], general_settings['max_f_evals'],
                          general_settings['verbose'], general_settings['verbose_interspace'],
                          general_settings['plot_pareto_front'], general_settings['plot_pareto_solutions'], general_settings['plot_dpi'],
                          sparsity_settings['sparsity_tol'], DDS_settings['method'], DDS_settings['verbose'], DDS_settings['feas_tol'],
                          refiner, MOSD_IFSD_settings, ALS_settings)

        else:
            raise NotImplementedError

        return algorithm
