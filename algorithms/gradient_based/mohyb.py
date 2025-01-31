import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.constrained_ifsd_adaptation import ConstrainedIFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mopg_adaptation import MOPGAdaptation
from algorithms.gradient_based.moiht import MOIHT
from algorithms.gradient_based.mospd import MOSPD
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.extended_problem import ExtendedProblem


class MOHyb(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_t_p1: float, max_t_p2: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 xy_diff: float, max_inner_iter_count: int, max_MOSD_iters: int, tau_0: float, max_tau_0_inc_factor: float, tau_inc_factor: float, epsilon_0: float, min_epsilon_0_dec_factor: float, epsilon_dec_factor: float, epsilon_simplex_toll: float,
                 L: float, L_inc_factor: float, theta_tol: float,
                 gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float,
                 refiner: str, MOSD_IFSD_settings: dict,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):
        
        if refiner == 'Multi-Start':
            refiner_instance = MOPGAdaptation(max_t_p2, max_f_evals,
                                              verbose, verbose_interspace,
                                              plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                              MOSD_IFSD_settings["theta_tol"],
                                              gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                              ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        elif refiner == 'SFSD':
            refiner_instance = ConstrainedIFSDAdaptation(max_t_p2, max_f_evals,
                                                         verbose, verbose_interspace,
                                                         plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                         MOSD_IFSD_settings["theta_tol"], MOSD_IFSD_settings["qth_quantile"],
                                                         gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                         ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        else:
            refiner_instance = None

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_t_p1, max_f_evals,
                                                verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                0.,
                                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                0., 0., 0., 0.,
                                                refiner_instance=refiner_instance)

        self.__mospd_instance = MOSPD(np.inf, np.inf, np.inf,
                                      False, verbose_interspace,
                                      False, False, plot_dpi,
                                      xy_diff, max_inner_iter_count, max_MOSD_iters, tau_0, max_tau_0_inc_factor, tau_inc_factor, epsilon_0, min_epsilon_0_dec_factor, epsilon_dec_factor, epsilon_simplex_toll,
                                      gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                      None, MOSD_IFSD_settings, 
                                      ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        
        self.__moiht_instance = MOIHT(np.inf, np.inf, np.inf,
                                      False, verbose_interspace,
                                      False, False, plot_dpi,
                                      L, L_inc_factor, theta_tol,
                                      gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                      None, MOSD_IFSD_settings,
                                      ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        
        self.__max_f_evals = max_f_evals if max_f_evals is not None else np.inf

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):
        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        self.output_data(f_list)

        if not self.evaluate_stopping_conditions():

            self.__moiht_instance.update_stopping_condition_current_value('max_f_evals', 0)
            self.__moiht_instance.update_stopping_condition_reference_value('max_f_evals', np.inf if self.__max_f_evals == np.inf else int((self.__max_f_evals - self.get_stopping_condition_current_value('max_f_evals'))/2))

            self.__moiht_instance.update_stopping_condition_reference_value('max_time', self._max_time / 2 - time.time() + self.get_stopping_condition_current_value('max_time')) # half of the time can be used

            p_list_moiht, f_list_moiht, _ = self.__moiht_instance.search(np.copy(p_list), np.copy(f_list), problem)
        
            self.update_stopping_condition_current_value('max_f_evals', self.get_stopping_condition_current_value('max_f_evals') + self.__moiht_instance.get_stopping_condition_current_value('max_f_evals'))
            
            print('N points found with MOIHT:', len(p_list_moiht))

            self.show_figure(p_list_moiht, f_list_moiht)
            self.output_data(f_list_moiht)

        else:
            p_list_moiht = None
            f_list_moiht = None

        if not self.evaluate_stopping_conditions():

            self.__mospd_instance.update_stopping_condition_current_value('max_f_evals', 0)
            self.__mospd_instance.update_stopping_condition_reference_value('max_f_evals', self.__max_f_evals - self.get_stopping_condition_current_value('max_f_evals')) 

            self.__mospd_instance.update_stopping_condition_reference_value('max_time', self._max_time - time.time() + self.get_stopping_condition_current_value('max_time')) # the remaining time is used here

            p_list_mospd, f_list_mospd, _ = self.__mospd_instance.search(np.copy(p_list), np.copy(f_list), problem)

            self.update_stopping_condition_current_value('max_f_evals', self.get_stopping_condition_current_value('max_f_evals') + self.__mospd_instance.get_stopping_condition_current_value('max_f_evals'))

            print('N points found with MOSPD:', len(p_list_mospd))

        else:
            p_list_mospd = None
            f_list_mospd = None

        if p_list_moiht is not None and p_list_mospd is not None:
            p_list = np.concatenate((p_list_moiht, p_list_mospd))
            f_list = np.concatenate((f_list_moiht, f_list_mospd))
        elif p_list_mospd is None:
            p_list = p_list_moiht
            f_list = f_list_moiht
        else:
            raise RuntimeError('MOIHT and MOSPD presented an unexpected behavior')

        self.show_figure(p_list, f_list)
        self.output_data(f_list)
        self.close_figure()

        res_dict = {'p1': {'p_list': np.copy(p_list), 
                           'f_list': np.copy(f_list), 
                           'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}}

        p_list, f_list, _ = self.call_refiner(p_list, f_list, problem)

        res_dict['p2'] = {'p_list': np.copy(p_list), 
                          'f_list': np.copy(f_list), 
                          'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}

        return res_dict
