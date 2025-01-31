import numpy as np
import time

from nsma.algorithms.memetic.nsma import NSMA

from algorithms.memetic.moiht_for_nsma import MOIHTForNSMA
from algorithms.memetic.mopg_for_nsma import MOPGForNSMA

from algorithms.gradient_based.cc_adaptations.constrained_ifsd_adaptation import ConstrainedIFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mopg_adaptation import MOPGAdaptation

from problems.extended_problem import ExtendedProblem
from general_utils.projection_utils import compute_projection_to_linear_constrs

class NSMAPortfolio(NSMA):

    def __init__(self,
                 max_t_p1: float, max_t_p2: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 pop_size: int, crossover_probability: float, crossover_eta: float, mutation_eta: float,
                 local_opt: str,
                 shift: float, crowding_quantile: float, n_opt: int,
                 FMOPG_max_iter: int, theta_for_stationarity: float, theta_tol: float, theta_dec_factor: float,
                 gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float,
                 L: float, L_inc_factor: float,
                 refiner: str, MOSD_IFSD_settings: dict,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):
        
        if refiner == 'Multi-Start':
            self._refiner_instance = MOPGAdaptation(max_t_p2, max_f_evals,
                                                    verbose, verbose_interspace,
                                                    plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                    MOSD_IFSD_settings["theta_tol"],
                                                    gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                    ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
            
        elif refiner == 'SFSD':
            self._refiner_instance = ConstrainedIFSDAdaptation(max_t_p2, max_f_evals,
                                                               verbose, verbose_interspace,
                                                               plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                               MOSD_IFSD_settings["theta_tol"], MOSD_IFSD_settings["qth_quantile"],
                                                               gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                               ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        else:
            self._refiner_instance = None

        NSMA.__init__(self,
                      None, max_t_p1, max_f_evals,
                      verbose, verbose_interspace,
                      plot_pareto_front, plot_pareto_solutions, plot_dpi,
                      pop_size,
                      crossover_probability, crossover_eta, mutation_eta,
                      shift, crowding_quantile,
                      n_opt, 0,
                      theta_for_stationarity, theta_tol, theta_dec_factor,
                      True, gurobi_method, gurobi_verbose,
                      0, 0, 0, 0)
        
        self._max_time = max_t_p1 * 60 if max_t_p1 is not None else np.inf

        if local_opt == 'MOSD':
            self._local_search_optimizer = MOPGForNSMA(theta_tol, gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                       ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                                       FMOPG_max_iter, max_t_p1, max_f_evals)
        elif local_opt == 'MOIHT':
            self._local_search_optimizer = MOIHTForNSMA(L, L_inc_factor, theta_tol, 
                                                        gurobi_method, gurobi_verbose, gurobi_feasibility_tol, 
                                                        FMOPG_max_iter, max_t_p1, max_f_evals)
        else:
            pass

    @staticmethod
    def objectives_powerset(m: int):
        return [tuple(range(m))]
    
    @staticmethod
    def project(x: np.array, problem: ExtendedProblem, max_time: float):

        if problem.betas is not None:
            if problem.beta_lb is not None or problem.beta_ub is not None:
                return compute_projection_to_linear_constrs(problem, x, max_time) 
            
        indices = np.argpartition(np.abs(x), problem.n - problem.s)

        x_proj = np.zeros(problem.n)
        x_proj[indices[problem.n - problem.s:]] = x[indices[problem.n - problem.s:]]

        if np.abs(np.sum(x_proj[indices[problem.n - problem.s:]])) < 1e-10:
            print("Projection Warning: Very Small Point")
            return np.full_like(x_proj, np.nan)
        
        else:
            x_proj[indices[problem.n - problem.s:]] /= np.sum(x_proj[indices[problem.n - problem.s:]])
            return x_proj
    
    def get_offsprings(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, crowding_list: np.array, problem: ExtendedProblem, surrogate_lb: np.array = None, surrogate_ub: np.array = None):
        off = super().get_offsprings(p_list, f_list, constraint_violations, crowding_list, problem, problem.lb_for_ini, problem.ub_for_ini)
        for idx_p in range(len(off)):
            off[idx_p] = self.project(off[idx_p], problem, self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
        return off
    
    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):
        p_list, f_list, elapsed_time = super().search(p_list, f_list, problem)

        res_dict = {'p1': {'p_list': np.copy(p_list), 
                           'f_list': np.copy(f_list), 
                           'elapsed_time': elapsed_time}}

        self.update_stopping_condition_current_value('max_time', time.time())

        if self._refiner_instance is not None:
            p_list, f_list, _ =  self._refiner_instance.search(p_list, f_list, problem)

        res_dict['p2'] = {'p_list': np.copy(p_list), 
                          'f_list': np.copy(f_list), 
                          'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time') + elapsed_time}
        
        return res_dict