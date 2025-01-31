import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.constrained_ifsd_adaptation import ConstrainedIFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mopg_adaptation import MOPGAdaptation
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.extended_problem import ExtendedProblem

from general_utils.pareto_utils import extract_support

class MOIHT(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_t_p1: float, max_t_p2: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
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
                                                theta_tol,
                                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                0., 0., 0., 0.,
                                                name_DDS='MOIHT_DS', refiner_instance=refiner_instance)

        self._L = L
        self._L_inc_factor = L_inc_factor

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):
        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        theta_p_list = -np.inf * np.ones(len(p_list))

        while not self.evaluate_stopping_conditions() and (theta_p_list < self._theta_tol).any():

            for index_p in range(len(p_list)):

                self.output_data(f_list)

                if self.evaluate_stopping_conditions():
                    break

                x_p_tmp = np.copy(p_list[index_p, :])
                f_p_tmp = np.copy(f_list[index_p, :])

                if theta_p_list[index_p] >= self._theta_tol:
                    continue # skip to next point

                J_p = problem.evaluate_functions_jacobian(x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                new_x_p_tmp, theta_p = self._direction_solver.compute_direction(problem, J_p, x_p=x_p_tmp, L=(self._L if self._L is not None else problem.L) * self._L_inc_factor, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
                theta_p_list[index_p] = theta_p

                if not self.evaluate_stopping_conditions() and theta_p < self._theta_tol:

                    new_f_p_tmp = problem.evaluate_functions(new_x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', 1)
                    
                    if np.sum(np.abs(new_x_p_tmp) >= problem.sparsity_tol) > problem.s:
                        print('Warning! Found a not feasible point! Optimization over!')
                        print(new_x_p_tmp)
                        theta_p_list[index_p] = 0
                        continue
                    else:
                        new_x_p_tmp[np.abs(new_x_p_tmp) < problem.sparsity_tol] = 0.
                        p_list[index_p, :] = new_x_p_tmp
                        f_list[index_p, :] = new_f_p_tmp

                self.show_figure(p_list, f_list)

        self.output_data(f_list)
        self.close_figure()

        p_list, f_list, _ = self.call_refiner(p_list, f_list, problem)

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')
