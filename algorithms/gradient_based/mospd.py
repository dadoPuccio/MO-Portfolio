import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.constrained_ifsd_adaptation import ConstrainedIFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mopg_adaptation import MOPGAdaptation
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.penalty_problem import PenaltyProblem
from problems.extended_problem import ExtendedProblem

from general_utils.pareto_utils import extract_support
from general_utils.projection_utils import compute_projection_to_linear_constrs


class MOSPD(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_t_p1: float, max_t_p2: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 xy_diff: float, max_inner_iter_count: int, max_MOSD_iters: int, tau_0: float, max_tau_0_inc_factor: float, tau_inc_factor: float, epsilon_0: float, min_epsilon_0_dec_factor: float, epsilon_dec_factor: float, epsilon_simplex_toll: float,
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
                                                ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                                name_DDS='MOSPD_DS', name_ALS='MOALS', refiner_instance=refiner_instance)

        self.__xy_diff = xy_diff

        self.__max_inner_iter_count = max_inner_iter_count
        self.__max_MOSD_iters = max_MOSD_iters

        self.__tau_0 = tau_0
        self.__max_tau_0_inc_factor = max_tau_0_inc_factor
        self.__tau_inc_factor = tau_inc_factor

        self.__epsilon_0 = epsilon_0
        self.__min_epsilon_0_dec_factor = min_epsilon_0_dec_factor
        self.__epsilon_dec_factor = epsilon_dec_factor
        self.__epsilon_simplex_toll = epsilon_simplex_toll

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem, lambdas: np.array = None):
        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        penalty_problems = []
        for index_p in range(len(p_list)):
            penalty_problems.append(PenaltyProblem(problem, 
                                                   np.copy(p_list[index_p, :]), 
                                                   self.__tau_0,
                                                   lambdas=lambdas[index_p] if lambdas is not None else None))
        
        first_time_list = [True] * len(p_list)

        epsilon = self.__epsilon_0

        while not self.evaluate_stopping_conditions():

            for index_p in range(len(p_list)):

                self.output_data(f_list)

                if self.evaluate_stopping_conditions():
                    break

                x_p_tmp = np.copy(p_list[index_p, :])
                    
                if not first_time_list[index_p]:
                    if np.linalg.norm(x_p_tmp - penalty_problems[index_p].y) <= self.__xy_diff:
                        continue # skip to next point
                else:
                    first_time_list[index_p] = False

                J_p = penalty_problems[index_p].evaluate_functions_jacobian(x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                for _ in range(self.__max_inner_iter_count):

                    if self.evaluate_stopping_conditions():
                        break

                    d_p, theta_p = self._direction_solver.compute_direction(penalty_problems[index_p], J_p, x_p=x_p_tmp, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                    if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                        break

                    new_point_found = False

                    penalty_f_p_tmp = penalty_problems[index_p].evaluate_functions(x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', 1)

                    for _ in range(self.__max_MOSD_iters):

                        if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                            break

                        new_x_p_tmp, new_penalty_f_p_tmp, alpha, f_eval = self._line_search.search(penalty_problems[index_p], x_p_tmp, penalty_f_p_tmp, d_p, theta_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                        if not self.evaluate_stopping_conditions() and new_x_p_tmp is not None:
                            new_point_found = True

                            x_p_tmp = new_x_p_tmp
                            penalty_f_p_tmp = new_penalty_f_p_tmp

                            J_p = penalty_problems[index_p].evaluate_functions_jacobian(x_p_tmp)
                            self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                            d_p, theta_p = self._direction_solver.compute_direction(penalty_problems[index_p], J_p, x_p=x_p_tmp, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                        else:
                            break

                    if new_point_found:

                        p_list[index_p, :] = np.copy(x_p_tmp)

                        y, _ = self.project(x_p_tmp, problem)

                        penalty_problems[index_p].y = y
                        f_list[index_p, :] = problem.evaluate_functions(penalty_problems[index_p].y)
                        self.add_to_stopping_condition_current_value('max_f_evals', 1)

                        J_p = penalty_problems[index_p].evaluate_functions_jacobian(x_p_tmp)
                        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                    else:
                        break

                penalty_problems[index_p].tau = min(penalty_problems[index_p].tau * self.__tau_inc_factor, self.__tau_0 * self.__max_tau_0_inc_factor)

                self.show_figure(p_list, f_list)

            epsilon = max(epsilon * self.__epsilon_dec_factor, self.__epsilon_0 * self.__min_epsilon_0_dec_factor)

        for index_p in range(len(p_list)):
            proj_y = self.final_project(penalty_problems[index_p].y, problem, 1)
            
            p_list[index_p, :] = proj_y
            if not np.isnan(proj_y).any():
                f_list[index_p, :] = problem.evaluate_functions(proj_y)
                self.add_to_stopping_condition_current_value('max_f_evals', 1)
            else:
                f_list[index_p, :] = np.full_like(f_list[index_p, :], np.nan)
            
        index_not_nan_inf_f = np.logical_and(~np.isnan(np.sum(f_list, axis=1)), ~np.isinf(np.sum(f_list, axis=1)))
        p_list = p_list[index_not_nan_inf_f, :]
        f_list = f_list[index_not_nan_inf_f, :]

        self.output_data(f_list)
        self.close_figure()

        if penalty_problems[0].m == 1:
            res_dict = {'p1': {'p_list': np.copy(p_list), 
                               'f_list': np.copy(f_list), 
                               'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}}

        p_list, f_list, _ = self.call_refiner(p_list, f_list, problem)

        if penalty_problems[0].m == 1:
            res_dict['p2'] = {'p_list': np.copy(p_list), 
                              'f_list': np.copy(f_list), 
                              'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}

        if penalty_problems[0].m == 1:
            return res_dict
        else:
            return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')
    
    @staticmethod
    def project(x: np.array, problem: ExtendedProblem):
        indices = np.argpartition(np.abs(x), problem.n - problem.s)
        x_proj = np.zeros(problem.n)
        x_proj[indices[problem.n - problem.s:]] = x[indices[problem.n - problem.s:]]
        return x_proj, indices

    @staticmethod
    def final_project(x: np.array, problem: ExtendedProblem, max_time: float):

        if problem.betas is not None:
            if problem.beta_lb is not None or problem.beta_ub is not None:
                return compute_projection_to_linear_constrs(problem, x, max_time)

        x_proj, indices = MOSPD.project(x, problem)

        if np.abs(np.sum(x_proj[indices[problem.n - problem.s:]])) < 1e-10:
            print("Projection Warning: Very Small Point")
            return np.full_like(x_proj, np.nan)
        
        else:
            x_proj[indices[problem.n - problem.s:]] /= np.sum(x_proj[indices[problem.n - problem.s:]])
            return x_proj

