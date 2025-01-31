import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.mopg_adaptation import MOPGAdaptation
from problems.extended_problem import ExtendedProblem

from nsma.algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG


class MOPGForNSMA(MOPGAdaptation, FMOPG):

    def __init__(self,
                 theta_tol: float,
                 gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float,
                 max_iter: float = None,
                 max_time: float = None,
                 max_f_evals: int = None):

        MOPGAdaptation.__init__(self,
                                max_time, max_f_evals,
                                False, 0,
                                False, False, 0,
                                theta_tol,
                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        
        self.update_stopping_condition_reference_value('max_iter', max_iter)

        self.__theta_array = np.array([-np.inf], dtype=float)
        self.add_stopping_condition('theta_tolerance', theta_tol, self.__theta_array[0], equal_required=True)

        self.__alpha_array = np.array([1], dtype=float)
        self.add_stopping_condition('min_alpha', 0, self.__alpha_array[0], smaller_value_required=True, equal_required=True)

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem, index_initial_point: int = None, I: tuple = None):

        n, m = p_list.shape[1], f_list.shape[1]
        
        p_list_tmp = p_list[index_initial_point, :].reshape(1, n)
        f_list_tmp = f_list[index_initial_point, :].reshape(1, m)
        theta_list_tmp = np.array([self.__theta_array[0]])

        optimization_success = False

        J = problem.evaluate_functions_jacobian(p_list_tmp[0, :])
        self.add_to_stopping_condition_current_value('max_f_evals', n)

        while not self.evaluate_stopping_conditions():

            n_iteration = self.get_stopping_condition_current_value('max_iter')

            d_p, theta_p = self._direction_solver.compute_direction(problem, J, x_p=p_list_tmp[n_iteration, :], subspace_support=np.where(np.abs(p_list_tmp[n_iteration, :]) >= problem.sparsity_tol)[0], time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
            self.__theta_array[n_iteration] = theta_p
            self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

            if not self.evaluate_stopping_conditions() and theta_p < self._theta_tol:

                new_x_p_tmp, new_f_p_tmp, alpha, f_eval = self._line_search.search(problem, p_list_tmp[n_iteration, :], f_list_tmp[n_iteration, :], d_p, theta_p)
                self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                self.__alpha_array[n_iteration] = alpha
                self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration])

                if not self.evaluate_stopping_conditions() and new_x_p_tmp is not None:

                    if np.sum(np.abs(new_x_p_tmp) >= problem.sparsity_tol) > problem.s:
                        print('Warning! Not found a feasible point! Optimization over!')
                        print(new_x_p_tmp)
                        break
                    else:
                        new_x_p_tmp[np.abs(new_x_p_tmp) < problem.sparsity_tol] = 0.

                    optimization_success = True

                    p_list_tmp = np.concatenate((p_list_tmp, new_x_p_tmp.reshape((1, n))), axis=0)
                    f_list_tmp = np.concatenate((f_list_tmp, new_f_p_tmp.reshape((1, m))), axis=0)

                    self.__theta_array = np.concatenate((self.__theta_array, np.array([-np.inf])), axis=0)
                    self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration + 1])

                    self.__alpha_array = np.concatenate((self.__alpha_array, np.array([1])), axis=0)
                    self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration + 1])

                    J = problem.evaluate_functions_jacobian(new_x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', n)

            self.add_to_stopping_condition_current_value('max_iter', 1)

        if optimization_success:
            p_list = np.concatenate((p_list, p_list_tmp[-1, :].reshape(1, n)), axis=0)
            f_list = np.concatenate((f_list, f_list_tmp[-1, :].reshape(1, m)), axis=0)
            theta_list_tmp = np.concatenate((theta_list_tmp, np.array([self.__theta_array[-1]])))

        return p_list, f_list, theta_list_tmp
    
    def reset_stopping_conditions_current_values(self, theta_tol: float):
        """
        Reset the stopping conditions current values
        :param theta_tol: the new current value for the stopping condition 'theta_tolerance'

        Notes:  The current values of the stopping conditions 'max_time' and 'max_f_evals' are changed by the memetic algorithm that employs FMOPG.
        """

        self.update_stopping_condition_current_value('max_iter', 0)

        self._theta_tol = theta_tol
        self.__theta_array = np.array([-np.inf], dtype=float)
        self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[0])
        self.update_stopping_condition_reference_value('theta_tolerance', theta_tol)

        self.__alpha_array = np.array([1], dtype=float)
        self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[0])
