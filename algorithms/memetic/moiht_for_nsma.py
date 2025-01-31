import numpy as np
import time

from nsma.algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG

from algorithms.gradient_based.moiht import MOIHT
from problems.extended_problem import ExtendedProblem


class MOIHTForNSMA(MOIHT, FMOPG):

    def __init__(self,
                 L: float,
                 L_inc_factor: float,
                 theta_tol: float,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 gurobi_feasibility_tol: float,
                 max_iter: float = None,
                 max_time: float = None,
                 max_f_evals: int = None):

        MOIHT.__init__(self, 
                       max_time, 0, max_f_evals,
                        False, 0,
                        False, False, 0,
                        L, L_inc_factor, theta_tol,
                        gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                        None, None, 0, 0, 0, 0)
        
        self.update_stopping_condition_reference_value('max_iter', max_iter)

        self.__theta_array = np.array([-np.inf], dtype=float)
        self.add_stopping_condition('theta_tolerance', theta_tol, self.__theta_array[0], equal_required=True)

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem, index_initial_point: int = None, I: tuple = None):

        _, n = p_list.shape
        m = f_list.shape[1]

        p_list_tmp = p_list[index_initial_point, :].reshape(1, n)
        f_list_tmp = f_list[index_initial_point, :].reshape(1, m)
        theta_list_tmp = np.array([self.__theta_array[0]])

        optimization_success = False

        J = problem.evaluate_functions_jacobian(p_list_tmp[0, :])
        self.add_to_stopping_condition_current_value('max_f_evals', n)

        while not self.evaluate_stopping_conditions():

            n_iteration = self.get_stopping_condition_current_value('max_iter')

            new_x_p_tmp, theta_p = self._direction_solver.compute_direction(problem, J, x_p=p_list_tmp[n_iteration, :], L=(self._L if self._L is not None else problem.L) * self._L_inc_factor, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
            self.__theta_array[n_iteration] = theta_p
            self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

            if theta_p < self._theta_tol:

                if np.sum(np.abs(new_x_p_tmp) >= problem.sparsity_tol) > problem.s:
                    print('Warning! Found a not feasible point! Optimization over!')
                    print(new_x_p_tmp)
                    break
                else:
                    new_x_p_tmp[np.abs(new_x_p_tmp) < problem.sparsity_tol] = 0.

                optimization_success = True

                p_list_tmp = np.concatenate((p_list_tmp, new_x_p_tmp.reshape((1, n))), axis=0)

                new_f_p_tmp = problem.evaluate_functions(new_x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', 1)
                f_list_tmp = np.concatenate((f_list_tmp, new_f_p_tmp.reshape((1, m))), axis=0)

                self.__theta_array = np.concatenate((self.__theta_array, np.array([-np.inf])), axis=0)
                self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration + 1])

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