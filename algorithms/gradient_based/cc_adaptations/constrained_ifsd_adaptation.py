import time
import random
import numpy as np
from nsma.general_utils.pareto_utils import pareto_efficient
from nsma.algorithms.genetic.genetic_utils.general_utils import calc_crowding_distance

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from direction_solvers.direction_solver_factory import DirectionDescentFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class ConstrainedIFSDAdaptation(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_time, max_f_evals,
                                                verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                theta_tol,
                                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                                name_DDS='Subspace_Steepest_Descent_DS', name_ALS='Boundconstrained_Front_ALS')

        self.__single_point_line_search = LineSearchFactory.get_line_search('MOALS', ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self.__qth_quantile = qth_quantile

        self.__additional_direction_solver = DirectionDescentFactory.get_direction_calculator('Feasible_Steepest_Descent_DS', gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):

        self.update_stopping_condition_current_value('max_time', time.time())

        efficient_point_idx = pareto_efficient(f_list)
        p_list = p_list[efficient_point_idx, :]
        f_list = f_list[efficient_point_idx, :]

        super_support_sets, idx_point_to_support = self.find_super_support_sets(p_list, f_list, problem)

        self.show_figure(p_list, f_list)

        crowding_quantile = np.inf

        while not self.evaluate_stopping_conditions():

            self.output_data(f_list, crowding_quantile=crowding_quantile)

            previous_p_list = np.copy(p_list)
            previous_f_list = np.copy(f_list)
            previous_visited = np.zeros(len(previous_p_list), dtype=bool)
            previous_idx_point_to_support = np.copy(idx_point_to_support)

            while not self.evaluate_stopping_conditions() and (previous_visited == False).any():

                index_p = self.spread_selection_strategy(previous_f_list, previous_visited)
                previous_visited[index_p] = True

                x_p = np.copy(previous_p_list[index_p, :])
                f_p = np.copy(previous_f_list[index_p, :])
                
                previous_f_list_by_support = previous_f_list[np.where(previous_idx_point_to_support == previous_idx_point_to_support[index_p])[0], :]

                idx_idx_point_to_support_by_support = np.where(idx_point_to_support == previous_idx_point_to_support[index_p])[0]

                p_list_by_support = p_list[idx_idx_point_to_support_by_support, :]
                f_list_by_support = f_list[idx_idx_point_to_support_by_support, :]
                idx_point_to_support_by_support = idx_point_to_support[idx_idx_point_to_support_by_support]

                if self.exists_dominating_point(f_p, f_list_by_support):
                    continue

                crowding_list = calc_crowding_distance(previous_f_list_by_support)
                is_finite_idx = np.isfinite(crowding_list)

                if len(crowding_list[is_finite_idx]) > 0:
                    crowding_quantile = np.quantile(crowding_list[is_finite_idx], self.__qth_quantile)
                else:
                    crowding_quantile = np.inf

                power_set = self.objectives_powerset(problem.m)

                J_p = problem.evaluate_functions_jacobian(x_p)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                common_d_p, common_theta_p = self._direction_solver.compute_direction(problem, J_p, x_p=x_p, subspace_support=super_support_sets[previous_idx_point_to_support[index_p]], time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                if not self.evaluate_stopping_conditions() and common_theta_p < self._theta_tol:

                    new_x_p, new_f_p, _, f_eval = self.__single_point_line_search.search(problem, x_p, f_p, common_d_p, common_theta_p)
                    self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                    if not self.evaluate_stopping_conditions() and new_x_p is not None:

                        if np.sum(np.abs(new_x_p) >= problem.sparsity_tol) > problem.s:
                            print('Warning! Not found a feasible point! Optimization over!')
                            print(new_x_p)
                            continue
                        else:
                            new_x_p[np.abs(new_x_p) < problem.sparsity_tol] = 0.

                        efficient_point_idx = self.fast_non_dominated_filter(f_list_by_support, new_f_p.reshape((1, problem.m)))

                        p_list_by_support = np.concatenate((p_list_by_support[efficient_point_idx, :], new_x_p.reshape((1, problem.n))), axis=0)
                        f_list_by_support = np.concatenate((f_list_by_support[efficient_point_idx, :], new_f_p.reshape((1, problem.m))), axis=0)
                        idx_point_to_support_by_support = np.concatenate((idx_point_to_support_by_support[efficient_point_idx], np.array([previous_idx_point_to_support[index_p]])))

                        x_p = np.copy(new_x_p)

                        previous_f_list_by_support[np.where((previous_f_list_by_support == f_p).all(axis=1))[0][0], :] = np.copy(new_f_p)
                        f_p = np.copy(new_f_p)

                        J_p = problem.evaluate_functions_jacobian(x_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                else:
                    power_set.remove(tuple(range(problem.m)))

                for I_k in power_set:

                    if self.evaluate_stopping_conditions() or self.exists_dominating_point(f_p, f_list_by_support) or crowding_list[np.where((previous_f_list_by_support == f_p).all(axis=1))[0][0]] < crowding_quantile:
                        break

                    partial_d_p, partial_theta_p = self._direction_solver.compute_direction(problem, J_p[I_k, ], x_p=x_p, subspace_support=super_support_sets[previous_idx_point_to_support[index_p]], time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                    if not self.evaluate_stopping_conditions() and partial_theta_p < self._theta_tol:

                        new_x_p, new_f_p, _, f_eval = self._line_search.search(problem, x_p, f_list_by_support, partial_d_p, 0, np.arange(problem.m))
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                        if not self.evaluate_stopping_conditions() and new_x_p is not None:
                            
                            if np.sum(np.abs(new_x_p) >= problem.sparsity_tol) > problem.s:
                                print('Warning! Not found a feasible point! Optimization over!')
                                print(new_x_p)
                                continue
                            else:
                                new_x_p[np.abs(new_x_p) < problem.sparsity_tol] = 0.

                            efficient_point_idx = self.fast_non_dominated_filter(f_list_by_support, new_f_p.reshape((1, problem.m)))

                            p_list_by_support = np.concatenate((p_list_by_support[efficient_point_idx, :], new_x_p.reshape((1, problem.n))), axis=0)
                            f_list_by_support = np.concatenate((f_list_by_support[efficient_point_idx, :], new_f_p.reshape((1, problem.m))), axis=0)
                            idx_point_to_support_by_support = np.concatenate((idx_point_to_support_by_support[efficient_point_idx], np.array([previous_idx_point_to_support[index_p]])))

                p_list = np.concatenate((p_list[np.setdiff1d(np.arange(len(p_list)), idx_idx_point_to_support_by_support), :], p_list_by_support))
                f_list = np.concatenate((f_list[np.setdiff1d(np.arange(len(f_list)), idx_idx_point_to_support_by_support), :], f_list_by_support))
                idx_point_to_support = np.concatenate((idx_point_to_support[np.setdiff1d(np.arange(len(idx_point_to_support)), idx_idx_point_to_support_by_support)], idx_point_to_support_by_support))

                self.show_figure(p_list, f_list)

        self.close_figure()
        self.output_data(f_list, crowding_quantile=crowding_quantile)

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')

    def find_super_support_sets(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):
        random.seed(16007)

        super_support_sets = []
        idx_point_to_support = None

        for index_p in range(len(p_list)):

            if self.evaluate_stopping_conditions():
                break

            if np.sum(np.abs(p_list[index_p, :]) >= problem.sparsity_tol) < problem.s:

                J_p = problem.evaluate_functions_jacobian(p_list[index_p, :])
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                z_p, theta_p = self.__additional_direction_solver.compute_direction(problem, J_p, x_p=p_list[index_p, :], time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                if np.sum(np.abs(z_p) >= problem.sparsity_tol) > problem.s:
                    print('Warning! Not found a feasible point! Optimization over!')
                    print(z_p)
                else:
                    z_p[np.abs(z_p) < problem.sparsity_tol] = 0.

                    if not self.evaluate_stopping_conditions() and theta_p < self._theta_tol:
                        new_x_p, new_f_p, _, f_eval = self.__single_point_line_search.search(problem, p_list[index_p, :], f_list[index_p, :], z_p - p_list[index_p, :], theta_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                        if not self.evaluate_stopping_conditions() and new_x_p is not None:
                            p_list[index_p, :] = new_x_p
                            f_list[index_p, :] = new_f_p

            support_p = list(np.where(np.abs(p_list[index_p, :]) >= problem.sparsity_tol)[0])

            if support_p not in super_support_sets:
                super_support_sets.append(support_p)

                idx_point_to_support = np.concatenate((idx_point_to_support, np.array([len(super_support_sets) - 1]))) if idx_point_to_support is not None else np.array([len(super_support_sets) - 1])
            else:
                idx_point_to_support = np.concatenate((idx_point_to_support, np.array([super_support_sets.index(support_p)]))) if idx_point_to_support is not None else np.array([super_support_sets.index(support_p)])

        for idx_support_1 in range(len(super_support_sets) - 1, -1, -1):

            if len(super_support_sets[idx_support_1]) < problem.s:

                for idx_support_2 in range(len(super_support_sets)):

                    if idx_support_2 != idx_support_1:

                        if set(super_support_sets[idx_support_1]) <= set(super_support_sets[idx_support_2]) and len(super_support_sets[idx_support_2]) == problem.s:

                            del super_support_sets[idx_support_1]

                            idx_point_to_support_1 = np.where(idx_point_to_support == idx_support_1)[0]

                            idx_point_to_support[np.where(idx_point_to_support > idx_support_1)[0]] -= 1

                            idx_point_to_support[idx_point_to_support_1] = idx_support_2 - (1 if idx_support_2 > idx_support_1 else 0)

                            break

                    if idx_support_2 == len(super_support_sets) - 1:

                        while len(super_support_sets[idx_support_1]) < problem.s:

                            random_number = random.randint(0, problem.n - 1)
                            while random_number in super_support_sets[idx_support_1]:
                                random_number = random.randint(0, problem.n - 1)

                            super_support_sets[idx_support_1].append(random_number)

        return super_support_sets, idx_point_to_support

    @staticmethod
    def spread_selection_strategy(f_list: np.array, visited: np.array):

        fronts = ConstrainedIFSDAdaptation.fast_non_dominated_sort(f_list)

        for front in fronts:
            for idx_point in front:
                if not visited[idx_point]:
                    return idx_point

    @staticmethod  
    def calc_domination_matrix(f_list: np.array):
        n, m = f_list.shape[0], f_list.shape[0]

        L = np.repeat(f_list, m, axis=0)
        R = np.tile(f_list, (n, 1))

        smaller = np.reshape(np.any(L < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 + np.logical_and(larger, np.logical_not(smaller)) * -1

        return M

    @staticmethod
    def fast_non_dominated_sort(f_list: np.array):

        M = ConstrainedIFSDAdaptation.calc_domination_matrix(f_list)

        n = M.shape[0]

        fronts = []
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)
        is_dominating = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)

        current_front = []

        for i in range(n):

            for j in range(i + 1, n):

                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        fronts.append(current_front)

        while n_ranked < n:

            next_front = []

            for i in current_front:

                for j in is_dominating[i]:

                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts