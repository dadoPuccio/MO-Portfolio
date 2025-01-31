import time
import numpy as np
from gurobipy import Model, GRB, quicksum

from algorithms.gradient_based.cc_adaptations.constrained_ifsd_adaptation import ConstrainedIFSDAdaptation
from nsma.algorithms.algorithm import Algorithm

class Gurobi(Algorithm):  
    
    def __init__(self, max_t_p1, max_t_p2, max_f_evals, verbose, verbose_interspace, 
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 sparse_tol, gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                 refiner, MOSD_IFSD_settings, ALS_settings):
        
        Algorithm.__init__(self, np.inf, max_t_p1, max_f_evals, verbose, verbose_interspace,
                           plot_pareto_front, plot_pareto_solutions, plot_dpi)

        self._max_time = max_t_p1 * 60 if max_t_p1 is not None else np.inf

        self.__sparse_tol = sparse_tol

        self._gurobi_method = gurobi_method if gurobi_method is not None else -1
        self._gurobi_verbose = gurobi_verbose
        self._gurobi_feas_tol = gurobi_feasibility_tol

        if refiner == 'SFSD':
            self._refiner_instance = ConstrainedIFSDAdaptation(max_t_p2, max_f_evals,
                                                               verbose, verbose_interspace,
                                                               plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                               MOSD_IFSD_settings["theta_tol"], MOSD_IFSD_settings["qth_quantile"],
                                                               gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                               ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])
        
        elif refiner == 'Multi-Start':
            self._refiner_instance = None
        
        else:
            self._refiner_instance = None

        self._front_mode = False
        

    def search(self, p_list, f_list, problem, lambdas=None):
        assert lambdas is not None
        assert len(p_list) == len(lambdas)

        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        index_point = 0

        while index_point < len(p_list):

            self.output_data(f_list)

            if self.evaluate_stopping_conditions():
                break

            self.add_to_stopping_condition_current_value('max_iter', 1)

            model = Model('Scalarization Problem')
            model.setParam('OutputFlag', self._gurobi_verbose)
            model.setParam('FeasibilityTol', self._gurobi_feas_tol)
            model.setParam('IntFeasTol', self._gurobi_feas_tol)
            model.setParam("MemLimit", 14) # 14GB of RAM max
            model.setParam('TimeLimit', max(self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'), 0))

            eigenvals, _ = np.linalg.eig(problem.Q)

            if np.min(eigenvals.real) < 0:
                model.setParam('NonConvex', 2)

            x = model.addMVar(problem.n, lb=problem.lb_for_ini, ub=problem.ub_for_ini, name='x')
            delta = model.addMVar(problem.n, vtype=GRB.BINARY, name='delta')

            obj = 0
            counter = 0
            if 'Variance' in problem.objective_names:
                obj += lambdas[index_point][counter] * problem.scaling_factors['var_mean'] * 1 / 2 * (x @ problem.Q @ x)
                counter += 1
            if 'Mean' in problem.objective_names:
                obj += lambdas[index_point][counter] * problem.scaling_factors['var_mean'] * (problem.c @ x)
                counter += 1
            if 'SR' in problem.objective_names:
                raise NotImplementedError("Gurobi cannot handle SR objective function")
            if 'ESG' in problem.objective_names:
                obj += lambdas[index_point][counter] * problem.scaling_factors['esg'] * (problem.ESG @ x)
                counter += 1
            if 'Skew' in problem.objective_names:
                raise NotImplementedError("Gurobi cannot handle Skew objective function")
            
            model.setObjective(obj)

            for j in range(problem.n):
                model.addSOS(GRB.SOS_TYPE1, [x[j], delta[j]], [1, 1])
            model.addConstr(np.ones(problem.n) @ delta >= problem.n - problem.s, name='Cardinality constraint')

            if problem.family_name() == 'Portfolio':
                model.addConstr(quicksum(x) == 1, name='Simplex Constraint')
                if problem.betas is not None:
                    if problem.beta_lb is not None:
                        model.addConstr(problem.betas @ x >= problem.beta_lb)
                    if problem.beta_ub is not None:
                        model.addConstr(problem.betas @ x <= problem.beta_ub) 

            for j in range(problem.n):
                x[j].start = p_list[index_point, j]
                delta[j].start = 0 if abs(p_list[index_point, j]) > 0 else 1

            model.update()
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                new_p = np.array([s.x for s in model.getVars()])[: problem.n]

                if np.sum(np.abs(new_p) >= problem.sparsity_tol) > problem.s:
                    print('Warning! Found a not feasible point! Optimization over!')
                    print(new_p)
                    p_list[index_point, :] = np.full_like(new_p, np.nan)
                    f_list[index_point, :] = problem.evaluate_functions(p_list[index_point, :])

                else:
                    new_p[np.abs(new_p) < problem.sparsity_tol] = 0.
                    p_list[index_point, :] = new_p
                    f_list[index_point, :] = problem.evaluate_functions(p_list[index_point, :])

            else:
                print("Gurobi error ", model.Status)
                p_list[index_point, :] = np.full_like(p_list[index_point, :], np.nan)
                f_list[index_point, :] = problem.evaluate_functions(p_list[index_point, :])

            index_point += 1

        index_not_nan_inf_f = np.logical_and(~np.isnan(np.sum(f_list, axis=1)), ~np.isinf(np.sum(f_list, axis=1)))
        p_list = p_list[index_not_nan_inf_f, :]
        f_list = f_list[index_not_nan_inf_f, :]

        self.show_figure(p_list, f_list)

        self.output_data(f_list)
        self.close_figure()

        res_dict = {'p1': {'p_list': np.copy(p_list), 
                           'f_list': np.copy(f_list), 
                           'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}}

        if self._refiner_instance is not None:
            p_list, f_list, _ = self._refiner_instance.search(p_list, f_list, problem)
            res_dict['p2'] = {'p_list': np.copy(p_list), 
                              'f_list': np.copy(f_list), 
                              'elapsed_time': time.time() - self.get_stopping_condition_current_value('max_time')}

        return res_dict
