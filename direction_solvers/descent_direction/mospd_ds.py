import numpy as np
from gurobipy import Model, GRB, quicksum

from nsma.direction_solvers.descent_direction.dds import DDS

from direction_solvers.gurobi_settings import ExtendedGurobiSettings
from problems.extended_problem import ExtendedProblem


class MOSPDDS(DDS, ExtendedGurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float):
        DDS.__init__(self)
        ExtendedGurobiSettings.__init__(self, gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, time_limit: float = None):
        assert x_p is not None

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model("MOSPD Descent Direction")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)
            model.setParam("FeasibilityTol", self._gurobi_feasibility_tol)
            if time_limit is not None:
                model.setParam("TimeLimit", max(time_limit, 0))

            z = model.addMVar(n, lb=problem.lb_box, ub=problem.ub_box, name="z")
            beta = model.addMVar(1, lb=-np.inf, ub=0., name="beta")

            obj = beta - (x_p @ z) + 0.5 * (z @ z) + 0.5 * (x_p @ x_p)
            model.setObjective(obj)

            for j in range(m):
                model.addConstr(Jac[j, :] @ z <= beta + Jac[j, :] @ x_p, name='Jacobian Constraint n°{}'.format(j))

            model.addConstr(quicksum(z) == 1, name='Simplex Constraint')
            if problem.betas is not None:
                if problem.beta_lb is not None:
                    model.addConstr(problem.betas @ z >= problem.beta_lb)
                if problem.beta_ub is not None:
                    model.addConstr(problem.betas @ z <= problem.beta_ub) 

            model.update()

            for i in range(n):
                z[i].start = float(x_p[i])
            beta.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()

                d_p = np.array([s.x for s in sol][:n]) - x_p
                theta_p = obj.getValue()

            else:
                print("Gurobi status was:", model.Status)
                return np.zeros(n), 0

        except AttributeError:
            print("Error in the formulation of the Gurobi Model!")
            return np.zeros(n), 0

        return d_p, theta_p
