import numpy as np
from gurobipy import Model, GRB, quicksum, GurobiError

from problems.extended_problem import ExtendedProblem


def compute_projection_to_linear_constrs(problem: ExtendedProblem, x_p: np.array = None, time_limit: float = None,
                       gurobi_verbose: bool = False, gurobi_method: int = 1, gurobi_feasibility_tol: float = 1e-7):
    assert x_p is not None
    
    try:
        model = Model("Projection to Linear Constraints")
        model.setParam("OutputFlag", gurobi_verbose)
        model.setParam("Method", gurobi_method)
        model.setParam("FeasibilityTol", gurobi_feasibility_tol)
        model.setParam('IntFeasTol', gurobi_feasibility_tol)
        model.setParam("MemLimit", 14) # 14GB of RAM max
        if time_limit is not None:
            model.setParam("TimeLimit", max(time_limit, 0))

        z = model.addMVar(problem.n, lb=problem.lb_for_ini, ub=problem.ub_for_ini, name="z")
        delta = model.addMVar(problem.n, vtype=GRB.BINARY, name='delta')

        obj = -(x_p @ z) + 0.5 * (z @ z) + 0.5 * (x_p @ x_p) + 1
        model.setObjective(obj)

        for j in range(problem.n):
            model.addSOS(GRB.SOS_TYPE1, [z[j], delta[j]], [1, 1])
        model.addConstr(np.ones(problem.n) @ delta >= problem.n - problem.s, name='Cardinality constraint')
        
        model.addConstr(quicksum(z) == 1, name='Simplex Constraint')
        if problem.betas is not None:
            if problem.beta_lb is not None:
                model.addConstr(problem.betas @ z >= problem.beta_lb)
            if problem.beta_ub is not None:
                model.addConstr(problem.betas @ z <= problem.beta_ub) 

        model.update()

        for i in range(problem.n):
            z[i].start = float(x_p[i])

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            sol = model.getVars()
            x_projected = np.array([s.x for s in sol][:problem.n])

            if np.sum(np.abs(x_projected) >= problem.sparsity_tol) > problem.s:
                print('Warning! Found a not feasible point! Optimization over!')
                print(x_projected)
                return np.full_like(x_p, np.nan)
            else:
                x_projected[np.abs(x_projected) < problem.sparsity_tol] = 0.

        else:
            print("Gurobi obtained status code:", model.Status)
            return np.full_like(x_p, np.nan)

    except GurobiError as e:
        print("Error in projection:", type(e).__name__, e)
        return np.full_like(x_p, np.nan)
    
    return x_projected
