from direction_solvers.descent_direction.constrained_steepest_descent_ds import ConstrainedSteepestDescentDS
from direction_solvers.descent_direction.moiht_ds import MOIHTDS
from direction_solvers.descent_direction.constrained_subspace_steepest_descent_ds import ConstrainedSubspaceSteepestDescentDS
from direction_solvers.descent_direction.mospd_ds import MOSPDDS

class DirectionDescentFactory:

    @staticmethod
    def get_direction_calculator(direction_type: str, gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float):

        if direction_type == 'Feasible_Steepest_Descent_DS':
            return ConstrainedSteepestDescentDS(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'MOIHT_DS':
            return MOIHTDS(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'Subspace_Steepest_Descent_DS':
            return ConstrainedSubspaceSteepestDescentDS(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'MOSPD_DS':
            return MOSPDDS(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        else:
            raise NotImplementedError
