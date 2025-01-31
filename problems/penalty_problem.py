import numpy as np

from nsma.problems.problem import Problem


class PenaltyProblem(Problem):

    def __init__(self, problem: Problem, y_0: np.array, tau_0: float, lambdas: np.array = None):
        Problem.__init__(self, problem.n)

        self.__problem = problem

        self.__y = y_0
        self.__tau = tau_0

        self.lb_box = problem.lb_for_ini
        self.ub_box = problem.ub_for_ini

        self.lambdas = lambdas

    def evaluate_functions(self, x: np.array):

        if self.lambdas is None:
            return self.__problem.evaluate_functions(x) + self.__tau/2 * np.dot(x - self.__y, x - self.__y) 
        
        else:
            objs = self.__problem.evaluate_functions(x)
            return np.array([np.sum(np.array([self.lambdas[j] * objs[j] for j in range(self.__problem.m)])) + self.__tau/2 * np.dot(x - self.__y, x - self.__y)])

    def evaluate_functions_jacobian(self, x: np.array):

        functions_jacobian = self.__problem.evaluate_functions_jacobian(x)
        penalty_gradient = self.__tau * (x - self.__y)

        if self.lambdas is None:
            jacobian = np.zeros((self.__problem.m, self.__problem.n))
            for i in range(self.__problem.m):
                jacobian[i, :] = functions_jacobian[i, :] + penalty_gradient
        
        else:
            jacobian = np.sum(np.array([self.lambdas[j] * functions_jacobian[j, :] for j in range(self.__problem.m)]), axis=0).reshape(1,-1) + penalty_gradient

        return jacobian

    def evaluate_constraints(self, x: np.array):
        return np.empty(0)

    def evaluate_constraints_jacobian(self, x: np.array):
        return np.empty(0)

    def check_point_feasibility(self, x: np.array):
        return True

    @Problem.objectives.setter
    def objectives(self, objectives: list):
        raise RuntimeError

    @Problem.general_constraints.setter
    def general_constraints(self, general_constraints: list):
        raise RuntimeError

    @Problem.lb.setter
    def lb(self, lb: np.array):
        raise RuntimeError

    @Problem.ub.setter
    def ub(self, ub: np.array):
        raise RuntimeError

    @property
    def n(self):
        return self.__problem.n

    @property
    def m(self):
        return self.__problem.m if self.lambdas is None else 1

    @staticmethod
    def name():
        return "Penalty Problem"

    @staticmethod
    def family_name():
        return "Penalty Problem"

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y: np.array):
        self.__y = y

    @property
    def tau(self):
        return self.__tau

    @tau.setter
    def tau(self, tau: float):
        self.__tau = tau

    @property
    def betas(self):
        return self.__problem.betas

    @property
    def beta_lb(self):
        return self.__problem.beta_lb
    
    @property
    def beta_ub(self):
        return self.__problem.beta_ub
