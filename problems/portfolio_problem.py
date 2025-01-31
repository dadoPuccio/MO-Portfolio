import numpy as np
import tensorflow as tf
import pickle
import scipy
import pandas as pd
from general_utils.parameter_extraction_utils import get_portfolio_parameters
from general_utils.projection_utils import compute_projection_to_linear_constrs

from problems.extended_problem import ExtendedProblem


class PortfolioProblem(ExtendedProblem):

    def __init__(self, problem_path: str, s: int, scaling_factors: dict, sparsity_tol: float, objectives: list, financial_index: str, scalarization: bool, beta_bounds: list):

        self.__problem_path = problem_path
        
        if '.xlsx' in problem_path:

            ESG_df = pd.read_excel(problem_path, "ESG_"+financial_index, header=None)

            try:
                index_df = pd.read_excel(problem_path, financial_index, header=None).iloc[:,1].to_numpy()
            except:
                index_df = None

            exp_returns, covariance_matr, coskw_tensor, ESG_scores, betas = get_portfolio_parameters(
                pd.read_excel(problem_path, "Prezzi_" + financial_index, header=None).iloc[:,1:].to_numpy(),
                ESG_df.drop(ESG_df.columns[0], axis=1), 
                index_df
            )

            self.Q = covariance_matr
            self.c = -exp_returns
            self.ESG = -ESG_scores
            self.Skew = -coskw_tensor
            self.betas = betas

        else:
            variance_covariance = scipy.io.loadmat(problem_path + "/" + problem_path[-4:] + 'variance_covariance.mat')
            self.Q = variance_covariance['variance_covariance']

            expected_returns = scipy.io.loadmat(problem_path + "/" + problem_path[-4:] + 'expected_returns.mat')
            self.c = np.squeeze(-expected_returns['expected_returns'])  

            self.betas=None
        
        ExtendedProblem.__init__(self, len(self.c), s, sparsity_tol)

        self.beta_lb = beta_bounds[0]
        self.beta_ub = beta_bounds[1]

        obj_list = []
        if 'Variance' in objectives:
            obj_list.append(scaling_factors['var_mean'] * 1 / 2 * tf.tensordot(self._z, tf.tensordot(self.Q, self._z, axes=1), axes=1))
        if 'Mean' in objectives:
            obj_list.append(scaling_factors['var_mean'] * tf.tensordot(self._z, self.c, axes=1))
        if 'SR' in objectives:
            obj_list.append(tf.tensordot(self._z, self.c, axes=1)/tf.sqrt(tf.tensordot(self._z, tf.tensordot(self.Q, self._z, axes=1), axes=1)))
        if 'ESG' in objectives:
            obj_list.append(scaling_factors['esg'] * tf.tensordot(self._z, self.ESG, axes=1))
        if 'Skew' in objectives:
            obj_list.append(scaling_factors['skew'] * tf.tensordot(tf.tensordot(tf.tensordot(self.Skew, self._z, axes=1), self._z, axes=1), self._z,axes=1))

        self.objectives = obj_list

        self.lb_for_ini = np.zeros(self.n)
        self.ub_for_ini = np.ones(self.n)

        self._L = max(0, scaling_factors['var_mean'] * np.linalg.norm(self.Q, ord=2))

        # Parameters required for scalarization
        self.scalarization = scalarization
        self.objective_names = objectives
        self.scaling_factors = scaling_factors
 
    def evaluate_functions(self, x: np.array):
        if np.isnan(np.sum(x)):
            return np.full(self.m, np.nan)
        return super().evaluate_functions(x)
        
    def name(self):
        return self.__problem_path

    @staticmethod
    def family_name():
        return "Portfolio"
    
    def generate_feasible_points_array(self, mod: str, size: int, seed: int = None):
        assert mod.lower() in ['rand_sparse', 'single_supp']
        assert seed is not None

        if mod.lower() == 'rand_sparse':
            rng = np.random.default_rng(seed)
            p_list = np.zeros((size, self.n), dtype=float)
            for i in range(size):
                p_list[i, :] = np.random.uniform(self.lb_for_ini, self.ub_for_ini)
                p_list[i, rng.choice(self.n, size=self.n - self.s, replace=False)] = 0.
                p_list[i, :] = p_list[i, :] / sum(p_list[i, :])

        elif mod.lower() == 'single_supp':
            p_list = np.zeros((size, self.n), dtype=float)
            for i in range(self.n):
                p_list[i, i] = 1

            rng = np.random.default_rng(seed)
            for i in range(self.n, size):
                p_list[i, :] = np.random.uniform(self.lb_for_ini, self.ub_for_ini)
                p_list[i, rng.choice(self.n, size=self.n - self.s, replace=False)] = 0.
                p_list[i, :] = p_list[i, :] / sum(p_list[i, :])

        else:
            raise NotImplementedError

        if self.betas is not None:
            if self.beta_lb is not None or self.beta_ub is not None:
                for i in range(p_list.shape[0]):
                    p_list[i, :] = compute_projection_to_linear_constrs(self, p_list[i, :], None)

        return p_list
    

    def generate_single_feasible_point(self, mod: str, seed: int = None):
        assert mod.lower() in ['rand_sparse', 'single_supp']
        assert seed is not None

        if mod.lower() == 'rand_sparse':
            rng = np.random.default_rng(seed)
            p_list = np.random.uniform(self.lb_for_ini, self.ub_for_ini)
            p_list[rng.choice(self.n, size=self.n - self.s, replace=False)] = 0.
            p_list = p_list / sum(p_list)

        elif mod.lower() == 'single_supp':

            p_list = np.zeros((self.n,), dtype=float)
            random_index = np.random.randint(0, self.n)
            p_list[random_index] = 1

        if self.betas is not None:
            if self.beta_lb is not None or self.beta_ub is not None:
                p_list = compute_projection_to_linear_constrs(self, p_list, None)

        return p_list
    