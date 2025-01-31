import time
import math
import numpy as np

from algorithms.memetic.nm_port import NSMAPortfolio

from problems.extended_problem import ExtendedProblem


class NSGA2Portfolio(NSMAPortfolio):

    def __init__(self, 
                 max_t_p1: float, max_t_p2: float, max_f_evals: int, 
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int, 
                 pop_size: int, 
                 crossover_probability: float,
                 crossover_eta: float,
                 mutation_eta: float,
                 gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float,
                 refiner: str, MOSD_IFSD_settings: dict,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):
        
        NSMAPortfolio.__init__(self,
                                max_t_p1, max_t_p2,
                                max_f_evals,
                                verbose,
                                verbose_interspace,
                                plot_pareto_front,
                                plot_pareto_solutions,
                                plot_dpi,
                                pop_size,
                                crossover_probability,
                                crossover_eta,
                                mutation_eta, None,
                                0, 0.0, np.inf,
                                0, 0, 0, 0, 
                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                0., 0.,
                                refiner, MOSD_IFSD_settings,
                                ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)