import numpy as np
import scipy
from math import comb
from itertools import combinations_with_replacement

from nsma.general_utils.pareto_utils import pareto_efficient

from problems.extended_problem import ExtendedProblem


def points_initialization(problem: ExtendedProblem, seed: int):
    if problem.family_name() == 'Portfolio':
        p_list = problem.generate_feasible_points_array('single_supp', 2 * problem.n, seed=seed)
    else:
        raise NotImplementedError("No initialization provided for ", problem.family_name())

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    return p_list, f_list, n_initial_points


def points_initialization_gurobi(problem: ExtendedProblem, seed: int):
    if problem.family_name() == 'Portfolio':
        if problem.m == 2:
            p_list = problem.generate_single_feasible_point('rand_sparse', seed=seed)
            p_list = np.tile(p_list, (problem.n * 2, 1))
            lambdas = np.array([[1, 2 ** ((2*i - 2 * problem.n + 1) / 2)] for i in range(2 * problem.n)])
        else:
            lambdas = create_lambdas(problem)
            p_list = problem.generate_single_feasible_point('rand_sparse', seed=seed)
            p_list = np.tile(p_list, (len(lambdas), 1))
    else:
        raise NotImplementedError("No initialization provided for ", problem.family_name())

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    return p_list, lambdas, f_list, n_initial_points


def points_initialization_scalarization(problem: ExtendedProblem, seed: int):
    if problem.family_name() == 'Portfolio':
        lambdas = create_lambdas(problem)
        p_list = problem.generate_single_feasible_point('rand_sparse', seed=seed)
        p_list = np.tile(p_list, (len(lambdas), 1))
        
    else:
        raise NotImplementedError("No initialization provided for ", problem.family_name())

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    return p_list, lambdas, f_list, n_initial_points


def points_postprocessing(p_list: np.array, f_list: np.array, problem: ExtendedProblem):
    assert len(p_list) == len(f_list)
    old_n_points, _ = p_list.shape

    for p in range(old_n_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    p_list, f_list = remove_duplicates_point(p_list, f_list)

    n_points, n = p_list.shape

    if old_n_points - n_points > 0:
        print('Warning: found {} duplicate points'.format(old_n_points - n_points))

    feasible = [True] * n_points
    feasibility_tol = 1e-4
    infeasible_points = 0
    for p in range(n_points):
        
        if (p_list[p, :] < -feasibility_tol + problem.lb_for_ini).any() or (p_list[p, :] > feasibility_tol + problem.ub_for_ini).any():
            feasible[p] = False
        elif np.sum(np.abs(p_list[p, :]) >= problem.sparsity_tol) > problem.s:
            feasible[p] = False
        elif np.abs(np.sum(p_list[p, :]) - 1) > feasibility_tol:
            feasible[p] = False
        elif problem.betas is not None:
            if problem.beta_lb is not None:
                if problem.betas @ p_list[p, :] < -feasibility_tol + problem.beta_lb:
                    feasible[p] = False
            if problem.beta_ub is not None:
                if problem.betas @ p_list[p, :] > feasibility_tol + problem.beta_ub:
                    feasible[p] = False
        
        if not feasible[p]:
            infeasible_points += 1
    
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    efficient_point_idx = pareto_efficient(f_list)
    p_list = p_list[efficient_point_idx, :]
    f_list = f_list[efficient_point_idx, :]

    print('Results: found {} feasible efficient points'.format(len(p_list)))
    print()

    return p_list, f_list


def remove_duplicates_point(p_list: np.array, f_list: np.array):

    is_duplicate = np.array([False] * p_list.shape[0])

    D = scipy.spatial.distance.cdist(p_list, p_list)
    D[np.triu_indices(len(p_list))] = np.inf

    D[np.isnan(D)] = np.inf

    is_duplicate[np.any(D < 1e-16, axis=1)] = True

    p_list = p_list[~is_duplicate]
    f_list = f_list[~is_duplicate]

    return p_list, f_list


def extract_support(x):
    return frozenset(np.nonzero(x)[0])


def create_lambdas(problem: ExtendedProblem):
    G = 1
    while True:
        num_combinations = comb(G + problem.m - 1, problem.m - 1)
        if num_combinations >= problem.n * 2:
            break
        G += 1
    
    points = np.linspace(0, 1, G + 1)
    combinations = list(combinations_with_replacement(range(G + 1), problem.m - 1))
    results = []
    for combi in combinations:
        breaks = np.array(combi) / G
        weights = np.diff([0] + breaks.tolist() + [1])
        if np.all(weights >= 0):
            results.append(weights)
    
    return np.array([res for res in results])