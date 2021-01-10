import numpy as np
from ortools.linear_solver import pywraplp

def manhattan(a, b):
    return np.linalg.norm(a-b, ord=1)

def euclidean(a, b):
    return np.linalg.norm(a-b, ord=2)

def emd(w, w_prime, d, max_value=1.0):
    """
    :param ndarray w:        The supply values of the transportation problem of shape (m).
    :param ndarray w_prime:  The demand values of the transportation problem of shape (n).
    :param ndarray d:        The distance between the clusters w and w_prime of shape (m, n).
    :param float max_value:  The max value that the supply and demand weights can have.

    :return:  The value of the EMD metric for a specific instance of the above parameters.
    :rtype:   Optional(double)
    """
    # get the number of supply and demand of the transportation problem
    m, n = d.shape
    # create the mip solver with the GLOP backend
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # create the variables of the transportation problem f_{ij}
    f_vars = []
    for w_var in range(m):
        f_constraint = []
        for w_prime_var in range(n):
            f_constraint.append(solver.NumVar(0.0, max_value, 'f_{}{}'.format(w_var+1,
                                                                              w_prime_var+1)))
        f_vars.append(f_constraint)

    # w (supply) constraints
    for w_var in range(m):
        sum_constraint = sum(f_vars[w_var])
        solver.Add(sum_constraint == w[w_var])

    # w prime (demand) constraints
    for w_prime_var in range(n):
        sum_constraint = sum(list(list(zip(*f_vars))[w_prime_var]))
        solver.Add(sum_constraint == w_prime[w_prime_var])

    # build the minimization target: \sum_{i=1}^m \sum_{j=1}^n f_{ij}d_{ij}
    target_min = sum([f_vars[w_var][w_prime_var] * d[w_var, w_prime_var]
                      for w_var in range(m) for w_prime_var in range(n)])

    # minimize objective function
    solver.Minimize(target_min)

    # solve the problem
    status = solver.Solve()

    # check for a solution and if it exists, return it
    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value()
    else:
        return None


# w = np.array([0.2, 0.3, 0.1, 0.4])
# w_prime = np.array([0.1, 0.1, 0.6, 0.2])
# d = np.array([[0, 7, 7, 12], [7, 0, 12, 7], [7, 12, 0, 7], [12, 7, 7, 0]])
#
# v = emd(w, w_prime, d)
# print(v)
