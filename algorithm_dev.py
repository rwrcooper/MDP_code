"""
File: algorithm_dev.py
Author: riley cooper
Email: rwr.cooper@gmail.com
Description:
    Script to develop MDP algorithms.

Usage:
    algorithm_dev.py

Options:
    -h, --help          Show this screen and exit.
"""


from docopt import docopt
import xarray as xr
import numpy as np
import random as rnd
import scipy.optimize
import scipy.sparse as sp


def make_pos(n, k, r):
    pos = np.zeros((n*k, r))
    # select random positions and update pos matrix
    for row in range(n*k):
        for col in range(r):
            pos[row][col] = rnd.randint(1, n-1)
    return(pos)


def make_tpm(n, k, r, pos):
    tpm = np.zeros((n*k, n))
    # now randomly generate porobabilities
    for row in range(n*k):
        # generate positive probability in first column
        tpm[row][0] = rnd.random()
        # generate positive probability in remaining select columns
        for col in range(r):
            # col_input = int(pos[row][col])
            tpm[row][int(pos[row][col])] = rnd.random()
        # divide entries by row sum to ensure probabilities sum to 1
        row_sum = np.sum(tpm[row])
        tpm[row] = tpm[row]/row_sum
    return(tpm)


def make_cost(n, k):
    cost = np.zeros((n*k, 1))
    for aa in range(n*k):
        cost[aa] = rnd.random()
    return(cost)


def make_initial_distr(n, k):
    distr = np.zeros((n, 1))
    for aa in range(n):
        distr[aa] = rnd.random()
    return(distr)


def make_D(nn, k):
    ii = np.identity(nn)
    D = np.zeros((nn*k, nn))
    count = -1
    for i in range(nn):
        for a in range(k):
            count = count + 1
            D[count] = ii[i]
    return(D)


def make_A(nn, k, tpm, beta):
    # count = -1
    A = np.zeros((nn*k, nn))
    # for i in range(n):
    #    for a in range(k):
    #        count = count +1
    #        for j in range(n):
    #            kron_delt = 0
    #            if i == j:
    #                kron_delt = 1
    #            A[count][j] = kron_delt-beta*tpm[count][j]
    D = make_D(nn, k)
    A = D-beta*tpm
    return(A)


def model_solver_primal(A, cost, distr):
    # this can be used for original model or aggregated model
    # will solve the primal.
    # will basically just be using linprog().....
    result = scipy.optimize.linprog(
        c=-distr, A_ub=A, b_ub=cost, bounds=(None, None),
        method="interior-point", callback=None,
        options={"maxiter": 1000, "tol": 1e-08})
    # method='simplex', callback=None,
    # options={'c0': None, 'A': None, 'b': None, 'postsolve_args': None,
    # 'maxiter': 1000, 'tol': 1e-09, 'disp': False, 'bland': False}, x0=None)

    return(result.x, -result.fun)


def model_solver_dual(A, cost, distr):
    # this can be used for original model or aggregated model
    # will solve the primal.
    # will basically just be using linprog().....
    print("Start Solve")

    result = scipy.optimize.linprog(
        c=cost, A_eq=A.transpose(), b_eq=distr, bounds=(0, None),
        method="interior-point", options={"maxiter": 1000, "tol": 1e-08})
    print("End Solve")

    return(result.x, result.fun)


def make_tpm_reg(n, k, r):
    pos = make_pos(n, k, r)
    tpm = make_tpm(n, k, r, pos)
    return(tpm)


def set_up(n, k, r, beta, type):
    # state space
    state_space = [i for i in range(n)]
    # make transition probability matrix with options
    # tpm = tpm_maker_options(n,k,r,type)
    # make reg tpm
    tpm = make_tpm_reg(n, k, r)

    # cost
    cost = make_cost(n, k)

    # initial distr
    distr = make_initial_distr(n, k)

    # matrix A for linear program
    A = make_A(n, k, tpm, beta)

    # solve original problem for opt value and decision variables.
    opt_dvs, opt_val = model_solver_primal(A, cost, distr)
    # opt_dvs, opt_val = model_solver_primal(A, cost, distr)

    return(state_space, cost, tpm, distr, opt_dvs, opt_val)


def main(args):

    # do something

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
