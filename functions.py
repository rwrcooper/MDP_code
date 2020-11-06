"""
File: MDP_functions.py
Author: riley cooper
Email: rwr.cooper@gmail.com
Description:
    MDP_functions for use in script MDP_algorithm.py.
"""

# import xarray as xr
import numpy as np
import random as rnd
import scipy.optimize
import gurobipy as gp
from gurobipy import GRB
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

'''
def model_solver_primal_gurobi(A, cost, distr, N):
    # Create a new model
    # m = gp.Model("Primal")
    # Create variables
    # v = np.ones((N,1))
    # changed shape and GRB.BINARY
    # x = m.addMVar(v, lb = GRB.INFINITY, ub = GRB.INFINITY, name="x")
    # also need to consider the bound on v (unbounded)
    # Set objective
    # obj = distr - (changed obj for just distr)
    # print(len(distr))
    # m.setObjective(distr*x, GRB.MAXIMIZE)
    # We already have A
    # We already have cost (rhs vector)
    # Add constraints
    # m.addConstr(A @ x <= cost, name="c")
    # Optimize model
    # m.optimize()
    # dvs = x.X
    # value = m.objVal
    # Create a new model
    m = gp.Model("matrix1")
    # Create variables
    x = m.addMVar(shape=N, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="x")
    # Set objective
    obj = distr.transpose()[0]
    m.setObjective(obj @ x, GRB.MAXIMIZE)
    A = sp.csr_matrix(A)
    rhs = cost.transpose()[0]
    # Add constraints
    m.addConstr(A @ x <= rhs, name="c")
    # Optimize model
    m.optimize()
    XX = m.X
    VV = m.objVal
    return(XX, VV)


def model_solver_dual_gurobi(A, cost, distr, N, k):
    # Create a new model
    m = gp.Model("Dual")
    # Create variables
    # changed shape and GRB.BINARY
    x = m.addMVar(N*k, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    # also need to consider the bound on v (unbounded)
    # Set objective
    # obj = distr - (changed obj for just distr)
    m.setObjective(cost @ x, GRB.MINIMIZE)
    # We already have A
    # We already have cost (rhs vector)
    # Add constraints
    A = sp.csr_matrix(A.transpose())
    m.addConstr(A @ x == distr.transpose()[0], name="c")
    # Optimize model
    m.optimize()
    dvs = x.X
    value = m.objVal
    return(dvs, value)
'''


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
    # print("Start Solve")

    result = scipy.optimize.linprog(
        c=cost, A_eq=A.transpose(), b_eq=distr, bounds=(0, None),
        method="interior-point", options={"maxiter": 1000, "tol": 1e-08})
    # print("End Solve")

    return(result.x, result.fun)

# def make_tpm_fred(n,k,r):
#    pos1 = make_pos(n/2,k,r)
#    pos2 = make_pos(n/2,k,r)
#    return(tpm)


def make_tpm_reg(n, k, r):
    pos = make_pos(n, k, r)
    tpm = make_tpm(n, k, r, pos)
    return(tpm)

# def tpm_maker_options(n,k,r,type):
#    if type == "Fred":
#        tpm = make_tpm_fred(n,k,r)
#    elif type == "Reg":
#        tpm = make_tpm_reg(n,k,r)
#    return(tpm)


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


def clusterC_maker(cluster, state_space):
    clusterC = [item for item in state_space if item not in cluster]
    return(clusterC)


# ##------------------Aggregated model functions
def model_aggregator(tpm, cluster, clusterC, cost, distr, n, k, d, beta):
    distr_d = make_distr_d(tpm, cluster, clusterC, cost, distr, n, k)
    cost_d = make_cost_d(tpm, cluster, clusterC, cost, distr, n, k, d)
    tpm_d = make_tpm_d(tpm, cluster, clusterC, cost, distr, n, k, d)
    A_d = make_A(nn=d+1, k=k, tpm=tpm_d, beta=beta)
    return(tpm_d, cost_d, distr_d, A_d)


def make_cost_d(tpm, cluster, clusterC, cost, distr, n, k, d):
    cost_d = []
    for i in cluster:
        for item in range(i*k, (i+1)*k):
            cost_d.append(cost[item][0])
    for a in range(k):
        extra = 0
        for i in clusterC:
            extra = extra + cost[i*k+a][0]
        extra = extra/(n-d)
        cost_d.append(extra)
    cost_d = np.array(cost_d)
    return(cost_d)


def make_distr_d(tpm, cluster, clusterC, cost, distr, n,k):
    distr_d = [distr[i] for i in cluster]
    extra_element = sum(distr[i] for i in clusterC)
    distr_d.append(extra_element)
    distr_d = np.array(distr_d)
    return(distr_d)


def make_tpm_d(tpm, cluster, clusterC, cost, distr, n, k, d):
    tpm_d = np.zeros(((d+1)*k, d+1))
    column_counter = -1

    for i in cluster:
        column_counter += 1
        new_column = []

        for j in cluster:

            for a in range(k):
                new_column.append(tpm[j*k+a, i])

        for a in range(k):
            entry = 0

            for j in clusterC:
                entry = entry + tpm[j*k+a, i]
            entry = entry/(n-d)
            new_column.append(entry)
        tpm_d[:, column_counter] = new_column
    last_column = []

    for i in cluster:

        for a in range(k):
            last_column.append(sum([tpm[(i*k)+a, j] for j in clusterC]))

    for a in range(k):
        extra = 0

        for i in clusterC:

            for j in clusterC:
                extra = extra + tpm[i*k+a, j]
        extra = extra/(n-d)
        last_column.append(extra)

    tpm_d[:, -1] = last_column

    return(tpm_d)


def make_opt_policy(dvs_dual_d, k, d):
    opt_pol = []

    # counter = -1

    for dd in range(d+1):

        for a in range(k):

            if dvs_dual_d[dd*k+a] > 0.0000001:
                opt_pol.append(a)
                break

    return(opt_pol)


def make_v_hat(dvs_dual_d, dvs_primal_d, opt_pol, cluster, clusterC, cost,
               beta, d, k, tpm):
    v_hat = []
    for kk in clusterC:
        c = cost[kk*k+opt_pol[-1]]
        b1 = beta*sum([tpm[kk*k+opt_pol[-1], cluster[j]]*dvs_primal_d[
            j] for j in range(d)])
        b2 = beta*sum([tpm[kk*k+opt_pol[-1], k_dash]*dvs_primal_d[
            -1] for k_dash in clusterC])
        v_kk = c+b1+b2-dvs_primal_d[-1]
        v_hat.append(v_kk)
    return(v_hat)


def make_y_hat(dvs_dual_d, dvs_primal_d, opt_pol, cluster, clusterC, cost,
               beta, d, k, tpm, distr):
    y_hat = []
    for kk in clusterC:
        b3 = sum([tpm[k_dash*k+opt_pol[-1], kk] for k_dash in clusterC])
        b1 = (1-beta*b3)*dvs_dual_d[d*k+opt_pol[-1]]
        b2 = beta*sum([tpm[cluster[i]*k+opt_pol[i], kk]*dvs_dual_d[i*k+opt_pol[
            i]] for i in range(d)])
        y_kk = b1-b2-distr[kk]
        y_hat.append(y_kk)
    return(y_hat)


def make_pre_M(v_hat, y_hat, n, d):
    pre_M = []
    L = len(v_hat)
    for i in range(L):
        pre_M.append(v_hat[i]*y_hat[i])
    pre_M = np.array(pre_M)
    return(pre_M)


def M_function(dvs_dual_d, dvs_primal_d, opt_pol, cluster, clusterC, cost,
               beta, n, d, k, tpm, distr):
    v_hat = make_v_hat(dvs_dual_d, dvs_primal_d, opt_pol, cluster, clusterC,
                       cost, beta, d, k, tpm)
    y_hat = make_y_hat(dvs_dual_d, dvs_primal_d, opt_pol, cluster, clusterC,
                       cost, beta, d, k, tpm, distr)
    pre_M = make_pre_M(v_hat, y_hat, n, d)
    return(pre_M)


def state_chooser(pre_M, clusterC):
    new_state_index = pre_M.argmax()
    new_state = clusterC[new_state_index]
    return(new_state)


def model_augmentor(cluster, new_state):
    new_cluster = cluster
    new_cluster.append(new_state)
    new_clusterC = clusterC_maker(cluster, state_space)
    return(new_cluster, new_clusterC, d)
