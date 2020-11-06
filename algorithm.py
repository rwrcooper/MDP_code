"""
File: MDP_algorithm.py
Author: riley cooper
Email: rwr.cooper@gmail.com
Description:
    Script to execute testing on the MDP_algorithm.

Usage:
    MDP_algorithm.py

Options:
    -h, --help          Show this screen and exit.
"""

from docopt import docopt
from functions import model_solver_primal, model_solver_dual, set_up
from functions import clusterC_maker, model_aggregator, make_opt_policy
from functions import M_function, state_chooser

# TODO: make whole procedure into function and get docopts working
# TODO: make script faster, more readible, remove need for gurobi

# TODO: have these values as inputs
n = 100
k = 10
r = 10
beta = 0.8

# TODO: save all data with appropriate formating


def algorithm_test(n, k, r, beta, type):

    dlist = []
    opt_val_list_list = []
    parameter_str = f"n, k, r, beta = {n}, {k}, {r}, {beta}"
    print(f"Testing algorithm type: {type} with parameters: {parameter_str}.")

    state_space, cost, tpm, distr, opt_dvs, opt_val = set_up(
        n, k, r, beta, type="Reg")

    d = 1
    cluster = [0]
    clusterC = clusterC_maker(cluster, state_space)
    opt_value_list = []
    y_hat_list_g = []
    v_hat_list_g = []

    diff = 1
    while diff >= 0:
        print(f"d = {d}")

        cost_d, distr_d, A_d = model_aggregator(
            tpm, cluster, clusterC, cost, distr, n, k, d, beta)

        dvs_primal_d, val_d_primal = model_solver_primal(
            A=A_d, cost=cost_d, distr=distr_d)

        dvs_dual_d, val_d_dual = model_solver_dual(
            A=A_d, cost=cost_d, distr=distr_d)

        opt_value_list.append(val_d_dual)

        opt_pol = make_opt_policy(dvs_dual_d, k, d)

        # print(f"Optimal value (primal): {val_d_primal}")
        # print(f"Optimal value (dual): {val_d_dual}")
        # print(f"Decision variables (primal): \n {dvs_primal_d}")
        # print(f"Decision variables (dual): \n {dvs_dual_d}")
        # break

        if type == "alg":

            pre_M = M_function(dvs_dual_d, dvs_primal_d, opt_pol, cluster,
                               clusterC, cost, beta, n, d, k, tpm, distr)

            test = True

            for item in pre_M:
                test = test and item <= 0

            if test:
                diff = -1
                print("EXIT: M is empty.")
                dlist.append(d)
                opt_val_list_list.append(opt_value_list)
                break

            if d > 1:
                diff = opt_value_list[-2]-opt_value_list[-1]

            if diff < 0.00001:
                print("STOP: Difference is less than 0.")
                dlist.append(d)
                opt_val_list_list.append(opt_value_list)
                print(f"Failed at d = {d}")
                print(f"Change in optimal value: \n {opt_value_list}")
                print(f"Optimal value: {opt_val}")
                abs_rel_diff = abs((opt_val-opt_value_list[-1])/opt_val)
                print(f"Absolute relative difference: {abs_rel_diff}")

                # TODO: save data now.

            new_state = state_chooser(pre_M, clusterC)

        elif type == "greedy":

            # for all states s in clusterC, aggregate s to cluster, solve
            # linear program for opt_val, calculate differences, select state
            # with smallest difference, calculare y_hat and v_hat and save data

            # new_state = ???
            # save value of v_hat, y_hat and their product


        cluster.append(new_state)

        clusterC = clusterC_maker(cluster, state_space)

        d = d + 1


def main(args):

    if True:
        algorithm_test(n, k, r, beta, type)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
