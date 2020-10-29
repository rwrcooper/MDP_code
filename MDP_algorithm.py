"""
File: MDP_algorithm.py
Author: riley cooper
Email: rwr.cooper@gmail.com
Description:
    Script to execute testing on the MDP_algorithm.

Usage:
    multi_model_slope_analysis.py

Options:
    -h, --help          Show this screen and exit.
"""

from docopt import docopt
from MDP_functions import *

# TODO: make whole procedure into function and get docopts working
# TODO: make script faster, more readible, remove need for gurobi

# ##----------------DATA
# n = 1000

# nlist = [100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3500,4000,4500,5000,6000,7000,8000,9000,10000]


def do_thing():

    nlist = [1000]

    N = len(nlist)

    dlist = []
    opt_list = []
    opt_val_list_list = []

    for i in range(N):
        n = nlist[i]
        print("n = ")
        print(n)

        k = 10
        r = 10
        beta = 0.8

        state_space, cost, tpm, distr, opt_dvs, opt_val = set_up(
            n, k, r, beta, type="Reg")

        d = 1
        cluster = [0]
        clusterC = clusterC_maker(cluster, state_space)
        opt_value_list = []

        diff = 1
        while diff >= 0:
            print("d = ")
            print(d)

            tpm_d, cost_d, distr_d, A_d = model_aggregator(
                tpm, cluster, clusterC, cost, distr, n, k, d, beta)

            dvs_primal_d, val_d_primal = model_solver_primal_gurobi(
                A=A_d, cost=cost_d, distr=distr_d, N=d+1)

            dvs_dual_d, val_d_dual = model_solver_dual_gurobi(
                A=A_d, cost=cost_d, distr=distr_d, N=d+1, k=k)

            opt_pol = make_opt_policy(dvs_dual_d, k, d)

            pre_M = M_function(dvs_dual_d, dvs_primal_d, opt_pol, cluster,
                               clusterC, cost, beta, n, d, k, tpm, distr)

            opt_value_list.append(val_d_dual)

            print(val_d_primal)
            print(val_d_dual)
            print(dvs_primal_d)
            print(dvs_dual_d)
            break

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

            new_state = state_chooser(pre_M, clusterC)

            if diff < 0.01:
                print("EXIT: Difference is less than 0.")
                dlist.append(d)
                opt_val_list_list.append(opt_value_list)
            cluster.append(new_state)

            clusterC = clusterC_maker(cluster, state_space)

            d += 1

    print(opt_val_list_list)
    print(dlist)


def main(args):

    if True:
        do_thing()

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
