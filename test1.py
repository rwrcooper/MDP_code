a=str(x)
b="\x"
c = 'x'
print(a)
print(b)
print(c)
###-----------TESTING/BUILDING

'''
d = 5

cluster = [0,2,4,6,8]

clusterC = clusterC_maker(cluster, state_space)

cost_d = make_cost_d(tpm, cluster, clusterC, cost, distribution, n,k,d)

distribution_d = make_distribution_d(tpm, cluster, clusterC, cost, distribution, n,k)

tpm_d = make_tpm_d(tpm, cluster, clusterC, cost, distribution, n,k,d)

A_d = make_A(nn=d+1,k=k,tpm = tpm_d,beta=beta)

dvs_primal_d, val_d_primal = model_solver_primal(A = A_d, cost = cost_d, distribution = distribution_d)

dvs_dual_d,val_d_dual = model_solver_dual(A = A_d, cost = cost_d, distribution = distribution_d)

opt_pol = make_opt_policy(dvs_dual_d,k,d)

v_hat = make_v_hat(dvs_dual_d, dvs_primal_d, opt_pol,cluster, clusterC,cost, beta, d, k, tpm)

y_hat = make_y_hat(dvs_dual_d, dvs_primal_d, opt_pol,cluster, clusterC,cost, beta, d, k, tpm,distribution)

print(v_hat)

print(y_hat)

pre_M = make_pre_M(v_hat, y_hat, n, d)

new_state = state_chooser(pre_M, clusterC)
'''
