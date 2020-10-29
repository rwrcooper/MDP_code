###----------------INTRO


###----------------SET UP
import numpy as np
import random as rnd
import scipy.optimize
#from scipy.optimize import linprog


###----------------TESTING THINGS
#print("Hello")
#aMatrix = np.zeros((2,3))
#print(aMatrix)
#aTest = np.ones((3,2))
#print(aTest[1]/2)


###----------------DATA
n = 1000
k = 5
r = 5
dStart = 2
beta = 0.8

dMax = n
d = dStart
#initialDistribution
#initialPolicy


def make_pos(n,k,r):
    pos = np.zeros((n*k, r))
    #select random positions and update pos matrix
    for row in range(n*k):
        for col in range(r):
            pos[row][col] = rnd.randint(1,n-1)
    return(pos)

pos = make_pos(n,k,r)
#print(pos)

def make_tpm(n,k,r,pos):
    tpm = np.zeros((n*k, n))
    #now randomly generate porobabilities
    for row in range(n*k):
        #generate positive probability in first column
        tpm[row][0] = rnd.random()
        #generate positive probability in remaining select columns
        for col in range(r):
            #col_input = int(pos[row][col])
            tpm[row][int(pos[row][col])] = rnd.random()
        #divide entries by row sum to ensure probabilities sum to 1
        row_sum = np.sum(tpm[row])
        tpm[row] = tpm[row]/row_sum
    return(tpm)

tpm = make_tpm(n,k,r,pos)
#print(tpm)

#randomly generate cost values,
def make_cost(n,k):
    cost = np.zeros((n*k,1))
    for aa in range(n*k):
        cost[aa] = rnd.random()
    return(cost)

def make_initial_distribution(n,k):
    distr = np.zeros((n,1))
    for aa in range(n):
        distr[aa] = rnd.random()
    return(distr)

cost = make_cost(n,k)
initial_d = make_initial_distribution(n,k)




#now, given the parameters, I have a transition probability matrix, but what do
#I actually need now?

#look at thesis

def make_D(n,k):
    ii = np.identity(n)
    D = np.zeros((n*k,n))
    count = -1
    for i in range(n):
        for a in range(k):
            count = count +1
            D[count] = ii[i]
    return(D)

def make_A(n,k,tpm,beta):
    count = -1
    A = np.zeros((n*k, n))
    #for i in range(n):
    #    for a in range(k):
    #        count = count +1
    #        for j in range(n):
    #            kron_delt = 0
    #            if i == j:
    #                kron_delt = 1
    #            A[count][j] = kron_delt-beta*tpm[count][j]
    D = make_D(n,k)
    A = D-beta*tpm
    return(A)

A = make_A(n,k,tpm,beta)
b = cost
c = initial_d

#I have A,b and c for the LP

#Now find out how to solve an LP in python

result = scipy.optimize.linprog(c = -initial_d, A_ub = A, b_ub = cost, bounds = (None,None))

opt_v = result.x
opt_val = -result.fun

print(opt_val)























#Now make A, b, c for linear program
